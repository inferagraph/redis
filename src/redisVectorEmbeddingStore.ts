import type {
  EmbeddingRecord,
  EmbeddingStore,
  NodeId,
  SearchVectorHit,
  SimilarHit,
  Vector,
} from '@inferagraph/core/data';
import { createClient } from 'redis';

import type { RedisLikeClient } from './types.js';

/**
 * Configuration for {@link RedisVectorEmbeddingStore}. Targets RediSearch
 * (Redis Stack); the store is provider-agnostic — vector dimensionality and
 * key naming are constructor options with neutral defaults so hosts can swap
 * embedding providers (OpenAI, Voyage, etc.) without forking this package.
 */
export interface RedisVectorEmbeddingStoreConfig {
  /** Pre-built node-redis client. One of `client` / `url` is required. */
  client?: RedisLikeClient;
  /** Redis connection URL. The factory builds a client internally when supplied. */
  url?: string;
  /**
   * Logical key namespace. Embedding hashes are stored at
   * `<keyPrefix>:embedding:<nodeId>`; the index name defaults to
   * `<keyPrefix>:embeddings:idx`. Defaults to `'inferagraph'`.
   */
  keyPrefix?: string;
  /** RediSearch index name. Defaults to `<keyPrefix>:embeddings:idx`. */
  indexName?: string;
  /** Vector dimensionality. Defaults to `3072` (matches `text-embedding-3-large`). */
  embeddingDimensions?: number;
}

const DEFAULT_KEY_PREFIX = 'inferagraph';
const DEFAULT_DIMENSIONS = 3072;

/**
 * Persistent {@link EmbeddingStore} backed by Redis Stack with a RediSearch
 * vector index over hash-stored embeddings.
 *
 * Storage layout:
 * - `<keyPrefix>:embedding:<nodeId>` — HASH containing
 *   `nodeId`, `embedding` (binary Float32Array), `embeddingHash`,
 *   `embeddingModel`, `embeddingVersion`, `embeddingGeneratedAt`.
 * - `<keyPrefix>:embeddings:idx` — RediSearch index over the hashes.
 *
 * The store also satisfies the optional `searchVector(queryVec, {top, container?})`
 * surface on `EmbeddingStore` so the engine's hybrid retrieval can call it
 * without a tier check.
 */
export class RedisVectorEmbeddingStore implements EmbeddingStore {
  private readonly client: RedisLikeClient;
  private readonly keyPrefix: string;
  private readonly indexName: string;
  /**
   * Vector dimensionality. Held on the instance for symmetry with
   * {@link provisionRedisVectorIndex} (single source of truth) — the read +
   * write paths don't need to know the dimension explicitly because
   * {@link vectorToBytes} packs whatever length the caller passes.
   */
  readonly dimensions: number;
  private connectPromise: Promise<void> | undefined;

  constructor(config: RedisVectorEmbeddingStoreConfig) {
    if (!config.client && !config.url) {
      throw new Error(
        'RedisVectorEmbeddingStore: either `url` or `client` must be provided',
      );
    }
    this.client =
      config.client ?? (createClient({ url: config.url! }) as unknown as RedisLikeClient);
    this.keyPrefix = config.keyPrefix ?? DEFAULT_KEY_PREFIX;
    this.indexName = config.indexName ?? `${this.keyPrefix}:embeddings:idx`;
    this.dimensions = config.embeddingDimensions ?? DEFAULT_DIMENSIONS;
  }

  /** Lookup keyed by `(nodeId, model, modelVersion, contentHash)`. */
  async get(
    nodeId: NodeId,
    model: string,
    modelVersion: string,
    contentHash: string,
  ): Promise<EmbeddingRecord | undefined> {
    await this.ensureConnected();
    if (!this.client.hGetAll) {
      throw new Error('RedisVectorEmbeddingStore: client does not support hGetAll');
    }
    const key = this.embeddingKey(nodeId);
    const hash = await this.client.hGetAll(key);
    if (!hash || Object.keys(hash).length === 0) return undefined;

    const storedModel = bufferToString(hash.embeddingModel);
    const storedVersion = bufferToString(hash.embeddingVersion);
    const storedHash = bufferToString(hash.embeddingHash);
    if (storedModel !== model) return undefined;
    if (storedVersion !== modelVersion) return undefined;
    if (storedHash !== contentHash) return undefined;

    const vectorBytes = hash.embedding;
    if (vectorBytes === undefined) return undefined;
    return {
      nodeId,
      vector: bytesToVector(vectorBytes),
      meta: {
        model: storedModel,
        modelVersion: storedVersion,
        contentHash: storedHash,
        generatedAt: bufferToString(hash.embeddingGeneratedAt) ?? '',
      },
    };
  }

  /**
   * Persist an embedding as a HASH; the RediSearch index automatically picks
   * it up because the index is configured with `PREFIX 1 <keyPrefix>:embedding:`.
   */
  async set(record: EmbeddingRecord): Promise<void> {
    await this.ensureConnected();
    if (!this.client.hSet) {
      throw new Error('RedisVectorEmbeddingStore: client does not support hSet');
    }
    const key = this.embeddingKey(record.nodeId);
    await this.client.hSet(key, {
      nodeId: record.nodeId,
      embedding: vectorToBytes(record.vector),
      embeddingModel: record.meta.model,
      embeddingVersion: record.meta.modelVersion,
      embeddingHash: record.meta.contentHash,
      embeddingGeneratedAt: record.meta.generatedAt,
    });
  }

  /**
   * Linear-scan-style similarity is satisfied by delegating to
   * {@link searchVector}, which uses the vector index. Model + version scope
   * filters are intentionally honored at write time (each entry stores its
   * own metadata), so this method ignores them.
   */
  async similar(
    queryVector: Vector,
    k: number,
    _model?: string,
    _modelVersion?: string,
  ): Promise<SimilarHit[]> {
    void _model;
    void _modelVersion;
    const hits = await this.searchVector(queryVector, { top: k });
    return hits.map((h) => ({ nodeId: h.nodeId, score: h.score }));
  }

  async clear(): Promise<void> {
    await this.ensureConnected();
    const keys: string[] = [];
    for await (const k of this.client.scanIterator({
      MATCH: `${this.keyPrefix}:embedding:*`,
      COUNT: 100,
    })) {
      keys.push(k);
    }
    if (keys.length > 0) {
      await this.client.del(keys);
    }
  }

  /**
   * Vector-native top-K via RediSearch KNN.
   *
   * Issues `FT.SEARCH <idx> "*=>[KNN $top @embedding $vec AS score]"
   * PARAMS 4 vec <bytes> top <K> SORTBY score ASC RETURN 1 nodeId DIALECT 2`.
   * RediSearch returns the COSINE distance (lower = more similar); we convert
   * to similarity (`1 - distance`) so the contract's "higher = more similar"
   * holds.
   */
  async searchVector(
    queryEmbedding: Vector,
    opts: { top: number; container?: 'units' | 'inferred_edges' },
  ): Promise<SearchVectorHit[]> {
    await this.ensureConnected();
    if (!this.client.ft) {
      throw new Error('RedisVectorEmbeddingStore: client does not support ft.search');
    }
    // The default index covers units; inferred-edges callers should use the
    // dedicated InferredEdgeStore. Honor the `container` option only as a
    // forward-compatibility hint — when set to 'inferred_edges' the caller
    // would normally route to RedisInferredEdgeStore.searchInferredEdges.
    void opts.container;

    const reply = await this.client.ft.search(
      this.indexName,
      `*=>[KNN $top @embedding $vec AS score]`,
      {
        PARAMS: {
          vec: vectorToBytes(queryEmbedding),
          top: opts.top,
        },
        SORTBY: { BY: 'score', DIRECTION: 'ASC' },
        RETURN: ['nodeId', 'score'],
        DIALECT: 2,
      },
    );

    return (reply?.documents ?? []).map((doc) => {
      const value = doc.value as Record<string, unknown>;
      const distance = parseFloat(String(value.score ?? '0'));
      const similarity = Number.isFinite(distance) ? 1 - distance : 0;
      return {
        nodeId: bufferToString(value.nodeId) ?? doc.id,
        score: similarity,
      };
    });
  }

  private embeddingKey(nodeId: NodeId): string {
    return `${this.keyPrefix}:embedding:${nodeId}`;
  }

  private async ensureConnected(): Promise<void> {
    if (this.client.isOpen) return;
    if (!this.connectPromise) {
      this.connectPromise = (async () => {
        try {
          await this.client.connect();
        } catch (err) {
          this.connectPromise = undefined;
          // eslint-disable-next-line no-console
          console.warn(
            `[RedisVectorEmbeddingStore] failed to connect: ${(err as Error).message}`,
          );
          throw err;
        }
      })();
    }
    return this.connectPromise;
  }
}

/**
 * Construct a {@link RedisVectorEmbeddingStore}. Accepts a pre-built `client`
 * OR a `url`; in the latter case the factory builds the underlying redis
 * client internally so consumers don't import the SDK directly.
 */
export function redisVectorEmbeddingStore(
  config: RedisVectorEmbeddingStoreConfig,
): EmbeddingStore {
  return new RedisVectorEmbeddingStore(config);
}

/**
 * Convert a `Vector` to a Float32Array packed into a Node `Buffer` — the
 * binary form RediSearch expects for a `VECTOR FLOAT32` field.
 */
export function vectorToBytes(vector: Vector): Buffer {
  const arr = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i++) arr[i] = vector[i];
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength);
}

/**
 * Decode the binary `embedding` field of a RediSearch hash back into a
 * plain `Vector` (`number[]`). Accepts either a `Buffer` (real client) or a
 * `string` (some serializers; treated as latin1 bytes).
 */
export function bytesToVector(value: string | Buffer): Vector {
  const buf = typeof value === 'string' ? Buffer.from(value, 'binary') : value;
  const view = new Float32Array(
    buf.buffer,
    buf.byteOffset,
    Math.floor(buf.byteLength / 4),
  );
  const out: number[] = new Array(view.length);
  for (let i = 0; i < view.length; i++) out[i] = view[i];
  return out;
}

function bufferToString(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'string') return value;
  if (value instanceof Buffer) return value.toString('utf8');
  return String(value);
}
