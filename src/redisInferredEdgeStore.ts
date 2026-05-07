import type {
  InferredEdge,
  InferredEdgeSource,
  InferredEdgeStore,
  NodeId,
  SearchVectorHit,
  Vector,
} from '@inferagraph/core/data';
import { createClient } from 'redis';

import type { RedisLikeClient } from './types.js';
import { vectorToBytes } from './redisVectorEmbeddingStore.js';

/**
 * Configuration for {@link RedisInferredEdgeStore}. Provider-agnostic — the
 * vector dimensionality and key naming are constructor options with neutral
 * defaults.
 */
export interface RedisInferredEdgeStoreConfig {
  /** Pre-built client. One of `client` / `url` is required. */
  client?: RedisLikeClient;
  /** Redis connection URL. Factory builds the client internally when supplied. */
  url?: string;
  /** Logical key namespace. Defaults to `'inferagraph'`. */
  keyPrefix?: string;
  /** RediSearch index name. Defaults to `<keyPrefix>:inferred_edges:idx`. */
  indexName?: string;
  /** Vector dimensionality. Defaults to `3072`. */
  embeddingDimensions?: number;
}

const DEFAULT_KEY_PREFIX = 'inferagraph';
const DEFAULT_DIMENSIONS = 3072;

/**
 * Persistent {@link InferredEdgeStore} backed by a separate RediSearch index
 * over edge HASHes.
 *
 * Storage layout:
 * - `<keyPrefix>:inferred_edge:<sourceId>:<targetId>:<type>` — HASH with
 *   `id`, `sourceId`, `targetId`, `type`, `score`, `sources`, `reasoning`,
 *   `embedding` (optional, Float32 binary).
 * - `<keyPrefix>:inferred_edges:idx` — RediSearch index with TAG indexes on
 *   `sourceId` and `targetId` plus a HNSW vector index on `embedding`.
 */
export class RedisInferredEdgeStore implements InferredEdgeStore {
  private readonly client: RedisLikeClient;
  private readonly keyPrefix: string;
  private readonly indexName: string;
  /**
   * Vector dimensionality. Held on the instance for symmetry with
   * {@link provisionRedisVectorIndex} — read + write paths use whatever
   * length the caller passes via {@link vectorToBytes}.
   */
  readonly dimensions: number;
  private connectPromise: Promise<void> | undefined;

  constructor(config: RedisInferredEdgeStoreConfig) {
    if (!config.client && !config.url) {
      throw new Error(
        'RedisInferredEdgeStore: either `url` or `client` must be provided',
      );
    }
    this.client =
      config.client ?? (createClient({ url: config.url! }) as unknown as RedisLikeClient);
    this.keyPrefix = config.keyPrefix ?? DEFAULT_KEY_PREFIX;
    this.indexName = config.indexName ?? `${this.keyPrefix}:inferred_edges:idx`;
    this.dimensions = config.embeddingDimensions ?? DEFAULT_DIMENSIONS;
  }

  async get(sourceId: NodeId, targetId: NodeId): Promise<InferredEdge | undefined> {
    await this.ensureConnected();
    if (!this.client.ft) {
      throw new Error('RedisInferredEdgeStore: client does not support ft.search');
    }
    const reply = await this.client.ft.search(
      this.indexName,
      `@sourceId:{${escapeTag(sourceId)}} @targetId:{${escapeTag(targetId)}}`,
      { LIMIT: { from: 0, size: 1 }, DIALECT: 2 },
    );
    if (!reply || (reply.total ?? 0) === 0 || reply.documents.length === 0) {
      return undefined;
    }
    return docToEdge(reply.documents[0].value);
  }

  async getAllForNode(nodeId: NodeId): Promise<InferredEdge[]> {
    await this.ensureConnected();
    if (!this.client.ft) {
      throw new Error('RedisInferredEdgeStore: client does not support ft.search');
    }
    const tag = escapeTag(nodeId);
    const reply = await this.client.ft.search(
      this.indexName,
      `(@sourceId:{${tag}}) | (@targetId:{${tag}})`,
      { LIMIT: { from: 0, size: 10_000 }, DIALECT: 2 },
    );
    return (reply?.documents ?? []).map((d) => docToEdge(d.value));
  }

  async getAll(): Promise<InferredEdge[]> {
    await this.ensureConnected();
    if (!this.client.ft) {
      throw new Error('RedisInferredEdgeStore: client does not support ft.search');
    }
    const reply = await this.client.ft.search(this.indexName, '*', {
      LIMIT: { from: 0, size: 10_000 },
      DIALECT: 2,
    });
    return (reply?.documents ?? []).map((d) => docToEdge(d.value));
  }

  /**
   * Bulk-replace the entire stored set. SCAN existing edge keys, DEL them in
   * one batch, then HSET the new entries. Idempotent on `(sourceId,
   * targetId, type)` collisions because the key encodes those three values
   * (last write wins, per the contract).
   */
  async set(edges: ReadonlyArray<InferredEdge>): Promise<void> {
    await this.ensureConnected();
    if (!this.client.hSet) {
      throw new Error('RedisInferredEdgeStore: client does not support hSet');
    }
    await this.deleteAllEdgeKeys();
    for (const edge of edges) {
      const key = this.edgeKey(edge.sourceId, edge.targetId, edge.type);
      await this.client.hSet(key, edgeToHashFields(edge));
    }
  }

  async clear(): Promise<void> {
    await this.ensureConnected();
    await this.deleteAllEdgeKeys();
  }

  /**
   * Vector-native top-K against the inferred-edges index. Uses the same
   * `KNN` syntax as {@link RedisVectorEmbeddingStore.searchVector}; converts
   * the returned distance to similarity (`1 - distance`) so the contract's
   * "higher = more similar" holds.
   */
  async searchInferredEdges(
    queryEmbedding: Vector,
    top: number,
  ): Promise<SearchVectorHit[]> {
    await this.ensureConnected();
    if (!this.client.ft) {
      throw new Error('RedisInferredEdgeStore: client does not support ft.search');
    }
    const reply = await this.client.ft.search(
      this.indexName,
      `*=>[KNN $top @embedding $vec AS score]`,
      {
        PARAMS: {
          vec: vectorToBytes(queryEmbedding),
          top,
        },
        SORTBY: { BY: 'score', DIRECTION: 'ASC' },
        RETURN: ['id', 'score'],
        DIALECT: 2,
      },
    );
    return (reply?.documents ?? []).map((doc) => {
      const value = doc.value as Record<string, unknown>;
      const distance = parseFloat(String(value.score ?? '0'));
      const similarity = Number.isFinite(distance) ? 1 - distance : 0;
      const id = bufferToString(value.id) ?? doc.id;
      return { nodeId: id, score: similarity };
    });
  }

  private edgeKey(sourceId: NodeId, targetId: NodeId, type: string): string {
    return `${this.keyPrefix}:inferred_edge:${sourceId}:${targetId}:${type}`;
  }

  private async deleteAllEdgeKeys(): Promise<void> {
    const keys: string[] = [];
    for await (const k of this.client.scanIterator({
      MATCH: `${this.keyPrefix}:inferred_edge:*`,
      COUNT: 100,
    })) {
      keys.push(k);
    }
    if (keys.length > 0) {
      await this.client.del(keys);
    }
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
            `[RedisInferredEdgeStore] failed to connect: ${(err as Error).message}`,
          );
          throw err;
        }
      })();
    }
    return this.connectPromise;
  }
}

/**
 * Construct a {@link RedisInferredEdgeStore}. Accepts a pre-built `client`
 * OR a `url`; in the latter case the factory builds the redis client
 * internally.
 */
export function redisInferredEdgeStore(
  config: RedisInferredEdgeStoreConfig,
): InferredEdgeStore {
  return new RedisInferredEdgeStore(config);
}

function edgeToHashFields(edge: InferredEdge): Record<string, string | number | Buffer> {
  const fields: Record<string, string | number | Buffer> = {
    id: edgeId(edge),
    sourceId: edge.sourceId,
    targetId: edge.targetId,
    type: edge.type,
    score: edge.score,
    sources: edge.sources.join(','),
  };
  if (edge.reasoning !== undefined) fields.reasoning = edge.reasoning;
  if (edge.perSource !== undefined) {
    fields.perSource = JSON.stringify(edge.perSource);
  }
  return fields;
}

function edgeId(edge: InferredEdge): string {
  return `${edge.sourceId}-${edge.targetId}-${edge.type}`;
}

function docToEdge(value: Record<string, unknown>): InferredEdge {
  const sources = parseSources(bufferToString(value.sources));
  const perSourceRaw = bufferToString(value.perSource);
  let perSource: InferredEdge['perSource'];
  if (perSourceRaw) {
    try {
      perSource = JSON.parse(perSourceRaw) as InferredEdge['perSource'];
    } catch {
      perSource = undefined;
    }
  }
  const reasoning = bufferToString(value.reasoning);
  return {
    sourceId: bufferToString(value.sourceId) ?? '',
    targetId: bufferToString(value.targetId) ?? '',
    type: bufferToString(value.type) ?? '',
    score: parseFloat(String(value.score ?? '0')),
    sources,
    reasoning,
    perSource,
  };
}

function parseSources(raw: string | undefined): InferredEdgeSource[] {
  if (!raw) return [];
  return raw
    .split(',')
    .filter((s) => s.length > 0)
    .filter(
      (s): s is InferredEdgeSource => s === 'graph' || s === 'embedding' || s === 'llm',
    );
}

function escapeTag(value: string): string {
  // RediSearch TAG values escape `,` `.` `<` `>` `{` `}` `[` `]` `\"` `'` `:` `;` `!` `@` `#` `$` `%` `^` `&` `*` `(` `)` `-` `+` `=` `~` `|` `\` and whitespace.
  return value.replace(/([,.<>{}[\]"':;!@#$%^&*()\-+=~|\\\s])/g, '\\$1');
}

function bufferToString(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'string') return value;
  if (value instanceof Buffer) return value.toString('utf8');
  return String(value);
}
