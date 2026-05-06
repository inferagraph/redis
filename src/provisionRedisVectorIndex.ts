import { createClient } from 'redis';

import type { RedisLikeClient } from './types.js';

/**
 * Configuration for {@link provisionRedisVectorIndex}. Provider-agnostic:
 * dimensions and key naming are constructor options. The default is to
 * provision BOTH the units (embeddings) index and the inferred-edges index;
 * pass `alsoProvisionInferredEdges: false` to skip the latter.
 */
export interface ProvisionRedisVectorIndexConfig {
  /** Pre-built node-redis client. One of `client` / `url` is required. */
  client?: RedisLikeClient;
  /** Redis connection URL. The function builds a client internally when supplied. */
  url?: string;
  /** Logical key namespace. Defaults to `'inferagraph'`. */
  keyPrefix?: string;
  /** Vector dimensionality. Defaults to `3072`. */
  embeddingDimensions?: number;
  /**
   * Whether to also provision the separate inferred-edges index. Defaults to
   * `true` so most consumers get both indexes from a single call.
   */
  alsoProvisionInferredEdges?: boolean;
}

const DEFAULT_KEY_PREFIX = 'inferagraph';
const DEFAULT_DIMENSIONS = 3072;

/**
 * Idempotent setup for the RediSearch vector indexes used by
 * {@link RedisVectorEmbeddingStore} and {@link RedisInferredEdgeStore}.
 *
 * Creates `<keyPrefix>:embeddings:idx` and (by default)
 * `<keyPrefix>:inferred_edges:idx`. If an index already exists, the function
 * catches the "index already exists" error and treats it as a no-op so this
 * call is safe to run on every deploy.
 *
 * Hosts call this once at deploy time before wiring the stores.
 */
export async function provisionRedisVectorIndex(
  config: ProvisionRedisVectorIndexConfig,
): Promise<void> {
  if (!config.client && !config.url) {
    throw new Error(
      'provisionRedisVectorIndex: either `url` or `client` must be provided',
    );
  }
  const client: RedisLikeClient =
    config.client ?? (createClient({ url: config.url! }) as unknown as RedisLikeClient);
  const keyPrefix = config.keyPrefix ?? DEFAULT_KEY_PREFIX;
  const dimensions = config.embeddingDimensions ?? DEFAULT_DIMENSIONS;
  const alsoEdges = config.alsoProvisionInferredEdges ?? true;

  if (!client.isOpen) {
    await client.connect();
  }
  if (!client.ft) {
    throw new Error('provisionRedisVectorIndex: client does not support ft.create');
  }

  await safeCreateIndex(client, `${keyPrefix}:embeddings:idx`, {
    nodeId: { type: 'TAG' },
    embeddingHash: { type: 'TAG' },
    embeddingModel: { type: 'TAG' },
    embeddingVersion: { type: 'TAG' },
    embedding: {
      type: 'VECTOR',
      ALGORITHM: 'HNSW',
      TYPE: 'FLOAT32',
      DIM: dimensions,
      DISTANCE_METRIC: 'COSINE',
    },
  }, {
    ON: 'HASH',
    PREFIX: { count: 1, prefix: `${keyPrefix}:embedding:` },
  });

  if (alsoEdges) {
    await safeCreateIndex(client, `${keyPrefix}:inferred_edges:idx`, {
      id: { type: 'TAG' },
      sourceId: { type: 'TAG' },
      targetId: { type: 'TAG' },
      type: { type: 'TAG' },
      score: { type: 'NUMERIC' },
      embedding: {
        type: 'VECTOR',
        ALGORITHM: 'HNSW',
        TYPE: 'FLOAT32',
        DIM: dimensions,
        DISTANCE_METRIC: 'COSINE',
      },
    }, {
      ON: 'HASH',
      PREFIX: { count: 1, prefix: `${keyPrefix}:inferred_edge:` },
    });
  }
}

async function safeCreateIndex(
  client: RedisLikeClient,
  indexName: string,
  schema: Record<string, unknown>,
  options: Record<string, unknown>,
): Promise<void> {
  try {
    await client.ft!.create(indexName, schema, options);
  } catch (err) {
    if (isIndexAlreadyExistsError(err)) return;
    throw err;
  }
}

function isIndexAlreadyExistsError(err: unknown): boolean {
  if (!err) return false;
  const msg =
    err instanceof Error
      ? err.message
      : typeof err === 'string'
        ? err
        : String((err as { message?: unknown })?.message ?? err);
  // RediSearch returns "Index already exists" (case varies by version).
  return /index\s+already\s+exists/i.test(msg);
}
