# @inferagraph/redis

Redis-backed RAG primitives for [`@inferagraph/core`](https://github.com/inferagraph/core)'s `AIEngine`. One package, four pluggable stores plus a one-shot index provisioner:

| Concept | Class | Factory |
|---|---|---|
| LLM response cache | `RedisCacheProvider` | `redisCacheProvider` |
| Multi-turn chat memory | `RedisConversationStore` | `redisConversationStore` |
| Vector embedding storage (RediSearch HNSW) | `RedisVectorEmbeddingStore` | `redisVectorEmbeddingStore` |
| Inferred-edge overlay (RediSearch HNSW) | `RedisInferredEdgeStore` | `redisInferredEdgeStore` |
| One-time index bootstrap | — | `provisionRedisVectorIndex` |

The vector and inferred-edge stores require **Redis Stack** (RediSearch + RedisJSON). The cache and conversation stores work against any Redis 6+.

## Installation

```bash
pnpm add @inferagraph/redis @inferagraph/core redis
```

### Migrating from `@inferagraph/redis-cache-provider`

```bash
pnpm remove @inferagraph/redis-cache-provider
pnpm add @inferagraph/redis
```

The class names did not change. The legacy factory `redisCache` is retained as a deprecated alias of `redisCacheProvider`. The cache `CacheProvider` shape now also exposes `delete(key)` and accepts a per-call `{ ttlSeconds }` on `set` (per `@inferagraph/core@0.9.0`).

## Cache provider

```ts
import { AIEngine } from '@inferagraph/core';
import { redisCacheProvider } from '@inferagraph/redis';

const cache = redisCacheProvider({
  url: process.env.REDIS_URL ?? 'redis://localhost:6379',
  // Optional. If both maxEntries and ttl are unset, defaults to (500, '24h').
  maxEntries: 1000,
  ttl: '12h',
});

const engine = new AIEngine({ /* ... */ });
engine.setCache(cache);

// Wider 0.9.0 surface:
await cache.set('key', 'value', { ttlSeconds: 30 }); // per-call TTL wins
await cache.delete('key');                            // single-key delete
await cache.clear();                                  // SCAN+DEL by prefix (NOT FLUSHDB)
```

| Option | Required | Description |
|---|---|---|
| `url` | one of `url` / `client` | Redis connection URL (`redis://...` or `rediss://...`). |
| `client` | one of `url` / `client` | Pre-built node-redis client. When provided, `url` is ignored. |
| `prefix` | No | Key prefix (default `infera:cache:`). |
| `maxEntries` | No | Maximum entries to retain. `-1` disables. See defaults note below. |
| `ttl` | No | Construction-time TTL. Number (ms) or duration (`5m`, `2h`, `7d`, `1w`). `-1` / `'-1'` disables. |

**Defaults**: when both `maxEntries` and `ttl` are unset, the provider defaults to `(500, '24h')`. Per-call `{ ttlSeconds }` on `set()` always wins over the construction-time default.

## Conversation store

```ts
import { AIEngine } from '@inferagraph/core';
import { redisConversationStore } from '@inferagraph/redis';

const conversations = redisConversationStore({
  url: process.env.REDIS_URL,
  // Optional defaults shown:
  keyPrefix: 'inferagraph:conversation',
  ttlSeconds: 86_400,
});

const engine = new AIEngine({ /* ... */ });
engine.setConversationStore(conversations);
```

Storage layout: one Redis `LIST` per conversation, keyed by `<keyPrefix>:<conversationId>`. `appendTurn` `LPUSH`-es a JSON-serialized turn, `LTRIM 0 999` caps the per-conversation history at 1000 turns, and `EXPIRE <ttlSeconds>` refreshes the TTL on every append. Malformed entries are skipped with a `console.warn` rather than thrown.

## Vector embedding store (RediSearch)

```ts
import { redisVectorEmbeddingStore, provisionRedisVectorIndex } from '@inferagraph/redis';

// Once at deploy time — idempotent. Creates BOTH the embeddings index AND the
// inferred_edges index by default; pass alsoProvisionInferredEdges:false to opt out.
await provisionRedisVectorIndex({
  url: process.env.REDIS_URL,
  embeddingDimensions: 3072,           // matches text-embedding-3-large
});

const embeddings = redisVectorEmbeddingStore({
  url: process.env.REDIS_URL,
  embeddingDimensions: 3072,
});

// engine.setEmbeddingStore(embeddings);
```

Storage layout:

- `<keyPrefix>:embedding:<nodeId>` — HASH containing `nodeId`, `embedding` (binary `Float32` array), `embeddingHash`, `embeddingModel`, `embeddingVersion`, `embeddingGeneratedAt`.
- `<keyPrefix>:embeddings:idx` — RediSearch index over the hashes (HNSW vector field, COSINE distance).

`searchVector(query, { top })` issues `FT.SEARCH <idx> "*=>[KNN $top @embedding $vec AS score]"` and converts the returned distance to similarity (`1 - distance`) so the contract's "higher = more similar" holds.

## Inferred-edge store (RediSearch)

```ts
import { redisInferredEdgeStore } from '@inferagraph/redis';

const inferred = redisInferredEdgeStore({
  url: process.env.REDIS_URL,
  embeddingDimensions: 3072,
});

// aiEngine.setInferredEdgeStore(inferred);
```

Storage layout:

- `<keyPrefix>:inferred_edge:<sourceId>:<targetId>:<type>` — HASH with `id`, `sourceId`, `targetId`, `type`, `score`, `sources`, `reasoning`, `perSource`, optional `embedding`.
- `<keyPrefix>:inferred_edges:idx` — RediSearch index with TAG indexes on `sourceId`, `targetId`, `type` plus an HNSW vector index on `embedding`. `getAllForNode(nodeId)` queries by `(@sourceId:{nodeId}) | (@targetId:{nodeId})` so both directions hit at O(log n).

`set(edges)` SCAN-deletes all existing `<prefix>:inferred_edge:*` keys, then HSETs the new ones (bulk replace, per the `InferredEdgeStore` contract). `searchInferredEdges(query, top)` issues a KNN query against the same index.

## Provisioning

```ts
import { provisionRedisVectorIndex } from '@inferagraph/redis';

await provisionRedisVectorIndex({
  url: process.env.REDIS_URL,
  keyPrefix: 'inferagraph',                   // default
  embeddingDimensions: 3072,                  // default
  alsoProvisionInferredEdges: true,           // default
});
```

Idempotent: catches RediSearch's "Index already exists" and treats it as a no-op. Run on every deploy without fear of failures.

## Notes

- All four stores connect lazily on first operation, so constructing them is cheap and won't throw.
- The package targets the protocol (Redis 6+ with RediSearch for vector stores), not a specific provider — the local `redis://localhost:6379` and managed Redis Cloud / Elasticache / Upstash all work transparently.
- `clear()` on the cache uses `SCAN` (never `KEYS`), so it is safe on large datasets and never touches keys outside the configured `prefix`.

## License

MIT
