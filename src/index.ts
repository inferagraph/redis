import type { CacheProvider, CacheConfig } from '@inferagraph/core';
import { parseTTL } from '@inferagraph/core';
import { createClient } from 'redis';

/**
 * Minimal subset of the node-redis v4 client surface this provider relies on.
 * Declared structurally so callers can pass any client (real or mocked) that
 * satisfies the shape — we never depend on concrete class identity.
 */
export interface RedisLikeClient {
  isOpen?: boolean;
  connect(): Promise<unknown>;
  get(key: string): Promise<string | null>;
  set(key: string, value: string, options?: { EX?: number; PX?: number }): Promise<unknown>;
  del(keys: string | string[]): Promise<number>;
  zAdd(
    key: string,
    member: { score: number; value: string } | { score: number; value: string }[],
  ): Promise<number>;
  zCard(key: string): Promise<number>;
  zRange(key: string, start: number, stop: number): Promise<string[]>;
  zRem(key: string, member: string | string[]): Promise<number>;
  scanIterator(options: { MATCH: string; COUNT?: number }): AsyncIterable<string>;
}

export interface RedisCacheConfig extends CacheConfig {
  /** Redis connection URL (e.g., 'redis://localhost:6379' or 'rediss://...'). */
  url?: string;
  /**
   * Key prefix to namespace this cache (default `infera:cache:`). Lets multiple
   * InferaGraph instances share a Redis instance without collisions.
   */
  prefix?: string;
  /**
   * Optional pre-built client. When provided, `url` is ignored. Useful for
   * tests and connection-pool reuse.
   */
  client?: RedisLikeClient;
}

const DEFAULT_PREFIX = 'infera:cache:';

/**
 * Redis-backed `CacheProvider` for `@inferagraph/core`'s `AIEngine`.
 *
 * Honors the same defaults as the in-memory `lruCache`:
 * - Both `maxEntries` and `ttl` unset → `(500, '24h')`.
 * - Only one set → unset bound treated as no-limit.
 * - Both set → both bounds enforced.
 * - `-1` / `'-1'` disables the corresponding bound.
 *
 * Connection is established lazily on first operation. Concurrent first-ops
 * share a single connect Promise.
 */
export function redisCache(config: RedisCacheConfig): CacheProvider {
  if (!config.client && !config.url) {
    throw new Error('redisCache: either `url` or `client` must be provided');
  }

  const prefix = config.prefix ?? DEFAULT_PREFIX;
  const indexKey = `${prefix}__index`;

  const hasMaxEntries = config.maxEntries !== undefined;
  const hasTtl = config.ttl !== undefined;

  let maxEntries: number; // -1 == no limit
  let ttlMs: number; // -1 == no limit
  if (!hasMaxEntries && !hasTtl) {
    maxEntries = 500;
    ttlMs = parseTTL('24h');
  } else {
    maxEntries = hasMaxEntries ? normalizeMaxEntries(config.maxEntries!) : -1;
    ttlMs = hasTtl ? parseTTL(config.ttl!) : -1;
  }

  // Lazy client construction so consumers don't pay a connection cost in the
  // sync constructor path. The provided `client` (if any) wins; otherwise we
  // build one from `url`.
  const client: RedisLikeClient =
    config.client ?? (createClient({ url: config.url! }) as unknown as RedisLikeClient);

  let connectPromise: Promise<void> | undefined;
  const ensureConnected = async (): Promise<void> => {
    if (client.isOpen) return;
    if (!connectPromise) {
      connectPromise = (async () => {
        try {
          await client.connect();
        } catch (err) {
          // Reset so a future operation can retry. Re-throw so the caller sees it.
          connectPromise = undefined;
          const masked = config.url ? maskUrl(config.url) : '<pre-built client>';
          // eslint-disable-next-line no-console
          console.warn(`[redisCache] failed to connect to ${masked}: ${(err as Error).message}`);
          throw err;
        }
      })();
    }
    return connectPromise;
  };

  const dataKey = (key: string): string => `${prefix}${key}`;

  return {
    async get(key: string): Promise<string | undefined> {
      await ensureConnected();
      const v = await client.get(dataKey(key));
      return v === null ? undefined : v;
    },

    async set(key: string, value: string): Promise<void> {
      await ensureConnected();
      const k = dataKey(key);

      const setOptions: { EX?: number; PX?: number } | undefined =
        ttlMs === -1
          ? undefined
          : ttlMs % 1000 === 0
            ? { EX: Math.floor(ttlMs / 1000) }
            : { PX: ttlMs };

      if (setOptions) {
        await client.set(k, value, setOptions);
      } else {
        await client.set(k, value);
      }

      if (maxEntries !== -1) {
        // Track insertion order in a ZSET; evict oldest when over capacity.
        await client.zAdd(indexKey, { score: Date.now(), value: key });
        const size = await client.zCard(indexKey);
        if (size > maxEntries) {
          const evictCount = size - maxEntries;
          const oldest = await client.zRange(indexKey, 0, evictCount - 1);
          if (oldest.length > 0) {
            await client.zRem(indexKey, oldest);
            await client.del(oldest.map((cacheKey) => dataKey(cacheKey)));
          }
        }
      }
    },

    async clear(): Promise<void> {
      await ensureConnected();
      const toDelete: string[] = [];
      // SCAN with MATCH is safe on large datasets; KEYS would block the server.
      for await (const k of client.scanIterator({ MATCH: `${prefix}*`, COUNT: 100 })) {
        toDelete.push(k);
      }
      if (toDelete.length > 0) {
        await client.del(toDelete);
      }
      // The index ZSET key starts with `${prefix}__index`, which matches
      // `${prefix}*`, so the SCAN above will already have included it. Calling
      // del again on a missing key would be a harmless no-op, but we skip it
      // to avoid an extra round-trip.
    },
  };
}

function normalizeMaxEntries(value: number): number {
  if (value === -1) return -1;
  if (!Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
    throw new Error(
      `redisCache: invalid maxEntries ${value}; expected a non-negative integer or -1`,
    );
  }
  return value;
}

function maskUrl(url: string): string {
  // Mask password component (between `:` and `@`) in a connection URL.
  // E.g. redis://default:hunter2@host:6379 -> redis://default:***@host:6379
  return url.replace(/(:\/\/[^:]+:)[^@]+(@)/, '$1***$2');
}

/* ------------------------------------------------------------------------ *
 * RedisConversationStore — implements `@inferagraph/core@^0.8.0`'s
 * ConversationStore for multi-turn chat memory.
 * ------------------------------------------------------------------------ */

/**
 * Subset of the node-redis v4 client surface `RedisConversationStore` needs.
 * Declared structurally so callers can pass any client (real or mocked) that
 * matches the shape — no concrete-class dependency.
 */
export interface RedisConversationLikeClient {
  isOpen?: boolean;
  connect(): Promise<unknown>;
  lPush(key: string, element: string | string[]): Promise<number>;
  lTrim(key: string, start: number, stop: number): Promise<string>;
  lRange(key: string, start: number, stop: number): Promise<string[]>;
  expire(key: string, seconds: number): Promise<number>;
  del(keys: string | string[]): Promise<number>;
}

/**
 * One conversation turn. Mirrors `@inferagraph/core@^0.8.0`'s
 * `ConversationTurn`. Re-declared here so this package's public surface
 * doesn't force consumers to import the type from core when constructing
 * the store — the runtime contract is what matters.
 */
export interface RedisConversationTurn {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  retrievedNodeIds?: string[];
}

export interface RedisConversationStoreConfig {
  /** Pre-built node-redis client. Required (matches the cache provider's `client` injection convention). */
  client: RedisConversationLikeClient;
  /** Key prefix for conversation lists. Default `inferagraph:conversation`. */
  keyPrefix?: string;
  /** TTL refreshed on every appendTurn. Default `86400` (24h). */
  ttlSeconds?: number;
}

const DEFAULT_CONVERSATION_PREFIX = 'inferagraph:conversation';
const DEFAULT_CONVERSATION_TTL_SECONDS = 86_400;
/** Hard cap on stored turns per conversation. LTRIM 0 (MAX-1). */
const MAX_TURNS_PER_CONVERSATION = 1000;
const ALLOWED_ROLES = new Set(['user', 'assistant']);

/**
 * Redis-backed `ConversationStore` for `@inferagraph/core@^0.8.0`'s `AIEngine`.
 *
 * Storage layout: one Redis LIST per conversation, keyed by
 * `<keyPrefix>:<conversationId>`. Each element is a JSON-serialized turn.
 * Newest turns sit at the head (LPUSH); reads pull the most-recent N via
 * LRANGE 0 N-1 and reverse so callers see oldest -> newest, matching LLM
 * conversation order. Each `appendTurn` refreshes a TTL via EXPIRE so
 * inactive conversations age out without manual cleanup.
 *
 * Defensive: malformed entries (corrupt JSON, wrong shape) are skipped with
 * a `console.warn` rather than thrown — one bad write must not poison the
 * entire conversation history.
 */
export class RedisConversationStore {
  private readonly client: RedisConversationLikeClient;
  private readonly keyPrefix: string;
  private readonly ttlSeconds: number;
  private connectPromise: Promise<void> | undefined;

  constructor(config: RedisConversationStoreConfig) {
    if (!config.client) {
      throw new Error('RedisConversationStore: `client` is required');
    }
    this.client = config.client;
    this.keyPrefix = config.keyPrefix ?? DEFAULT_CONVERSATION_PREFIX;
    this.ttlSeconds = config.ttlSeconds ?? DEFAULT_CONVERSATION_TTL_SECONDS;
  }

  async appendTurn(conversationId: string, turn: RedisConversationTurn): Promise<void> {
    await this.ensureConnected();
    const key = this.dataKey(conversationId);
    const payload = JSON.stringify(turn);
    await this.client.lPush(key, payload);
    await this.client.lTrim(key, 0, MAX_TURNS_PER_CONVERSATION - 1);
    await this.client.expire(key, this.ttlSeconds);
  }

  async getTurns(conversationId: string, limit: number): Promise<RedisConversationTurn[]> {
    if (limit <= 0) return [];
    await this.ensureConnected();
    const key = this.dataKey(conversationId);
    const raw = await this.client.lRange(key, 0, limit - 1);
    if (raw.length === 0) return [];
    const parsed: RedisConversationTurn[] = [];
    for (const entry of raw) {
      const turn = safeParseTurn(entry);
      if (turn) parsed.push(turn);
    }
    // LPUSH puts newest at head. Reverse so callers see oldest -> newest.
    parsed.reverse();
    return parsed;
  }

  async clear(conversationId: string): Promise<void> {
    await this.ensureConnected();
    await this.client.del([this.dataKey(conversationId)]);
  }

  private dataKey(conversationId: string): string {
    return `${this.keyPrefix}:${conversationId}`;
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
            `[RedisConversationStore] failed to connect: ${(err as Error).message}`,
          );
          throw err;
        }
      })();
    }
    return this.connectPromise;
  }
}

function safeParseTurn(raw: string): RedisConversationTurn | undefined {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn(
      `[RedisConversationStore] dropping malformed JSON turn: ${(err as Error).message}`,
    );
    return undefined;
  }
  if (!isConversationTurn(value)) {
    // eslint-disable-next-line no-console
    console.warn('[RedisConversationStore] dropping turn with unexpected shape');
    return undefined;
  }
  return value;
}

function isConversationTurn(value: unknown): value is RedisConversationTurn {
  if (!value || typeof value !== 'object') return false;
  const v = value as Record<string, unknown>;
  if (typeof v.role !== 'string' || !ALLOWED_ROLES.has(v.role)) return false;
  if (typeof v.content !== 'string') return false;
  if (typeof v.timestamp !== 'number' || !Number.isFinite(v.timestamp)) return false;
  if (v.retrievedNodeIds !== undefined) {
    if (!Array.isArray(v.retrievedNodeIds)) return false;
    if (!v.retrievedNodeIds.every((id) => typeof id === 'string')) return false;
  }
  return true;
}
