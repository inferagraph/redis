import type { CacheProvider } from '@inferagraph/core';
import { parseTTL } from '@inferagraph/core';
import { createClient } from 'redis';

import type { RedisCacheConfig, RedisLikeClient } from './types.js';

const DEFAULT_PREFIX = 'infera:cache:';

/**
 * Redis-backed `CacheProvider` (core 0.9.0+ wider shape).
 *
 * Honors the same defaults as the in-memory `lruCache`:
 * - Both `maxEntries` and `ttl` unset -> `(500, '24h')`.
 * - Only one set -> unset bound treated as no-limit.
 * - Both set -> both bounds enforced.
 * - `-1` / `'-1'` disables the corresponding bound.
 *
 * The class is exposed for direct use; most consumers should prefer the
 * {@link redisCacheProvider} factory which constructs the underlying redis
 * client when a `url` is supplied.
 */
export class RedisCacheProvider implements CacheProvider {
  private readonly prefix: string;
  private readonly indexKey: string;
  private readonly maxEntries: number;
  private readonly defaultTtlMs: number;
  private readonly client: RedisLikeClient;
  private readonly maskedUrl: string;
  private connectPromise: Promise<void> | undefined;

  constructor(config: RedisCacheConfig) {
    if (!config.client && !config.url) {
      throw new Error('RedisCacheProvider: either `url` or `client` must be provided');
    }

    this.prefix = config.prefix ?? DEFAULT_PREFIX;
    this.indexKey = `${this.prefix}__index`;

    const hasMaxEntries = config.maxEntries !== undefined;
    const hasTtl = config.ttl !== undefined;

    if (!hasMaxEntries && !hasTtl) {
      this.maxEntries = 500;
      this.defaultTtlMs = parseTTL('24h');
    } else {
      this.maxEntries = hasMaxEntries ? normalizeMaxEntries(config.maxEntries!) : -1;
      this.defaultTtlMs = hasTtl ? parseTTL(config.ttl!) : -1;
    }

    this.client =
      config.client ?? (createClient({ url: config.url! }) as unknown as RedisLikeClient);
    this.maskedUrl = config.url ? maskUrl(config.url) : '<pre-built client>';
  }

  async get(key: string): Promise<string | undefined> {
    await this.ensureConnected();
    const v = await this.client.get(this.dataKey(key));
    return v === null ? undefined : v;
  }

  async set(
    key: string,
    value: string,
    opts?: { ttlSeconds?: number },
  ): Promise<void> {
    await this.ensureConnected();
    const k = this.dataKey(key);

    const setOptions = this.resolveSetTtlOptions(opts);

    if (setOptions) {
      await this.client.set(k, value, setOptions);
    } else {
      await this.client.set(k, value);
    }

    if (this.maxEntries !== -1) {
      // Track insertion order in a ZSET; evict oldest when over capacity.
      await this.client.zAdd(this.indexKey, { score: Date.now(), value: key });
      const size = await this.client.zCard(this.indexKey);
      if (size > this.maxEntries) {
        const evictCount = size - this.maxEntries;
        const oldest = await this.client.zRange(this.indexKey, 0, evictCount - 1);
        if (oldest.length > 0) {
          await this.client.zRem(this.indexKey, oldest);
          await this.client.del(oldest.map((cacheKey) => this.dataKey(cacheKey)));
        }
      }
    }
  }

  async delete(key: string): Promise<void> {
    await this.ensureConnected();
    await this.client.del(this.dataKey(key));
    if (this.maxEntries !== -1) {
      await this.client.zRem(this.indexKey, key);
    }
  }

  async clear(): Promise<void> {
    await this.ensureConnected();
    const toDelete: string[] = [];
    // SCAN with MATCH is safe on large datasets; KEYS would block the server.
    // Deliberately scoped to `${prefix}*` so we never touch keys outside our
    // namespace (NOT FLUSHDB).
    for await (const k of this.client.scanIterator({ MATCH: `${this.prefix}*`, COUNT: 100 })) {
      toDelete.push(k);
    }
    if (toDelete.length > 0) {
      await this.client.del(toDelete);
    }
  }

  private dataKey(key: string): string {
    return `${this.prefix}${key}`;
  }

  /**
   * Decide which TTL (per-call vs construction default) applies to a SET.
   * Per-call wins. Returns `undefined` when no TTL applies (no EX/PX).
   */
  private resolveSetTtlOptions(
    opts: { ttlSeconds?: number } | undefined,
  ): { EX: number } | { PX: number } | undefined {
    if (opts?.ttlSeconds !== undefined) {
      // Per-call TTL is always seconds-precision per the contract.
      if (opts.ttlSeconds <= 0) return undefined;
      return { EX: Math.floor(opts.ttlSeconds) };
    }
    if (this.defaultTtlMs === -1) return undefined;
    return this.defaultTtlMs % 1000 === 0
      ? { EX: Math.floor(this.defaultTtlMs / 1000) }
      : { PX: this.defaultTtlMs };
  }

  private async ensureConnected(): Promise<void> {
    if (this.client.isOpen) return;
    if (!this.connectPromise) {
      this.connectPromise = (async () => {
        try {
          await this.client.connect();
        } catch (err) {
          // Reset so a future operation can retry. Re-throw so the caller sees it.
          this.connectPromise = undefined;
          // eslint-disable-next-line no-console
          console.warn(
            `[RedisCacheProvider] failed to connect to ${this.maskedUrl}: ${(err as Error).message}`,
          );
          throw err;
        }
      })();
    }
    return this.connectPromise;
  }
}

/**
 * Construct a {@link RedisCacheProvider}. Accepts either a pre-built `client`
 * or a `url`; when only `url` is provided, the factory builds the underlying
 * `redis` client internally so consumers don't import the SDK.
 */
export function redisCacheProvider(config: RedisCacheConfig): CacheProvider {
  return new RedisCacheProvider(config);
}

/**
 * @deprecated Use {@link redisCacheProvider} (or `RedisCacheProvider`)
 *  directly. Kept temporarily as an alias to ease migration off the old
 *  package name `@inferagraph/redis-cache-provider`.
 */
export function redisCache(config: RedisCacheConfig): CacheProvider {
  return new RedisCacheProvider(config);
}

function normalizeMaxEntries(value: number): number {
  if (value === -1) return -1;
  if (!Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
    throw new Error(
      `RedisCacheProvider: invalid maxEntries ${value}; expected a non-negative integer or -1`,
    );
  }
  return value;
}

function maskUrl(url: string): string {
  // Mask password component (between `:` and `@`) in a connection URL.
  // E.g. redis://default:hunter2@host:6379 -> redis://default:***@host:6379
  return url.replace(/(:\/\/[^:]+:)[^@]+(@)/, '$1***$2');
}
