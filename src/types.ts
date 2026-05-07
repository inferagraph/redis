/**
 * Structural typings for the subset of the node-redis v4 client surface used
 * across this package. Declared here so every module can import the same
 * shape and tests can supply lightweight in-memory fakes without depending
 * on the concrete `redis` runtime.
 */

import type { CacheConfig } from '@inferagraph/core/data';

/**
 * Result row returned by `client.ft.search`. Mirrors the @redis/search shape
 * we actually consume — `total` count + `documents[].id` + `documents[].value`
 * (the parsed hash fields).
 */
export interface RedisFtSearchReply {
  total: number;
  documents: Array<{
    id: string;
    value: Record<string, unknown>;
  }>;
}

/**
 * Subset of the `ft.*` (RediSearch) command surface this package consumes.
 * Declared separately so callers wishing to stub only RediSearch can do so
 * without re-implementing the full client surface.
 */
export interface RedisFtCommands {
  create(
    indexName: string,
    schema: Record<string, unknown>,
    options?: Record<string, unknown>,
  ): Promise<string>;
  search(
    indexName: string,
    query: string,
    options?: Record<string, unknown>,
  ): Promise<RedisFtSearchReply>;
  dropIndex(indexName: string, options?: Record<string, unknown>): Promise<string>;
}

/**
 * Subset of the node-redis v4 hash command surface this package's vector
 * stores rely on (HSET / HGETALL / HDEL).
 */
export interface RedisHashCommands {
  hSet(
    key: string,
    fieldsOrObject: Record<string, string | number | Buffer>,
  ): Promise<number>;
  hGetAll(key: string): Promise<Record<string, string | Buffer>>;
}

/**
 * Minimal subset of the node-redis v4 client surface this package relies on.
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
  // Conversation list ops:
  lPush?(key: string, element: string | string[]): Promise<number>;
  lTrim?(key: string, start: number, stop: number): Promise<string>;
  lRange?(key: string, start: number, stop: number): Promise<string[]>;
  expire?(key: string, seconds: number): Promise<number>;
  // Hash ops (vector stores):
  hSet?: RedisHashCommands['hSet'];
  hGetAll?: RedisHashCommands['hGetAll'];
  // RediSearch namespace (vector stores).
  ft?: RedisFtCommands;
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
