import { describe, it, expect, beforeEach } from 'vitest';
import {
  RedisCacheProvider,
  redisCacheProvider,
  type RedisLikeClient,
} from '../src/index.js';

/**
 * Spec-list tests for the WIDENED `CacheProvider` shape introduced by
 * `@inferagraph/core@0.9.0`:
 * - `set(key, value, opts?: { ttlSeconds?: number })`
 * - `delete(key)`
 * - `clear()` (already covered, but spec-list item #7 asserts SCAN+DEL).
 *
 * Plus factory checks: `redisCacheProvider({ client })` and
 * `redisCacheProvider({ url })`.
 */

class FakeRedis implements RedisLikeClient {
  isOpen = false;
  store = new Map<string, string>();
  zsets = new Map<string, { score: number; value: string }[]>();
  ttls = new Map<string, { type: 'EX' | 'PX'; value: number }>();
  commands: Array<{ cmd: string; args: unknown[] }> = [];

  async connect(): Promise<void> {
    this.isOpen = true;
  }
  async get(key: string): Promise<string | null> {
    this.commands.push({ cmd: 'GET', args: [key] });
    return this.store.has(key) ? this.store.get(key)! : null;
  }
  async set(
    key: string,
    value: string,
    options?: { EX?: number; PX?: number },
  ): Promise<unknown> {
    this.commands.push({ cmd: 'SET', args: [key, value, options] });
    this.store.set(key, value);
    if (options?.EX !== undefined) this.ttls.set(key, { type: 'EX', value: options.EX });
    else if (options?.PX !== undefined)
      this.ttls.set(key, { type: 'PX', value: options.PX });
    else this.ttls.delete(key);
    return 'OK';
  }
  async del(keys: string | string[]): Promise<number> {
    const arr = Array.isArray(keys) ? keys : [keys];
    this.commands.push({ cmd: 'DEL', args: arr });
    let n = 0;
    for (const k of arr) {
      if (this.store.delete(k)) n += 1;
      if (this.zsets.delete(k)) n += 1;
      this.ttls.delete(k);
    }
    return n;
  }
  async zAdd(
    key: string,
    member: { score: number; value: string } | { score: number; value: string }[],
  ): Promise<number> {
    this.commands.push({ cmd: 'ZADD', args: [key, member] });
    const list = this.zsets.get(key) ?? [];
    const members = Array.isArray(member) ? member : [member];
    let added = 0;
    for (const m of members) {
      const idx = list.findIndex((x) => x.value === m.value);
      if (idx >= 0) list[idx] = m;
      else {
        list.push(m);
        added += 1;
      }
    }
    list.sort((a, b) => a.score - b.score);
    this.zsets.set(key, list);
    return added;
  }
  async zCard(key: string): Promise<number> {
    return (this.zsets.get(key) ?? []).length;
  }
  async zRange(key: string, start: number, stop: number): Promise<string[]> {
    return (this.zsets.get(key) ?? []).slice(start, stop + 1).map((m) => m.value);
  }
  async zRem(key: string, member: string | string[]): Promise<number> {
    this.commands.push({ cmd: 'ZREM', args: [key, member] });
    const list = this.zsets.get(key);
    if (!list) return 0;
    const arr = Array.isArray(member) ? member : [member];
    const before = list.length;
    this.zsets.set(
      key,
      list.filter((m) => !arr.includes(m.value)),
    );
    return before - (this.zsets.get(key)?.length ?? 0);
  }
  async *scanIterator(options: { MATCH: string; COUNT?: number }): AsyncIterable<string> {
    this.commands.push({ cmd: 'SCAN', args: [options] });
    const match = options.MATCH;
    const prefix = match.endsWith('*') ? match.slice(0, -1) : match;
    for (const k of this.store.keys()) if (k.startsWith(prefix)) yield k;
    for (const k of this.zsets.keys()) if (k.startsWith(prefix)) yield k;
  }
}

describe('RedisCacheProvider — widened CacheProvider (core 0.9.0)', () => {
  let mock: FakeRedis;

  beforeEach(() => {
    mock = new FakeRedis();
  });

  describe('set with per-call ttlSeconds', () => {
    it('issues SET ... EX 30 when opts.ttlSeconds is 30', async () => {
      const cache = redisCacheProvider({ client: mock, maxEntries: -1, ttl: -1 });
      await cache.set('k', 'v', { ttlSeconds: 30 });
      const setCmds = mock.commands.filter((c) => c.cmd === 'SET');
      expect(setCmds).toHaveLength(1);
      expect(setCmds[0].args[2]).toEqual({ EX: 30 });
    });

    it('per-call ttlSeconds wins over construction-time default', async () => {
      const cache = redisCacheProvider({ client: mock, maxEntries: -1, ttl: '1h' });
      await cache.set('k', 'v', { ttlSeconds: 5 });
      const setCmds = mock.commands.filter((c) => c.cmd === 'SET');
      expect(setCmds[0].args[2]).toEqual({ EX: 5 });
    });

    it('omits EX/PX when ttlSeconds is 0 (per-call opt-out)', async () => {
      const cache = redisCacheProvider({ client: mock, maxEntries: -1, ttl: '1h' });
      await cache.set('k', 'v', { ttlSeconds: 0 });
      const setCmds = mock.commands.filter((c) => c.cmd === 'SET');
      // Per-call ttlSeconds <= 0 must opt out of any TTL even when a
      // construction-time default would otherwise apply.
      expect(setCmds[0].args[2]).toBeUndefined();
    });

    it('omits EX/PX when ttlSeconds is negative (per-call opt-out)', async () => {
      const cache = redisCacheProvider({ client: mock, maxEntries: -1, ttl: '1h' });
      await cache.set('k', 'v', { ttlSeconds: -5 });
      const setCmds = mock.commands.filter((c) => c.cmd === 'SET');
      expect(setCmds[0].args[2]).toBeUndefined();
    });
  });

  describe('set without opts uses construction-time default', () => {
    it('falls back to construction-time ttl when opts omitted', async () => {
      const cache = redisCacheProvider({ client: mock, maxEntries: -1, ttl: '5m' });
      await cache.set('k', 'v');
      const setCmds = mock.commands.filter((c) => c.cmd === 'SET');
      expect(setCmds[0].args[2]).toEqual({ EX: 5 * 60 });
    });
  });

  describe('set without any TTL omits EX', () => {
    it('omits EX/PX when ttl is -1 and no per-call opts', async () => {
      const cache = redisCacheProvider({ client: mock, maxEntries: -1, ttl: -1 });
      await cache.set('k', 'v');
      const setCmds = mock.commands.filter((c) => c.cmd === 'SET');
      expect(setCmds[0].args[2]).toBeUndefined();
    });
  });

  describe('delete(key)', () => {
    it('issues DEL <prefixed key>', async () => {
      const cache = redisCacheProvider({ client: mock, maxEntries: -1, ttl: -1 });
      await cache.set('k', 'v');
      mock.commands.length = 0;

      await cache.delete('k');

      const delCmds = mock.commands.filter((c) => c.cmd === 'DEL');
      expect(delCmds).toHaveLength(1);
      // The fake records `args` as the normalized array of keys; for a
      // single-key delete we expect exactly one entry equal to the namespaced key.
      expect(delCmds[0].args).toEqual(['infera:cache:k']);
      expect(mock.store.has('infera:cache:k')).toBe(false);
    });

    it('also drops the key from the eviction index when maxEntries is bounded', async () => {
      const cache = redisCacheProvider({ client: mock, maxEntries: 10, ttl: -1 });
      await cache.set('k', 'v');
      expect(mock.zsets.get('infera:cache:__index')?.length).toBe(1);

      await cache.delete('k');

      // Index entry removed.
      expect((mock.zsets.get('infera:cache:__index') ?? []).map((m) => m.value)).toEqual(
        [],
      );
    });

    it('is a no-op (no throw) when the key is missing', async () => {
      const cache = redisCacheProvider({ client: mock, maxEntries: -1, ttl: -1 });
      await expect(cache.delete('absent')).resolves.toBeUndefined();
    });
  });

  describe('clear()', () => {
    it('SCANs + DELs all prefixed keys (NOT FLUSHDB)', async () => {
      const cache = redisCacheProvider({ client: mock, maxEntries: 10, ttl: '1h' });
      await cache.set('a', '1');
      await cache.set('b', '2');
      // Seed a key OUTSIDE the prefix to prove it survives.
      mock.store.set('other:foo', 'bar');
      mock.commands.length = 0;

      await cache.clear();

      // SCAN was used (not KEYS), and DEL only targeted the prefixed keys.
      expect(mock.commands.some((c) => c.cmd === 'SCAN')).toBe(true);
      expect(mock.commands.every((c) => c.cmd !== 'FLUSHDB')).toBe(true);
      expect(mock.store.has('other:foo')).toBe(true);
      expect(mock.store.has('infera:cache:a')).toBe(false);
      expect(mock.store.has('infera:cache:b')).toBe(false);
    });
  });

  describe('redisCacheProvider factory', () => {
    it('accepts a pre-built client (escape hatch)', async () => {
      const cache = redisCacheProvider({ client: mock });
      await cache.set('k', 'v');
      expect(mock.isOpen).toBe(true);
    });

    it('accepts a url and constructs a client internally without throwing', () => {
      // We don't actually connect; constructor must not throw on URL-only path.
      expect(() => redisCacheProvider({ url: 'redis://localhost:6379' })).not.toThrow();
    });

    it('returns an instance that is also constructable as RedisCacheProvider directly', () => {
      const cache = new RedisCacheProvider({ client: mock });
      expect(typeof cache.set).toBe('function');
      expect(typeof cache.delete).toBe('function');
      expect(typeof cache.clear).toBe('function');
      expect(typeof cache.get).toBe('function');
    });
  });
});
