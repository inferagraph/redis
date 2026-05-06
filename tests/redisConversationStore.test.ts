import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  RedisConversationStore,
  type RedisConversationLikeClient,
} from '../src/index.js';
import type { ConversationStore, ConversationTurn } from './__shims/coreInterfaces.js';

/**
 * Hand-rolled in-memory fake of the node-redis v4 surface
 * `RedisConversationStore` relies on (LIST + EXPIRE + DEL).
 *
 * Records every command so tests can assert behavior; exposes the underlying
 * map for whitebox assertions where useful.
 */
class FakeRedisList implements RedisConversationLikeClient {
  isOpen = true;
  // Each list value is the raw stored string. Newest at index 0 (LPUSH order).
  lists = new Map<string, string[]>();
  ttls = new Map<string, number>();
  commands: Array<{ cmd: string; args: unknown[] }> = [];

  async connect(): Promise<void> {
    this.isOpen = true;
  }

  async lPush(key: string, element: string | string[]): Promise<number> {
    this.commands.push({ cmd: 'LPUSH', args: [key, element] });
    const list = this.lists.get(key) ?? [];
    const elements = Array.isArray(element) ? element : [element];
    // node-redis lPush prepends each element in order, leaving the LAST
    // argument at the head. Match that behavior.
    for (const el of elements) {
      list.unshift(el);
    }
    this.lists.set(key, list);
    return list.length;
  }

  async lTrim(key: string, start: number, stop: number): Promise<string> {
    this.commands.push({ cmd: 'LTRIM', args: [key, start, stop] });
    const list = this.lists.get(key);
    if (!list) return 'OK';
    // Negative indexes count from the end (Redis semantics), but we only
    // need positive-index support for this provider. Keep the simple slice.
    const trimmed = list.slice(start, stop + 1);
    this.lists.set(key, trimmed);
    return 'OK';
  }

  async lRange(key: string, start: number, stop: number): Promise<string[]> {
    this.commands.push({ cmd: 'LRANGE', args: [key, start, stop] });
    const list = this.lists.get(key) ?? [];
    return list.slice(start, stop + 1);
  }

  async expire(key: string, seconds: number): Promise<number> {
    this.commands.push({ cmd: 'EXPIRE', args: [key, seconds] });
    if (!this.lists.has(key)) return 0;
    this.ttls.set(key, seconds);
    return 1;
  }

  async del(keys: string | string[]): Promise<number> {
    const arr = Array.isArray(keys) ? keys : [keys];
    this.commands.push({ cmd: 'DEL', args: arr });
    let deleted = 0;
    for (const k of arr) {
      if (this.lists.delete(k)) deleted += 1;
      this.ttls.delete(k);
    }
    return deleted;
  }
}

const turn = (overrides: Partial<ConversationTurn> = {}): ConversationTurn => ({
  role: 'user',
  content: 'hello',
  timestamp: 1_700_000_000_000,
  ...overrides,
});

describe('RedisConversationStore', () => {
  let mock: FakeRedisList;

  beforeEach(() => {
    mock = new FakeRedisList();
  });

  describe('appendTurn', () => {
    it('LPUSHes the JSON-serialized turn to <prefix>:<conversationId>', async () => {
      const store = new RedisConversationStore({ client: mock });
      const t = turn({ content: 'who lived in eden?' });

      await store.appendTurn('abc123', t);

      const lpush = mock.commands.find((c) => c.cmd === 'LPUSH');
      expect(lpush).toBeDefined();
      expect(lpush!.args[0]).toBe('inferagraph:conversation:abc123');
      expect(JSON.parse(lpush!.args[1] as string)).toEqual(t);
    });

    it('calls LTRIM 0 999 after LPUSH to bound list length', async () => {
      const store = new RedisConversationStore({ client: mock });

      await store.appendTurn('abc123', turn());

      const lpushIdx = mock.commands.findIndex((c) => c.cmd === 'LPUSH');
      const ltrimIdx = mock.commands.findIndex((c) => c.cmd === 'LTRIM');
      expect(lpushIdx).toBeGreaterThanOrEqual(0);
      expect(ltrimIdx).toBeGreaterThan(lpushIdx);
      const ltrim = mock.commands[ltrimIdx];
      expect(ltrim.args).toEqual(['inferagraph:conversation:abc123', 0, 999]);
    });

    it('calls EXPIRE with the configured ttlSeconds', async () => {
      const store = new RedisConversationStore({ client: mock, ttlSeconds: 3600 });

      await store.appendTurn('abc123', turn());

      const expire = mock.commands.find((c) => c.cmd === 'EXPIRE');
      expect(expire).toBeDefined();
      expect(expire!.args).toEqual(['inferagraph:conversation:abc123', 3600]);
    });

    it('defaults ttlSeconds to 86400 (24h)', async () => {
      const store = new RedisConversationStore({ client: mock });

      await store.appendTurn('abc123', turn());

      const expire = mock.commands.find((c) => c.cmd === 'EXPIRE');
      expect(expire!.args).toEqual(['inferagraph:conversation:abc123', 86400]);
    });

    it("defaults keyPrefix to 'inferagraph:conversation'", async () => {
      const store = new RedisConversationStore({ client: mock });

      await store.appendTurn('xyz', turn());

      expect(mock.lists.has('inferagraph:conversation:xyz')).toBe(true);
    });

    it('honors a custom keyPrefix', async () => {
      const store = new RedisConversationStore({
        client: mock,
        keyPrefix: 'biblegraph:conversation',
      });

      await store.appendTurn('xyz', turn());

      expect(mock.lists.has('biblegraph:conversation:xyz')).toBe(true);
      expect(mock.lists.has('inferagraph:conversation:xyz')).toBe(false);
    });
  });

  describe('getTurns', () => {
    it('calls LRANGE 0 limit-1 and returns parsed turns oldest -> newest', async () => {
      const store = new RedisConversationStore({ client: mock });
      // Seed in chronological order (oldest first) by appending.
      const t1 = turn({ content: 'first', timestamp: 1 });
      const t2 = turn({ role: 'assistant', content: 'second', timestamp: 2 });
      const t3 = turn({ content: 'third', timestamp: 3 });
      await store.appendTurn('abc', t1);
      await store.appendTurn('abc', t2);
      await store.appendTurn('abc', t3);
      mock.commands.length = 0;

      const result = await store.getTurns('abc', 10);

      const lrange = mock.commands.find((c) => c.cmd === 'LRANGE');
      expect(lrange).toBeDefined();
      expect(lrange!.args).toEqual(['inferagraph:conversation:abc', 0, 9]);
      // Oldest -> newest, regardless of LPUSH internal ordering.
      expect(result.map((r) => r.content)).toEqual(['first', 'second', 'third']);
    });

    it('returns an empty array when the key is missing', async () => {
      const store = new RedisConversationStore({ client: mock });

      const result = await store.getTurns('does-not-exist', 10);

      expect(result).toEqual([]);
    });

    it('skips malformed JSON entries with a console.warn', async () => {
      const store = new RedisConversationStore({ client: mock });
      const good = turn({ content: 'good' });
      // Seed directly: corrupt entry next to a valid one.
      mock.lists.set('inferagraph:conversation:abc', [
        JSON.stringify(good),
        '{not valid json',
      ]);

      const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const result = await store.getTurns('abc', 10);

      expect(result).toHaveLength(1);
      expect(result[0].content).toBe('good');
      expect(warn).toHaveBeenCalled();
      warn.mockRestore();
    });

    it('skips entries whose shape fails the ConversationTurn check', async () => {
      const store = new RedisConversationStore({ client: mock });
      const good = turn({ content: 'good' });
      // Seed: valid JSON but wrong shape (missing required keys / wrong types).
      mock.lists.set('inferagraph:conversation:abc', [
        JSON.stringify(good),
        JSON.stringify({ foo: 'bar' }),
        JSON.stringify({ role: 'system', content: 'x', timestamp: 1 }), // role not in allowed set
        JSON.stringify({ role: 'user', content: 42, timestamp: 1 }), // wrong content type
      ]);

      const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const result = await store.getTurns('abc', 10);

      expect(result).toHaveLength(1);
      expect(result[0].content).toBe('good');
      expect(warn).toHaveBeenCalled();
      warn.mockRestore();
    });
  });

  describe('clear', () => {
    it('DELs the keyed list', async () => {
      const store = new RedisConversationStore({ client: mock });
      await store.appendTurn('abc', turn());
      mock.commands.length = 0;

      await store.clear('abc');

      const del = mock.commands.find((c) => c.cmd === 'DEL');
      expect(del).toBeDefined();
      expect(del!.args).toEqual(['inferagraph:conversation:abc']);
      expect(mock.lists.has('inferagraph:conversation:abc')).toBe(false);
    });
  });

  describe('round-trip', () => {
    it('appendTurn N times -> getTurns returns the N turns oldest -> newest', async () => {
      const store = new RedisConversationStore({ client: mock });
      const turns: ConversationTurn[] = Array.from({ length: 5 }, (_, i) =>
        turn({
          role: i % 2 === 0 ? 'user' : 'assistant',
          content: `msg-${i}`,
          timestamp: 1000 + i,
          retrievedNodeIds: i % 2 === 1 ? [`node-${i}`] : undefined,
        }),
      );

      for (const t of turns) {
        await store.appendTurn('conv1', t);
      }

      const result = await store.getTurns('conv1', 100);
      expect(result).toEqual(turns);
    });
  });

  describe('contract', () => {
    it('is assignable to ConversationStore', () => {
      // Compile-time check: RedisConversationStore satisfies the core interface.
      const s: ConversationStore = new RedisConversationStore({ client: mock });
      expect(s).toBeDefined();
    });
  });
});
