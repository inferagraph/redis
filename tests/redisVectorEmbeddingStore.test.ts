import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  RedisVectorEmbeddingStore,
  redisVectorEmbeddingStore,
  vectorToBytes,
  bytesToVector,
  type RedisLikeClient,
} from '../src/index.js';
import type { EmbeddingRecord } from '@inferagraph/core/data';

/**
 * Hand-rolled in-memory fake of the node-redis v4 surface this store uses:
 * HSET / HGETALL / SCAN / DEL / FT.SEARCH.
 */
class FakeRedisHashSearch implements RedisLikeClient {
  isOpen = true;
  hashes = new Map<string, Record<string, string | Buffer>>();
  commands: Array<{ cmd: string; args: unknown[] }> = [];
  // FT.SEARCH responses are queued; tests can `searchReplies.push(reply)`.
  searchReplies: Array<{ total: number; documents: { id: string; value: Record<string, unknown> }[] }> = [];

  async connect(): Promise<void> {
    this.isOpen = true;
  }
  async get(): Promise<string | null> {
    return null;
  }
  async set(): Promise<unknown> {
    return 'OK';
  }
  async del(keys: string | string[]): Promise<number> {
    const arr = Array.isArray(keys) ? keys : [keys];
    this.commands.push({ cmd: 'DEL', args: arr });
    let n = 0;
    for (const k of arr) {
      if (this.hashes.delete(k)) n += 1;
    }
    return n;
  }
  async zAdd(): Promise<number> {
    return 0;
  }
  async zCard(): Promise<number> {
    return 0;
  }
  async zRange(): Promise<string[]> {
    return [];
  }
  async zRem(): Promise<number> {
    return 0;
  }
  async *scanIterator(options: { MATCH: string; COUNT?: number }): AsyncIterable<string> {
    this.commands.push({ cmd: 'SCAN', args: [options] });
    const match = options.MATCH;
    const prefix = match.endsWith('*') ? match.slice(0, -1) : match;
    for (const k of this.hashes.keys()) if (k.startsWith(prefix)) yield k;
  }

  async hSet(
    key: string,
    fields: Record<string, string | number | Buffer>,
  ): Promise<number> {
    this.commands.push({ cmd: 'HSET', args: [key, fields] });
    const stored: Record<string, string | Buffer> = {};
    for (const [k, v] of Object.entries(fields)) {
      stored[k] = v instanceof Buffer ? v : String(v);
    }
    this.hashes.set(key, stored);
    return Object.keys(stored).length;
  }

  async hGetAll(key: string): Promise<Record<string, string | Buffer>> {
    this.commands.push({ cmd: 'HGETALL', args: [key] });
    return this.hashes.get(key) ?? {};
  }

  ft = {
    create: async (
      indexName: string,
      schema: Record<string, unknown>,
      options?: Record<string, unknown>,
    ): Promise<string> => {
      this.commands.push({ cmd: 'FT.CREATE', args: [indexName, schema, options] });
      return 'OK';
    },
    search: async (
      indexName: string,
      query: string,
      options?: Record<string, unknown>,
    ): Promise<{ total: number; documents: { id: string; value: Record<string, unknown> }[] }> => {
      this.commands.push({ cmd: 'FT.SEARCH', args: [indexName, query, options] });
      return this.searchReplies.shift() ?? { total: 0, documents: [] };
    },
    dropIndex: async (indexName: string): Promise<string> => {
      this.commands.push({ cmd: 'FT.DROPINDEX', args: [indexName] });
      return 'OK';
    },
  };
}

const sampleRecord = (overrides: Partial<EmbeddingRecord> = {}): EmbeddingRecord => ({
  nodeId: 'node-1',
  vector: [0.1, 0.2, 0.3],
  meta: {
    model: 'text-embedding-3-large',
    modelVersion: '1',
    contentHash: 'abc123',
    generatedAt: '2026-05-06T00:00:00.000Z',
  },
  ...overrides,
});

describe('RedisVectorEmbeddingStore', () => {
  let mock: FakeRedisHashSearch;

  beforeEach(() => {
    mock = new FakeRedisHashSearch();
  });

  describe('set', () => {
    it('HSETs the embedding hash with all required fields at <keyPrefix>:embedding:<nodeId>', async () => {
      const store = redisVectorEmbeddingStore({ client: mock });
      const rec = sampleRecord();
      await store.set(rec);

      const hsetCmd = mock.commands.find((c) => c.cmd === 'HSET');
      expect(hsetCmd).toBeDefined();
      expect(hsetCmd!.args[0]).toBe('inferagraph:embedding:node-1');

      const stored = mock.hashes.get('inferagraph:embedding:node-1')!;
      expect(stored.nodeId).toBe('node-1');
      expect(stored.embeddingModel).toBe('text-embedding-3-large');
      expect(stored.embeddingVersion).toBe('1');
      expect(stored.embeddingHash).toBe('abc123');
      expect(stored.embeddingGeneratedAt).toBe('2026-05-06T00:00:00.000Z');
      // The embedding field must be the raw Float32 binary buffer.
      expect(stored.embedding).toBeInstanceOf(Buffer);
      const decoded = bytesToVector(stored.embedding as Buffer);
      expect(decoded.map((v) => Number(v.toFixed(6)))).toEqual([0.1, 0.2, 0.3].map((v) => Number(v.toFixed(6))));
    });
  });

  describe('searchVector', () => {
    it('issues FT.SEARCH with KNN syntax and DIALECT 2', async () => {
      const store = redisVectorEmbeddingStore({ client: mock });
      mock.searchReplies.push({ total: 0, documents: [] });

      await store.searchVector([0.1, 0.2, 0.3], { top: 5 });

      const search = mock.commands.find((c) => c.cmd === 'FT.SEARCH');
      expect(search).toBeDefined();
      expect(search!.args[0]).toBe('inferagraph:embeddings:idx');
      const query = search!.args[1] as string;
      expect(query).toMatch(/KNN \$top @embedding \$vec AS score/);
      const opts = search!.args[2] as Record<string, unknown>;
      expect(opts.DIALECT).toBe(2);
      expect((opts.PARAMS as Record<string, unknown>).top).toBe(5);
      expect((opts.PARAMS as Record<string, unknown>).vec).toBeInstanceOf(Buffer);
    });

    it('returns hits with similarity = 1 - distance and correct nodeIds', async () => {
      const store = redisVectorEmbeddingStore({ client: mock });
      // Distances 0.1, 0.4, 0.9 -> similarities 0.9, 0.6, 0.1.
      mock.searchReplies.push({
        total: 3,
        documents: [
          { id: 'inferagraph:embedding:n1', value: { nodeId: 'n1', score: '0.1' } },
          { id: 'inferagraph:embedding:n2', value: { nodeId: 'n2', score: '0.4' } },
          { id: 'inferagraph:embedding:n3', value: { nodeId: 'n3', score: '0.9' } },
        ],
      });

      const hits = await store.searchVector([0.1, 0.2], { top: 3 });

      expect(hits).toHaveLength(3);
      expect(hits[0]).toEqual({ nodeId: 'n1', score: 0.9 });
      expect(hits[1].nodeId).toBe('n2');
      expect(hits[1].score).toBeCloseTo(0.6, 6);
      expect(hits[2]).toEqual({ nodeId: 'n3', score: expect.closeTo(0.1, 6) as unknown as number });
    });
  });

  describe('get', () => {
    it('returns the record when stored model+version+hash match', async () => {
      const store = redisVectorEmbeddingStore({ client: mock });
      await store.set(sampleRecord());

      const result = await store.get('node-1', 'text-embedding-3-large', '1', 'abc123');

      expect(result).toBeDefined();
      expect(result!.nodeId).toBe('node-1');
      expect(result!.meta.model).toBe('text-embedding-3-large');
    });

    it('returns undefined when stored hash differs (cache miss on edit)', async () => {
      const store = redisVectorEmbeddingStore({ client: mock });
      await store.set(sampleRecord());

      const result = await store.get('node-1', 'text-embedding-3-large', '1', 'different');
      expect(result).toBeUndefined();
    });

    it('returns undefined when nothing is stored', async () => {
      const store = redisVectorEmbeddingStore({ client: mock });
      const result = await store.get('absent', 'm', 'v', 'h');
      expect(result).toBeUndefined();
    });
  });

  describe('factory', () => {
    it('accepts a pre-built client', () => {
      expect(() => redisVectorEmbeddingStore({ client: mock })).not.toThrow();
    });
    it('accepts a url and constructs a client internally', () => {
      expect(() =>
        redisVectorEmbeddingStore({ url: 'redis://localhost:6379' }),
      ).not.toThrow();
    });
    it('exposes RedisVectorEmbeddingStore class for direct construction', () => {
      const s = new RedisVectorEmbeddingStore({ client: mock });
      expect(typeof s.set).toBe('function');
      expect(typeof s.get).toBe('function');
      expect(typeof s.searchVector).toBe('function');
    });
  });

  describe('vectorToBytes / bytesToVector', () => {
    it('round-trips a vector through Float32 binary', () => {
      const original = [0.1, 0.2, 0.3, 0.4, 0.5];
      const bytes = vectorToBytes(original);
      expect(bytes).toBeInstanceOf(Buffer);
      const restored = bytesToVector(bytes);
      for (let i = 0; i < original.length; i++) {
        expect(restored[i]).toBeCloseTo(original[i], 6);
      }
    });

    it('decodes a string-encoded (latin1 bytes) embedding payload', () => {
      // Some serializers expose the binary HASH field as a latin1 string
      // rather than a Buffer; bytesToVector must handle that path.
      const buf = vectorToBytes([1.5, -2.25]);
      const asString = buf.toString('binary');
      const restored = bytesToVector(asString);
      expect(restored[0]).toBeCloseTo(1.5, 6);
      expect(restored[1]).toBeCloseTo(-2.25, 6);
    });
  });

  describe('config validation', () => {
    it('throws when neither client nor url is supplied', () => {
      expect(() =>
        redisVectorEmbeddingStore(
          {} as { client?: RedisLikeClient; url?: string },
        ),
      ).toThrow(/url.*or.*client/i);
    });
  });

  describe('similar (delegates to searchVector)', () => {
    it('forwards top-K and ignores model/version filters per contract', async () => {
      const store = redisVectorEmbeddingStore({ client: mock });
      mock.searchReplies.push({
        total: 2,
        documents: [
          { id: 'inferagraph:embedding:n1', value: { nodeId: 'n1', score: '0.2' } },
          { id: 'inferagraph:embedding:n2', value: { nodeId: 'n2', score: '0.5' } },
        ],
      });

      // Pass model+version filters; they're intentionally ignored.
      const hits = await store.similar([0.1, 0.2], 2, 'some-model', 'v1');

      expect(hits).toHaveLength(2);
      expect(hits[0]).toEqual({ nodeId: 'n1', score: 0.8 });
      expect(hits[1].nodeId).toBe('n2');
      const search = mock.commands.find((c) => c.cmd === 'FT.SEARCH')!;
      const opts = search.args[2] as Record<string, unknown>;
      expect((opts.PARAMS as Record<string, unknown>).top).toBe(2);
    });
  });

  describe('clear', () => {
    it('SCANs and DELs all <prefix>:embedding:* keys, leaving siblings alone', async () => {
      const store = redisVectorEmbeddingStore({ client: mock });
      mock.hashes.set('inferagraph:embedding:a', { nodeId: 'a' });
      mock.hashes.set('inferagraph:embedding:b', { nodeId: 'b' });
      mock.hashes.set('inferagraph:other:c', { nodeId: 'c' });

      await store.clear();

      expect(mock.hashes.has('inferagraph:embedding:a')).toBe(false);
      expect(mock.hashes.has('inferagraph:embedding:b')).toBe(false);
      expect(mock.hashes.has('inferagraph:other:c')).toBe(true);
      // SCAN was used (not KEYS).
      expect(mock.commands.some((c) => c.cmd === 'SCAN')).toBe(true);
    });

    it('is a no-op when no matching keys exist', async () => {
      const store = redisVectorEmbeddingStore({ client: mock });
      await expect(store.clear()).resolves.toBeUndefined();
      // No DEL because nothing matched.
      expect(mock.commands.some((c) => c.cmd === 'DEL')).toBe(false);
    });
  });

  describe('searchVector container hint', () => {
    it("accepts container: 'inferred_edges' as a forward-compat hint without changing behavior", async () => {
      const store = redisVectorEmbeddingStore({ client: mock });
      mock.searchReplies.push({ total: 0, documents: [] });

      await store.searchVector([0.1], { top: 1, container: 'inferred_edges' });

      // Hint is honored at the type level only — the call still targets the
      // unit embeddings index, the inferred-edges store has its own search.
      const search = mock.commands.find((c) => c.cmd === 'FT.SEARCH')!;
      expect(search.args[0]).toBe('inferagraph:embeddings:idx');
    });

    it('returns score=0 when the distance string is non-numeric (defensive parse)', async () => {
      const store = redisVectorEmbeddingStore({ client: mock });
      mock.searchReplies.push({
        total: 1,
        documents: [
          { id: 'inferagraph:embedding:n1', value: { nodeId: 'n1', score: 'NaN' } },
        ],
      });

      const hits = await store.searchVector([0.1], { top: 1 });

      expect(hits).toHaveLength(1);
      // parseFloat('NaN') -> NaN -> Number.isFinite false -> similarity 0.
      expect(hits[0].score).toBe(0);
    });

    it('falls back to doc.id when the hash has no nodeId field', async () => {
      const store = redisVectorEmbeddingStore({ client: mock });
      mock.searchReplies.push({
        total: 1,
        documents: [
          // No nodeId in value; bufferToString returns undefined and the
          // store falls back to doc.id.
          { id: 'inferagraph:embedding:fallback', value: { score: '0.3' } },
        ],
      });

      const hits = await store.searchVector([0.1], { top: 1 });
      expect(hits[0].nodeId).toBe('inferagraph:embedding:fallback');
    });
  });

  describe('client-capability guards', () => {
    function clientWithout(
      missing: 'hSet' | 'hGetAll' | 'ft',
    ): RedisLikeClient {
      const stub = new FakeRedisHashSearch();
      // Strip the missing capability so the matching guard fires.
      (stub as unknown as Record<string, unknown>)[missing] = undefined;
      return stub;
    }

    it('throws on set() when the client lacks hSet', async () => {
      const store = redisVectorEmbeddingStore({ client: clientWithout('hSet') });
      await expect(store.set(sampleRecord())).rejects.toThrow(/hSet/);
    });

    it('throws on get() when the client lacks hGetAll', async () => {
      const store = redisVectorEmbeddingStore({ client: clientWithout('hGetAll') });
      await expect(store.get('n', 'm', 'v', 'h')).rejects.toThrow(/hGetAll/);
    });

    it('throws on searchVector() when the client lacks ft', async () => {
      const store = redisVectorEmbeddingStore({ client: clientWithout('ft') });
      await expect(store.searchVector([0.1], { top: 1 })).rejects.toThrow(/ft\.search/);
    });
  });

  describe('ensureConnected', () => {
    it('calls client.connect() when isOpen is false', async () => {
      const closed = new FakeRedisHashSearch();
      closed.isOpen = false;
      const connectSpy = vi.spyOn(closed, 'connect');
      const store = redisVectorEmbeddingStore({ client: closed });
      await store.set(sampleRecord());
      expect(connectSpy).toHaveBeenCalledTimes(1);
    });

    it('warns and rethrows on connect failure, allows retry', async () => {
      const closed = new FakeRedisHashSearch();
      closed.isOpen = false;
      const connectSpy = vi
        .spyOn(closed, 'connect')
        .mockRejectedValueOnce(new Error('ECONNREFUSED'))
        .mockImplementationOnce(async () => {
          closed.isOpen = true;
        });
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const store = redisVectorEmbeddingStore({ client: closed });

      await expect(store.set(sampleRecord())).rejects.toThrow(/ECONNREFUSED/);
      expect(warn).toHaveBeenCalledWith(
        expect.stringContaining('[RedisVectorEmbeddingStore] failed to connect'),
      );

      await store.set(sampleRecord());
      expect(connectSpy).toHaveBeenCalledTimes(2);
      warn.mockRestore();
    });
  });
});
