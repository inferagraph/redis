import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  RedisInferredEdgeStore,
  redisInferredEdgeStore,
  type RedisLikeClient,
} from '../src/index.js';
import type { InferredEdge } from '@inferagraph/core/data';

class FakeRedisHashSearch implements RedisLikeClient {
  isOpen = true;
  hashes = new Map<string, Record<string, string | Buffer>>();
  commands: Array<{ cmd: string; args: unknown[] }> = [];
  searchReplies: Array<{ total: number; documents: { id: string; value: Record<string, unknown> }[] }> = [];

  async connect(): Promise<void> {}
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
    for (const k of arr) if (this.hashes.delete(k)) n += 1;
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
    for (const [k, v] of Object.entries(fields))
      stored[k] = v instanceof Buffer ? v : String(v);
    this.hashes.set(key, stored);
    return Object.keys(stored).length;
  }
  async hGetAll(key: string): Promise<Record<string, string | Buffer>> {
    this.commands.push({ cmd: 'HGETALL', args: [key] });
    return this.hashes.get(key) ?? {};
  }
  ft = {
    create: async (): Promise<string> => 'OK',
    search: async (
      indexName: string,
      query: string,
      options?: Record<string, unknown>,
    ): Promise<{ total: number; documents: { id: string; value: Record<string, unknown> }[] }> => {
      this.commands.push({ cmd: 'FT.SEARCH', args: [indexName, query, options] });
      return this.searchReplies.shift() ?? { total: 0, documents: [] };
    },
    dropIndex: async (): Promise<string> => 'OK',
  };
}

const sampleEdge = (overrides: Partial<InferredEdge> = {}): InferredEdge => ({
  sourceId: 'a',
  targetId: 'b',
  type: 'related_to',
  score: 0.42,
  sources: ['graph', 'embedding'],
  reasoning: undefined,
  perSource: undefined,
  ...overrides,
});

describe('RedisInferredEdgeStore', () => {
  let mock: FakeRedisHashSearch;

  beforeEach(() => {
    mock = new FakeRedisHashSearch();
  });

  describe('set (bulk replace)', () => {
    it('SCANs and DELs all existing edge keys, then HSETs the new entries', async () => {
      const store = redisInferredEdgeStore({ client: mock });
      // Pre-seed an existing edge so we can prove it gets wiped.
      mock.hashes.set('inferagraph:inferred_edge:old:edge:t', { id: 'old' });

      await store.set([sampleEdge({ sourceId: 'x', targetId: 'y', type: 'related_to' })]);

      // SCAN was used to enumerate existing keys.
      expect(mock.commands.some((c) => c.cmd === 'SCAN')).toBe(true);
      // The pre-seeded entry was deleted.
      expect(mock.hashes.has('inferagraph:inferred_edge:old:edge:t')).toBe(false);
      // The new entry was HSET.
      const hset = mock.commands.filter((c) => c.cmd === 'HSET');
      expect(hset).toHaveLength(1);
      expect(hset[0].args[0]).toBe('inferagraph:inferred_edge:x:y:related_to');
    });

    it('serializes sources and perSource into hash fields', async () => {
      const store = redisInferredEdgeStore({ client: mock });
      await store.set([
        sampleEdge({
          sources: ['graph', 'embedding', 'llm'],
          reasoning: 'because',
          perSource: { graph: { rank: 1, raw: 0.5 } },
        }),
      ]);

      const hsetArgs = mock.commands.find((c) => c.cmd === 'HSET')!.args[1] as Record<
        string,
        unknown
      >;
      expect(hsetArgs.sources).toBe('graph,embedding,llm');
      expect(hsetArgs.reasoning).toBe('because');
      expect(JSON.parse(hsetArgs.perSource as string)).toEqual({
        graph: { rank: 1, raw: 0.5 },
      });
    });
  });

  describe('getAllForNode', () => {
    it('queries by sourceId/targetId TAG indexes via OR', async () => {
      const store = redisInferredEdgeStore({ client: mock });
      mock.searchReplies.push({
        total: 1,
        documents: [
          {
            id: 'inferagraph:inferred_edge:a:b:rel',
            value: {
              sourceId: 'a',
              targetId: 'b',
              type: 'rel',
              score: '0.7',
              sources: 'graph',
            },
          },
        ],
      });

      const edges = await store.getAllForNode('a');

      const search = mock.commands.find((c) => c.cmd === 'FT.SEARCH')!;
      expect(search.args[0]).toBe('inferagraph:inferred_edges:idx');
      const query = search.args[1] as string;
      expect(query).toMatch(/@sourceId:\{a\}/);
      expect(query).toMatch(/@targetId:\{a\}/);
      expect(query).toMatch(/\|/);
      expect(edges).toHaveLength(1);
      expect(edges[0].sourceId).toBe('a');
      expect(edges[0].targetId).toBe('b');
      expect(edges[0].type).toBe('rel');
      expect(edges[0].score).toBeCloseTo(0.7, 6);
      expect(edges[0].sources).toEqual(['graph']);
    });
  });

  describe('searchInferredEdges', () => {
    it('issues FT.SEARCH with KNN syntax and converts distance to similarity', async () => {
      const store = new RedisInferredEdgeStore({ client: mock });
      mock.searchReplies.push({
        total: 2,
        documents: [
          { id: 'k1', value: { id: 'edge-1', score: '0.2' } },
          { id: 'k2', value: { id: 'edge-2', score: '0.5' } },
        ],
      });

      const hits = await store.searchInferredEdges([0.1, 0.2], 5);

      const search = mock.commands.find((c) => c.cmd === 'FT.SEARCH')!;
      expect(search.args[1]).toMatch(/KNN \$top @embedding \$vec AS score/);
      expect(hits).toHaveLength(2);
      expect(hits[0]).toEqual({ nodeId: 'edge-1', score: 0.8 });
      expect(hits[1].nodeId).toBe('edge-2');
      expect(hits[1].score).toBeCloseTo(0.5, 6);
    });
  });

  describe('clear', () => {
    it('removes all <prefix>:inferred_edge:* keys', async () => {
      const store = redisInferredEdgeStore({ client: mock });
      mock.hashes.set('inferagraph:inferred_edge:a:b:t', { id: 'x' });
      mock.hashes.set('inferagraph:inferred_edge:c:d:t', { id: 'y' });
      mock.hashes.set('other:thing', { id: 'z' });

      await store.clear();

      expect(mock.hashes.has('inferagraph:inferred_edge:a:b:t')).toBe(false);
      expect(mock.hashes.has('inferagraph:inferred_edge:c:d:t')).toBe(false);
      expect(mock.hashes.has('other:thing')).toBe(true);
    });
  });

  describe('factory', () => {
    it('accepts a pre-built client', () => {
      expect(() => redisInferredEdgeStore({ client: mock })).not.toThrow();
    });
    it('accepts a url and constructs a client internally', () => {
      expect(() =>
        redisInferredEdgeStore({ url: 'redis://localhost:6379' }),
      ).not.toThrow();
    });
    it('throws when neither client nor url is supplied', () => {
      expect(() =>
        redisInferredEdgeStore(
          {} as { client?: RedisLikeClient; url?: string },
        ),
      ).toThrow(/url.*or.*client/i);
    });
  });

  describe('get(sourceId, targetId)', () => {
    it('returns the matching edge when FT.SEARCH yields a document', async () => {
      const store = redisInferredEdgeStore({ client: mock });
      mock.searchReplies.push({
        total: 1,
        documents: [
          {
            id: 'inferagraph:inferred_edge:a:b:rel',
            value: {
              sourceId: 'a',
              targetId: 'b',
              type: 'rel',
              score: '0.7',
              sources: 'graph,llm',
              reasoning: 'because',
              perSource: JSON.stringify({ graph: { rank: 1, raw: 0.5 } }),
            },
          },
        ],
      });

      const edge = await store.get('a', 'b');

      expect(edge).toBeDefined();
      expect(edge!.sourceId).toBe('a');
      expect(edge!.targetId).toBe('b');
      expect(edge!.score).toBeCloseTo(0.7, 6);
      expect(edge!.sources).toEqual(['graph', 'llm']);
      expect(edge!.reasoning).toBe('because');
      expect(edge!.perSource).toEqual({ graph: { rank: 1, raw: 0.5 } });
      const search = mock.commands.find((c) => c.cmd === 'FT.SEARCH')!;
      const query = search.args[1] as string;
      expect(query).toMatch(/@sourceId:\{a\}/);
      expect(query).toMatch(/@targetId:\{b\}/);
    });

    it('returns undefined when FT.SEARCH reports zero documents', async () => {
      const store = redisInferredEdgeStore({ client: mock });
      mock.searchReplies.push({ total: 0, documents: [] });

      const edge = await store.get('missing', 'also-missing');
      expect(edge).toBeUndefined();
    });

    it('drops perSource silently when stored JSON is malformed', async () => {
      const store = redisInferredEdgeStore({ client: mock });
      mock.searchReplies.push({
        total: 1,
        documents: [
          {
            id: 'inferagraph:inferred_edge:a:b:rel',
            value: {
              sourceId: 'a',
              targetId: 'b',
              type: 'rel',
              score: '0.5',
              sources: 'graph',
              perSource: '{not valid json',
            },
          },
        ],
      });

      const edge = await store.get('a', 'b');
      expect(edge).toBeDefined();
      // Malformed JSON is swallowed and perSource left undefined.
      expect(edge!.perSource).toBeUndefined();
    });

    it('decodes Buffer-typed hash fields to strings', async () => {
      const store = redisInferredEdgeStore({ client: mock });
      mock.searchReplies.push({
        total: 1,
        documents: [
          {
            id: 'inferagraph:inferred_edge:a:b:rel',
            value: {
              // Real node-redis returns binary HASH fields as Buffer; bufferToString
              // must utf8-decode them.
              sourceId: Buffer.from('a', 'utf8'),
              targetId: Buffer.from('b', 'utf8'),
              type: Buffer.from('rel', 'utf8'),
              score: Buffer.from('0.42', 'utf8'),
              sources: Buffer.from('graph,embedding', 'utf8'),
              reasoning: Buffer.from('proof', 'utf8'),
            },
          },
        ],
      });

      const edge = await store.get('a', 'b');
      expect(edge!.sourceId).toBe('a');
      expect(edge!.targetId).toBe('b');
      expect(edge!.type).toBe('rel');
      expect(edge!.score).toBeCloseTo(0.42, 6);
      expect(edge!.sources).toEqual(['graph', 'embedding']);
      expect(edge!.reasoning).toBe('proof');
    });

    it('coerces non-string non-Buffer values via String()', async () => {
      const store = redisInferredEdgeStore({ client: mock });
      mock.searchReplies.push({
        total: 1,
        documents: [
          {
            id: 'inferagraph:inferred_edge:a:b:rel',
            value: {
              // Numeric fields fall through bufferToString's String() branch.
              sourceId: 42 as unknown as string,
              targetId: 'b',
              type: 'rel',
              score: '0.1',
              sources: 'graph',
            },
          },
        ],
      });

      const edge = await store.get('a', 'b');
      expect(edge!.sourceId).toBe('42');
    });
  });

  describe('getAll', () => {
    it('issues FT.SEARCH * and maps every document to an edge', async () => {
      const store = redisInferredEdgeStore({ client: mock });
      mock.searchReplies.push({
        total: 2,
        documents: [
          {
            id: 'k1',
            value: {
              sourceId: 'a',
              targetId: 'b',
              type: 'rel',
              score: '0.3',
              sources: 'graph',
            },
          },
          {
            id: 'k2',
            value: {
              sourceId: 'c',
              targetId: 'd',
              type: 'rel',
              score: '0.6',
              sources: 'embedding,llm',
            },
          },
        ],
      });

      const edges = await store.getAll();

      expect(edges).toHaveLength(2);
      const search = mock.commands.find((c) => c.cmd === 'FT.SEARCH')!;
      expect(search.args[1]).toBe('*');
      expect(edges[0].sources).toEqual(['graph']);
      expect(edges[1].sources).toEqual(['embedding', 'llm']);
    });

    it('returns [] when FT.SEARCH yields no documents', async () => {
      const store = redisInferredEdgeStore({ client: mock });
      mock.searchReplies.push({ total: 0, documents: [] });
      const edges = await store.getAll();
      expect(edges).toEqual([]);
    });
  });

  describe('client-capability guards', () => {
    function clientWithout(
      missing: 'hSet' | 'ft',
    ): RedisLikeClient {
      const stub = new FakeRedisHashSearch();
      (stub as unknown as Record<string, unknown>)[missing] = undefined;
      return stub;
    }

    it('throws on set() when the client lacks hSet', async () => {
      const store = redisInferredEdgeStore({ client: clientWithout('hSet') });
      await expect(store.set([sampleEdge()])).rejects.toThrow(/hSet/);
    });

    it('throws on get() when the client lacks ft', async () => {
      const store = redisInferredEdgeStore({ client: clientWithout('ft') });
      await expect(store.get('a', 'b')).rejects.toThrow(/ft\.search/);
    });

    it('throws on getAllForNode() when the client lacks ft', async () => {
      const store = redisInferredEdgeStore({ client: clientWithout('ft') });
      await expect(store.getAllForNode('a')).rejects.toThrow(/ft\.search/);
    });

    it('throws on getAll() when the client lacks ft', async () => {
      const store = redisInferredEdgeStore({ client: clientWithout('ft') });
      await expect(store.getAll()).rejects.toThrow(/ft\.search/);
    });

    it('throws on searchInferredEdges() when the client lacks ft', async () => {
      const store = redisInferredEdgeStore({ client: clientWithout('ft') });
      await expect(store.searchInferredEdges([0.1], 1)).rejects.toThrow(/ft\.search/);
    });
  });

  describe('ensureConnected', () => {
    it('calls client.connect() when isOpen is false', async () => {
      const closed = new FakeRedisHashSearch();
      closed.isOpen = false;
      const connectSpy = vi.spyOn(closed, 'connect');
      const store = redisInferredEdgeStore({ client: closed });
      await store.clear();
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
      const store = redisInferredEdgeStore({ client: closed });

      await expect(store.clear()).rejects.toThrow(/ECONNREFUSED/);
      expect(warn).toHaveBeenCalledWith(
        expect.stringContaining('[RedisInferredEdgeStore] failed to connect'),
      );

      await store.clear();
      expect(connectSpy).toHaveBeenCalledTimes(2);
      warn.mockRestore();
    });
  });

  describe('searchInferredEdges defensive parsing', () => {
    it('falls back to doc.id when the hash has no id field', async () => {
      const store = new RedisInferredEdgeStore({ client: mock });
      mock.searchReplies.push({
        total: 1,
        documents: [
          // No id in value; the store should fall back to doc.id.
          { id: 'doc-fallback', value: { score: '0.3' } },
        ],
      });

      const hits = await store.searchInferredEdges([0.1], 1);
      expect(hits[0].nodeId).toBe('doc-fallback');
    });

    it('returns score=0 when distance is non-numeric', async () => {
      const store = new RedisInferredEdgeStore({ client: mock });
      mock.searchReplies.push({
        total: 1,
        documents: [{ id: 'k', value: { id: 'edge-1', score: 'NaN' } }],
      });

      const hits = await store.searchInferredEdges([0.1], 1);
      expect(hits[0].score).toBe(0);
    });
  });
});
