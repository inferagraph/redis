import { describe, it, expect, beforeEach } from 'vitest';
import {
  RedisInferredEdgeStore,
  redisInferredEdgeStore,
  type RedisLikeClient,
} from '../src/index.js';
import type { InferredEdge } from '@inferagraph/core';

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
  });
});
