import { describe, it, expect, beforeEach } from 'vitest';
import { provisionRedisVectorIndex, type RedisLikeClient } from '../src/index.js';

class FakeRedisFt implements RedisLikeClient {
  isOpen = true;
  commands: Array<{ cmd: string; args: unknown[] }> = [];
  /** When non-null, the next ft.create call throws with this message. */
  nextCreateError: string | null = null;

  async connect(): Promise<void> {
    this.isOpen = true;
  }
  async get(): Promise<string | null> {
    return null;
  }
  async set(): Promise<unknown> {
    return 'OK';
  }
  async del(): Promise<number> {
    return 0;
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
  async *scanIterator(): AsyncIterable<string> {
    return;
  }
  ft = {
    create: async (
      indexName: string,
      schema: Record<string, unknown>,
      options?: Record<string, unknown>,
    ): Promise<string> => {
      this.commands.push({ cmd: 'FT.CREATE', args: [indexName, schema, options] });
      if (this.nextCreateError) {
        const msg = this.nextCreateError;
        this.nextCreateError = null;
        throw new Error(msg);
      }
      return 'OK';
    },
    search: async (): Promise<{
      total: number;
      documents: { id: string; value: Record<string, unknown> }[];
    }> => ({ total: 0, documents: [] }),
    dropIndex: async (): Promise<string> => 'OK',
  };
}

describe('provisionRedisVectorIndex', () => {
  let mock: FakeRedisFt;

  beforeEach(() => {
    mock = new FakeRedisFt();
  });

  describe('embeddings index', () => {
    it('creates <keyPrefix>:embeddings:idx with HNSW vector schema and HASH PREFIX', async () => {
      await provisionRedisVectorIndex({ client: mock });

      const create = mock.commands.find((c) => c.cmd === 'FT.CREATE')!;
      expect(create.args[0]).toBe('inferagraph:embeddings:idx');
      const schema = create.args[1] as Record<string, { type: string } & Record<string, unknown>>;
      expect(schema.nodeId.type).toBe('TAG');
      expect(schema.embedding.type).toBe('VECTOR');
      expect(schema.embedding.ALGORITHM).toBe('HNSW');
      expect(schema.embedding.DIM).toBe(3072);
      expect(schema.embedding.DISTANCE_METRIC).toBe('COSINE');

      const opts = create.args[2] as Record<string, unknown>;
      expect(opts.ON).toBe('HASH');
      expect(opts.PREFIX).toEqual({ count: 1, prefix: 'inferagraph:embedding:' });
    });

    it('treats "Index already exists" as idempotent no-op', async () => {
      mock.nextCreateError = 'Index already exists';
      // alsoProvisionInferredEdges defaults true; the second create call must
      // also succeed even though the first failed-as-idempotent.
      await expect(
        provisionRedisVectorIndex({ client: mock }),
      ).resolves.toBeUndefined();
      // Both create calls were issued.
      expect(mock.commands.filter((c) => c.cmd === 'FT.CREATE')).toHaveLength(2);
    });

    it('rethrows non-idempotent errors', async () => {
      mock.nextCreateError = 'CROSSSLOT Keys in request do not hash to the same slot';
      await expect(provisionRedisVectorIndex({ client: mock })).rejects.toThrow(
        /CROSSSLOT/,
      );
    });
  });

  describe('inferred_edges index', () => {
    it('creates <keyPrefix>:inferred_edges:idx by default (alsoProvisionInferredEdges=true)', async () => {
      await provisionRedisVectorIndex({ client: mock });

      const creates = mock.commands.filter((c) => c.cmd === 'FT.CREATE');
      expect(creates).toHaveLength(2);
      const edgesCreate = creates[1];
      expect(edgesCreate.args[0]).toBe('inferagraph:inferred_edges:idx');
      const schema = edgesCreate.args[1] as Record<string, { type: string }>;
      expect(schema.sourceId.type).toBe('TAG');
      expect(schema.targetId.type).toBe('TAG');
      const opts = edgesCreate.args[2] as Record<string, unknown>;
      expect(opts.PREFIX).toEqual({ count: 1, prefix: 'inferagraph:inferred_edge:' });
    });

    it('skips inferred_edges when alsoProvisionInferredEdges: false', async () => {
      await provisionRedisVectorIndex({
        client: mock,
        alsoProvisionInferredEdges: false,
      });
      const creates = mock.commands.filter((c) => c.cmd === 'FT.CREATE');
      expect(creates).toHaveLength(1);
      expect(creates[0].args[0]).toBe('inferagraph:embeddings:idx');
    });
  });

  describe('config', () => {
    it('honors custom keyPrefix and embeddingDimensions', async () => {
      await provisionRedisVectorIndex({
        client: mock,
        keyPrefix: 'biblegraph',
        embeddingDimensions: 1536,
        alsoProvisionInferredEdges: false,
      });
      const create = mock.commands.find((c) => c.cmd === 'FT.CREATE')!;
      expect(create.args[0]).toBe('biblegraph:embeddings:idx');
      const schema = create.args[1] as Record<string, { DIM?: number }>;
      expect(schema.embedding.DIM).toBe(1536);
      const opts = create.args[2] as Record<string, unknown>;
      expect(opts.PREFIX).toEqual({ count: 1, prefix: 'biblegraph:embedding:' });
    });

    it('throws when neither client nor url is supplied', async () => {
      await expect(
        provisionRedisVectorIndex({} as { client?: RedisLikeClient; url?: string }),
      ).rejects.toThrow(/url.*or.*client/i);
    });
  });
});
