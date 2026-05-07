import { describe, it, expect, beforeEach, vi } from 'vitest';
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

  describe('connect path', () => {
    it('calls client.connect() when isOpen is false', async () => {
      const closed = new FakeRedisFt();
      closed.isOpen = false;
      const connectSpy = vi.spyOn(closed, 'connect');

      await provisionRedisVectorIndex({ client: closed, alsoProvisionInferredEdges: false });

      expect(connectSpy).toHaveBeenCalledTimes(1);
      expect(closed.commands.filter((c) => c.cmd === 'FT.CREATE')).toHaveLength(1);
    });

    it('throws when the connected client lacks ft', async () => {
      const noFt = new FakeRedisFt();
      // Strip the ft surface so the guard fires post-connect.
      (noFt as unknown as { ft: undefined }).ft = undefined;

      await expect(
        provisionRedisVectorIndex({ client: noFt }),
      ).rejects.toThrow(/ft\.create/);
    });
  });

  describe('isIndexAlreadyExistsError fallbacks', () => {
    it('treats a string thrown value containing the marker as idempotent', async () => {
      // Some legacy paths throw a raw string instead of an Error.
      const ft = mock.ft;
      let attempt = 0;
      ft.create = async (
        indexName: string,
        schema: Record<string, unknown>,
        options?: Record<string, unknown>,
      ): Promise<string> => {
        mock.commands.push({ cmd: 'FT.CREATE', args: [indexName, schema, options] });
        attempt += 1;
        if (attempt === 1) {
          // String, not Error — exercises the typeof === 'string' branch.
          // eslint-disable-next-line @typescript-eslint/only-throw-error
          throw 'Index already exists';
        }
        return 'OK';
      };

      await expect(
        provisionRedisVectorIndex({ client: mock }),
      ).resolves.toBeUndefined();
    });

    it('treats an object with a message property containing the marker as idempotent', async () => {
      // Non-Error, non-string thrown object — exercises the String() coerce branch.
      const ft = mock.ft;
      let attempt = 0;
      ft.create = async (
        indexName: string,
        schema: Record<string, unknown>,
        options?: Record<string, unknown>,
      ): Promise<string> => {
        mock.commands.push({ cmd: 'FT.CREATE', args: [indexName, schema, options] });
        attempt += 1;
        if (attempt === 1) {
          // eslint-disable-next-line @typescript-eslint/only-throw-error
          throw { message: 'Index already exists in cluster' };
        }
        return 'OK';
      };

      await expect(
        provisionRedisVectorIndex({ client: mock }),
      ).resolves.toBeUndefined();
    });

    it('rethrows a non-Error non-string value that does not match the marker', async () => {
      const ft = mock.ft;
      ft.create = async (): Promise<string> => {
        // eslint-disable-next-line @typescript-eslint/only-throw-error
        throw { code: 'WRONGTYPE' };
      };

      await expect(
        provisionRedisVectorIndex({ client: mock }),
      ).rejects.toBeDefined();
    });

    it('rethrows when the thrown value is null/undefined-shaped', async () => {
      const ft = mock.ft;
      ft.create = async (): Promise<string> => {
        // eslint-disable-next-line @typescript-eslint/only-throw-error
        throw null;
      };

      await expect(
        provisionRedisVectorIndex({ client: mock }),
      ).rejects.toBeDefined();
    });
  });
});
