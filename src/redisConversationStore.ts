import type { ConversationStore, ConversationTurn } from '@inferagraph/core/data';
import { createClient } from 'redis';

/**
 * Subset of node-redis v4 the conversation store needs (LIST + EXPIRE + DEL).
 * Re-exported for tests so a stub can target only the conversation surface.
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

export interface RedisConversationStoreConfig {
  /** Pre-built node-redis client. One of `client` / `url` is required. */
  client?: RedisConversationLikeClient;
  /** Redis connection URL. The factory builds a client internally when supplied. */
  url?: string;
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
 * Redis-backed `ConversationStore` for `@inferagraph/core@^0.9.0`'s `AIEngine`.
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
export class RedisConversationStore implements ConversationStore {
  private readonly client: RedisConversationLikeClient;
  private readonly keyPrefix: string;
  private readonly ttlSeconds: number;
  private connectPromise: Promise<void> | undefined;

  constructor(config: RedisConversationStoreConfig) {
    if (!config.client && !config.url) {
      throw new Error('RedisConversationStore: either `url` or `client` must be provided');
    }
    this.client =
      config.client ??
      (createClient({ url: config.url! }) as unknown as RedisConversationLikeClient);
    this.keyPrefix = config.keyPrefix ?? DEFAULT_CONVERSATION_PREFIX;
    this.ttlSeconds = config.ttlSeconds ?? DEFAULT_CONVERSATION_TTL_SECONDS;
  }

  async appendTurn(conversationId: string, turn: ConversationTurn): Promise<void> {
    await this.ensureConnected();
    const key = this.dataKey(conversationId);
    const payload = JSON.stringify(turn);
    await this.client.lPush(key, payload);
    await this.client.lTrim(key, 0, MAX_TURNS_PER_CONVERSATION - 1);
    await this.client.expire(key, this.ttlSeconds);
  }

  async getTurns(conversationId: string, limit: number): Promise<ConversationTurn[]> {
    if (limit <= 0) return [];
    await this.ensureConnected();
    const key = this.dataKey(conversationId);
    const raw = await this.client.lRange(key, 0, limit - 1);
    if (raw.length === 0) return [];
    const parsed: ConversationTurn[] = [];
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

/**
 * Construct a {@link RedisConversationStore}. Accepts a pre-built client OR
 * a `url`; when only `url` is provided, the factory constructs the underlying
 * redis client internally so consumers never import the SDK.
 */
export function redisConversationStore(
  config: RedisConversationStoreConfig,
): ConversationStore {
  return new RedisConversationStore(config);
}

function safeParseTurn(raw: string): ConversationTurn | undefined {
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

function isConversationTurn(value: unknown): value is ConversationTurn {
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

// Keep the public type re-exports for the legacy import site (RedisConversationTurn).
/**
 * @deprecated Re-export of the core `ConversationTurn` type. Use
 *  `import type { ConversationTurn } from '@inferagraph/core/data';` instead.
 */
export type RedisConversationTurn = ConversationTurn;
