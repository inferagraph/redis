/**
 * Local shim mirroring `@inferagraph/core`'s ConversationStore +
 * ConversationTurn types. Used by tests as a stable structural reference so
 * core minor-version bumps don't ripple into the test surface. Source of
 * truth lives in `@inferagraph/core`'s `src/ai/ConversationStore.ts`.
 */
export interface ConversationTurn {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  retrievedNodeIds?: string[];
}

export interface ConversationStore {
  getTurns(conversationId: string, limit: number): Promise<ConversationTurn[]>;
  appendTurn(conversationId: string, turn: ConversationTurn): Promise<void>;
  clear(conversationId: string): Promise<void>;
}
