/**
 * Local shim for `@inferagraph/core@0.8.0`'s ConversationStore + ConversationTurn
 * types. Used by tests because the locally-installed core (devDep) is 0.6.0
 * which predates these types. Source of truth lives in
 * `@inferagraph/core@0.8.0`'s `src/ai/ConversationStore.ts` (branch `llm`,
 * commit `4f154b3`). Keep this file structurally identical to that source.
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
