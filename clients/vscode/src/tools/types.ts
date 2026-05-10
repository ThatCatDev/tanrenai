/**
 * Result returned by a tool shim. `ok=true` plus `content` is success;
 * `ok=false` with `error` is a tool-level failure (the model will see it and
 * may retry). Throwing from a shim is reserved for infrastructure errors.
 */
export interface ToolResult {
  ok: boolean;
  content?: string;
  error?: string;
}

export interface ToolContext {
  /** Workspace root from init.workspaceRoot — used to resolve relative paths. */
  workspaceRoot: string;
}

export type ToolImpl = (rawArgs: string, ctx: ToolContext) => Promise<ToolResult>;

export function ok(content: string): ToolResult {
  return { ok: true, content };
}

export function err(message: string): ToolResult {
  return { ok: false, error: message };
}

export function parseArgs<T>(raw: string): { args: T } | { error: string } {
  try {
    return { args: JSON.parse(raw) as T };
  } catch (e) {
    return { error: `invalid arguments: ${(e as Error).message}` };
  }
}
