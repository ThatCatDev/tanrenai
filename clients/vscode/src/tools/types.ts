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
  /**
   * Show the user a diff of a proposed file edit and wait for their
   * decision. Returns true to apply, false to reject. Tools that mutate
   * the workspace should always go through this — gives the user a
   * chance to inspect the change before it lands.
   */
  approveEdit(opts: ApproveEditOpts): Promise<boolean>;
}

export interface ApproveEditOpts {
  /** Display name (relative path) for the file being edited. */
  label: string;
  /** Real file URI on disk. */
  uri: import('vscode').Uri;
  /** Proposed full-file content after the edit. */
  proposed: string;
  /** Original content (omit when creating a new file). */
  original?: string;
  /** A short description, e.g. "Replace 12 chars" or "Create new file". */
  summary: string;
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
