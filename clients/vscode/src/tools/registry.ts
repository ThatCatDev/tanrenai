import { fileRead } from './fileRead';
import { fileWrite } from './fileWrite';
import { patchFile } from './patchFile';
import { ToolImpl, ToolResult } from './types';

const REGISTRY: Record<string, ToolImpl> = {
  file_read: fileRead,
  file_write: fileWrite,
  patch_file: patchFile,
};

/** Names the extension claims to intercept (sent in init.interceptedTools). */
export const interceptedToolNames: string[] = Object.keys(REGISTRY);

export async function dispatchTool(
  name: string,
  rawArgs: string,
  workspaceRoot: string,
): Promise<ToolResult> {
  const impl = REGISTRY[name];
  if (!impl) {
    return { ok: false, error: `tool ${name} is not implemented in the extension` };
  }
  try {
    return await impl(rawArgs, { workspaceRoot });
  } catch (e) {
    return { ok: false, error: `tool ${name} threw: ${(e as Error).message}` };
  }
}
