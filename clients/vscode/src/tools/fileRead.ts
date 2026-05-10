import * as vscode from 'vscode';
import { resolvePath } from './paths';
import { err, ok, parseArgs, ToolImpl } from './types';

const MAX_FILE_READ_BYTES = 32 * 1024;

interface FileReadArgs {
  path: string;
}

/**
 * file_read shim — reads the editor's view of the file (unsaved buffer if
 * open, else disk). Mirrors the Go tool's 32KB truncation so the model sees
 * the same shape regardless of which side runs the tool.
 */
export const fileRead: ToolImpl = async (raw, ctx) => {
  const parsed = parseArgs<FileReadArgs>(raw);
  if ('error' in parsed) {
    return err(parsed.error);
  }
  const { path: rawPath } = parsed.args;
  if (!rawPath) {
    return err('path is required');
  }

  const uri = resolvePath(rawPath, ctx.workspaceRoot);

  let text: string;
  try {
    const doc = await vscode.workspace.openTextDocument(uri);
    text = doc.getText();
  } catch (e) {
    return err(`failed to read file: ${(e as Error).message}`);
  }

  // Match Go behaviour: byte-truncate at 32KB, append a notice.
  const buf = Buffer.from(text, 'utf8');
  if (buf.length > MAX_FILE_READ_BYTES) {
    const head = buf.subarray(0, MAX_FILE_READ_BYTES).toString('utf8');

    return ok(
      `${head}\n\n[truncated: file is ${buf.length} bytes, showing first ${MAX_FILE_READ_BYTES}]`,
    );
  }

  return ok(text);
};
