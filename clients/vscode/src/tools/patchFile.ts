import * as vscode from 'vscode';
import { resolvePath } from './paths';
import { err, ok, parseArgs, ToolImpl } from './types';

interface PatchFileArgs {
  path: string;
  old_string: string;
  new_string: string;
}

const SNIPPET_BYTES = 500;

/**
 * patch_file shim — find-and-replace edit. old_string must match exactly
 * once. Mirrors the Go tool's error messages so the model gets the same
 * hints when matching fails. Applies via WorkspaceEdit so VS Code shows
 * the change inline and undo restores the prior state.
 */
export const patchFile: ToolImpl = async (raw, ctx) => {
  const parsed = parseArgs<PatchFileArgs>(raw);
  if ('error' in parsed) {
    return err(parsed.error);
  }
  const { path: rawPath, old_string: oldString, new_string: newString } = parsed.args;
  if (!rawPath) {
    return err('path is required');
  }
  if (!oldString) {
    return err('old_string is required');
  }
  if (oldString === newString) {
    return err('old_string and new_string are identical — nothing to change');
  }

  const uri = resolvePath(rawPath, ctx.workspaceRoot);
  let doc: vscode.TextDocument;
  try {
    doc = await vscode.workspace.openTextDocument(uri);
  } catch (e) {
    if ((e as { code?: string }).code === 'FileNotFound') {
      return err(`file not found: ${rawPath}`);
    }

    return err(`failed to read file: ${(e as Error).message}`);
  }

  const text = doc.getText();
  const matches = countOccurrences(text, oldString);
  if (matches === 0) {
    const snippet = text.length > SNIPPET_BYTES ? text.slice(0, SNIPPET_BYTES) + '\n...(truncated)' : text;

    return err(
      `old_string not found in ${rawPath}. File content starts with:\n${snippet}`,
    );
  }
  if (matches > 1) {
    return err(
      `old_string matches ${matches} locations in ${rawPath}. Include more surrounding context in old_string to make it unique.`,
    );
  }

  const startOffset = text.indexOf(oldString);
  const range = new vscode.Range(
    doc.positionAt(startOffset),
    doc.positionAt(startOffset + oldString.length),
  );

  const edit = new vscode.WorkspaceEdit();
  edit.replace(uri, range, newString);
  const applied = await vscode.workspace.applyEdit(edit);
  if (!applied) {
    return err('VS Code declined to apply the edit');
  }
  try {
    if (doc.isDirty) {
      await doc.save();
    }
  } catch {
    // best-effort
  }

  return ok(
    `Replaced ${oldString.length} chars with ${newString.length} chars in ${rawPath}`,
  );
};

function countOccurrences(haystack: string, needle: string): number {
  if (!needle) {
    return 0;
  }
  let count = 0;
  let from = 0;
  while (true) {
    const i = haystack.indexOf(needle, from);
    if (i < 0) {
      break;
    }
    count++;
    from = i + needle.length;
  }

  return count;
}
