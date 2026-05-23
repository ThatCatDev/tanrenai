import * as vscode from 'vscode';
import { resolvePath } from './paths';
import { err, ok, parseArgs, ToolImpl } from './types';

interface FileWriteArgs {
  path: string;
  content: string;
}

/**
 * file_write shim — replaces the file's full contents (or creates it).
 * Now goes through approveEdit first: opens VS Code's diff editor and
 * waits for the user's decision before applying. WorkspaceEdit so undo
 * still restores the prior state.
 */
export const fileWrite: ToolImpl = async (raw, ctx) => {
  const parsed = parseArgs<FileWriteArgs>(raw);
  if ('error' in parsed) {
    return err(parsed.error);
  }
  const { path: rawPath, content } = parsed.args;
  if (!rawPath) {
    return err('path is required');
  }
  if (typeof content !== 'string') {
    return err('content is required');
  }

  const uri = resolvePath(rawPath, ctx.workspaceRoot);

  let exists = true;
  let original: string | undefined;
  try {
    await vscode.workspace.fs.stat(uri);
    const doc = await vscode.workspace.openTextDocument(uri);
    original = doc.getText();
  } catch {
    exists = false;
  }

  const summary = exists
    ? `Replace ${rawPath} (${content.length} chars)`
    : `Create ${rawPath} (${content.length} chars)`;

  const approved = await ctx.approveEdit({
    label: rawPath,
    uri,
    proposed: content,
    original,
    summary,
  });
  if (!approved) {
    return err(`User rejected the proposed edit to ${rawPath}`);
  }

  const edit = new vscode.WorkspaceEdit();
  if (!exists) {
    edit.createFile(uri, { ignoreIfExists: false });
    edit.insert(uri, new vscode.Position(0, 0), content);
  } else {
    const doc = await vscode.workspace.openTextDocument(uri);
    const fullRange = new vscode.Range(
      doc.positionAt(0),
      doc.positionAt(doc.getText().length),
    );
    edit.replace(uri, fullRange, content);
  }

  const applied = await vscode.workspace.applyEdit(edit);
  if (!applied) {
    return err('VS Code declined to apply the edit (file may be read-only or in conflict)');
  }

  // Save so the change persists if the editor isn't open. Best-effort.
  try {
    const doc = await vscode.workspace.openTextDocument(uri);
    if (doc.isDirty) {
      await doc.save();
    }
  } catch {
    // ignore — write was applied; saving is opportunistic
  }

  return ok(exists ? `Replaced ${rawPath}` : `Created ${rawPath}`);
};
