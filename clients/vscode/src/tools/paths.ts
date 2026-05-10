import * as path from 'node:path';
import * as vscode from 'vscode';

/**
 * Resolve a tool-supplied path string into a vscode.Uri, treating relative
 * paths as workspace-relative. Matches the Go side, which chdir's into the
 * workspace root at startup so relative paths Just Work.
 */
export function resolvePath(rawPath: string, workspaceRoot: string): vscode.Uri {
  if (path.isAbsolute(rawPath)) {
    return vscode.Uri.file(rawPath);
  }
  if (workspaceRoot) {
    return vscode.Uri.file(path.resolve(workspaceRoot, rawPath));
  }

  return vscode.Uri.file(path.resolve(rawPath));
}
