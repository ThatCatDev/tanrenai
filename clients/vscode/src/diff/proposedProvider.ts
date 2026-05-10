import * as vscode from 'vscode';

/**
 * Read-only virtual document provider for the agent's proposed content.
 * Lets us open a `vscode.diff` view between the real file and the
 * proposed change without writing anything to disk first.
 */
export class ProposedContentProvider implements vscode.TextDocumentContentProvider {
  static readonly scheme = 'tanrenai-proposed';

  private contents = new Map<string, string>();
  private readonly emitter = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this.emitter.event;

  /** Register a proposed-content body for a virtual URI. */
  set(uri: vscode.Uri, content: string): void {
    this.contents.set(uri.toString(), content);
    this.emitter.fire(uri);
  }

  /** Drop a proposed-content body. The diff tab can stay open but will go blank. */
  clear(uri: vscode.Uri): void {
    this.contents.delete(uri.toString());
    this.emitter.fire(uri);
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.contents.get(uri.toString()) ?? '';
  }

  /**
   * Build a virtual URI for the proposed content. The query string holds a
   * disambiguator so a single file can have multiple in-flight previews
   * without clobbering each other.
   */
  uriFor(realUri: vscode.Uri, id: string): vscode.Uri {
    return vscode.Uri.from({
      scheme: ProposedContentProvider.scheme,
      path: realUri.path,
      query: `id=${id}`,
    });
  }
}
