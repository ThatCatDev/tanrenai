import * as vscode from 'vscode';

export type ConnectionState =
  | { status: 'idle' }
  | { status: 'no_credentials' }
  | { status: 'connecting' }
  | { status: 'connected'; model: string; toolCount: number }
  | { status: 'error'; message: string };

/**
 * Sidebar webview. v1 phase 2 shows connection state and exposes a "Login"
 * button when credentials are missing. Chat surface lands in the next
 * commit.
 */
export class ChatViewProvider implements vscode.WebviewViewProvider {
  static readonly viewType = 'tanrenai.chat';

  private view?: vscode.WebviewView;
  private state: ConnectionState = { status: 'idle' };

  constructor(private readonly extensionUri: vscode.Uri) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, 'media')],
    };
    view.webview.html = this.renderHtml(view.webview);

    view.webview.onDidReceiveMessage((msg: { type: string }) => {
      if (msg.type === 'login') {
        void vscode.commands.executeCommand('tanrenai.login');
      } else if (msg.type === 'reconnect') {
        void vscode.commands.executeCommand('tanrenai.reconnect');
      }
    });

    this.postState();
  }

  setState(state: ConnectionState): void {
    this.state = state;
    this.postState();
  }

  private postState(): void {
    void this.view?.webview.postMessage({ type: 'state', state: this.state });
  }

  private renderHtml(webview: vscode.Webview): string {
    const csp = [
      "default-src 'none'",
      `style-src ${webview.cspSource} 'unsafe-inline'`,
      `script-src ${webview.cspSource} 'unsafe-inline'`,
    ].join('; ');

    return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="Content-Security-Policy" content="${csp}" />
    <title>Tanrenai</title>
    <style>
      body { font-family: var(--vscode-font-family); padding: 1rem; color: var(--vscode-foreground); }
      .status { font-size: 0.85rem; opacity: 0.8; }
      .status.connected { color: var(--vscode-charts-green); }
      .status.error { color: var(--vscode-charts-red); }
      button {
        background: var(--vscode-button-background);
        color: var(--vscode-button-foreground);
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 2px;
        cursor: pointer;
        margin-top: 0.5rem;
      }
      button:hover { background: var(--vscode-button-hoverBackground); }
      .panel { padding: 1rem 0; }
      .placeholder { font-size: 0.85rem; opacity: 0.6; margin-top: 1rem; }
    </style>
  </head>
  <body>
    <div id="root">
      <div class="panel"><div class="status">Loading…</div></div>
    </div>
    <script>
      const vscode = acquireVsCodeApi();
      const root = document.getElementById('root');

      function render(state) {
        switch (state.status) {
          case 'idle':
            root.innerHTML = '<div class="panel"><div class="status">Initialising…</div></div>';
            break;
          case 'no_credentials':
            root.innerHTML =
              '<div class="panel">' +
              '<div class="status">Not signed in.</div>' +
              '<button id="login">Sign in to Tanrenai</button>' +
              '</div>';
            document.getElementById('login').addEventListener('click', () => {
              vscode.postMessage({ type: 'login' });
            });
            break;
          case 'connecting':
            root.innerHTML = '<div class="panel"><div class="status">Connecting…</div></div>';
            break;
          case 'connected':
            root.innerHTML =
              '<div class="panel">' +
              '<div class="status connected">Connected</div>' +
              '<div class="placeholder">Model: ' + escape(state.model) + '</div>' +
              '<div class="placeholder">' + state.toolCount + ' tools available</div>' +
              '<div class="placeholder">Chat UI lands in the next commit.</div>' +
              '</div>';
            break;
          case 'error':
            root.innerHTML =
              '<div class="panel">' +
              '<div class="status error">Error: ' + escape(state.message) + '</div>' +
              '<button id="retry">Retry</button>' +
              '</div>';
            document.getElementById('retry').addEventListener('click', () => {
              vscode.postMessage({ type: 'reconnect' });
            });
            break;
        }
      }

      function escape(s) {
        return String(s)
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');
      }

      window.addEventListener('message', (event) => {
        const msg = event.data;
        if (msg.type === 'state') {
          render(msg.state);
        }
      });
    </script>
  </body>
</html>`;
  }
}
