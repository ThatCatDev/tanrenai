import * as vscode from 'vscode';
import type {
  ConnectionState,
  Mode,
  SelectionAttachment,
  WebviewInbound,
  WebviewOutbound,
} from './protocol';

export type { ConnectionState, Mode, SelectionAttachment, WebviewInbound, WebviewOutbound };

export interface ChatViewListener {
  onSend(content: string, attachments?: SelectionAttachment[]): void;
  onCancel(): void;
  onCancelConnect(): void;
  onPickModel(): void;
  onClearChat(): void;
  onSetMode(mode: Mode): void;
  onApprovalDecision(id: string, action: 'allow' | 'deny' | 'always'): void;
  onAttachRequest(): void;
  onLogin(): void;
  onLogout(): void;
  onReconnect(): void;
  onStopGpu(): void;
  onDestroyGpu(): void;
  onShowGpuStatus(): void;
  /** Called after the webview has finished mounting (or remounting). */
  onMounted(): void;
}

/**
 * Sidebar webview. Loads the Preact bundle (dist/webview.js) which renders
 * the chat surface. This class is now a thin host-side bridge: forward
 * inbound user actions to the controller, push outbound state updates to
 * the bundle.
 */
export class ChatViewProvider implements vscode.WebviewViewProvider {
  static readonly viewType = 'tanrenai.chat';

  private view?: vscode.WebviewView;
  private state: ConnectionState = { status: 'idle' };
  private listener?: ChatViewListener;
  private pending: WebviewOutbound[] = [];

  constructor(private readonly extensionUri: vscode.Uri) {}

  setListener(listener: ChatViewListener): void {
    this.listener = listener;
  }

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [
        vscode.Uri.joinPath(this.extensionUri, 'dist'),
        vscode.Uri.joinPath(this.extensionUri, 'media'),
      ],
    };
    view.webview.html = this.renderHtml(view.webview);

    view.webview.onDidReceiveMessage((msg: WebviewInbound) => {
      switch (msg.type) {
        case 'send':
          this.listener?.onSend(msg.content, msg.attachments);
          break;
        case 'attach_request':
          this.listener?.onAttachRequest();
          break;
        case 'attach_clear':
          // No-op on host; webview handles its own state. Kept for symmetry.
          break;
        case 'cancel':
          this.listener?.onCancel();
          break;
        case 'cancel_connect':
          this.listener?.onCancelConnect();
          break;
        case 'pick_model':
          this.listener?.onPickModel();
          break;
        case 'clear_chat':
          this.listener?.onClearChat();
          break;
        case 'set_mode':
          this.listener?.onSetMode(msg.mode);
          break;
        case 'approval_decision':
          this.listener?.onApprovalDecision(msg.id, msg.action);
          break;
        case 'login':
          this.listener?.onLogin();
          break;
        case 'logout':
          this.listener?.onLogout();
          break;
        case 'reconnect':
          this.listener?.onReconnect();
          break;
        case 'stop_gpu':
          this.listener?.onStopGpu();
          break;
        case 'destroy_gpu':
          this.listener?.onDestroyGpu();
          break;
        case 'show_gpu_status':
          this.listener?.onShowGpuStatus();
          break;
      }
    });

    // Push the latest connection state as soon as the bundle mounts, then
    // ask the controller to replay any prior transcript (for remount).
    this.post({ type: 'state', state: this.state });
    for (const msg of this.pending) {
      this.post(msg);
    }
    this.pending = [];
    this.listener?.onMounted();
  }

  setState(state: ConnectionState): void {
    this.state = state;
    this.post({ type: 'state', state });
  }

  send(msg: WebviewOutbound): void {
    if (!this.view) {
      this.pending.push(msg);

      return;
    }
    this.post(msg);
  }

  private post(msg: WebviewOutbound): void {
    void this.view?.webview.postMessage(msg);
  }

  private renderHtml(webview: vscode.Webview): string {
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, 'dist', 'webview.js'),
    );
    const styleUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, 'media', 'chat.css'),
    );
    const nonce = makeNonce();

    const csp = [
      "default-src 'none'",
      `style-src ${webview.cspSource}`,
      `script-src 'nonce-${nonce}'`,
      `font-src ${webview.cspSource}`,
    ].join('; ');

    return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="Content-Security-Policy" content="${csp}" />
    <link rel="stylesheet" href="${styleUri}" />
    <title>Tanrenai</title>
  </head>
  <body>
    <div id="root"></div>
    <script nonce="${nonce}" src="${scriptUri}"></script>
  </body>
</html>`;
  }
}

function makeNonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let out = '';
  for (let i = 0; i < 32; i++) {
    out += chars.charAt(Math.floor(Math.random() * chars.length));
  }

  return out;
}
