import * as vscode from 'vscode';

export type ConnectionState =
  | { status: 'idle' }
  | { status: 'no_credentials' }
  | { status: 'connecting'; progress?: string; warn?: boolean }
  | { status: 'connected'; model: string; toolCount: number }
  | { status: 'error'; message: string };

export type WebviewInbound =
  | { type: 'send'; content: string }
  | { type: 'cancel' }
  | { type: 'cancel_connect' }
  | { type: 'pick_model' }
  | { type: 'login' }
  | { type: 'reconnect' };

export type WebviewOutbound =
  | { type: 'state'; state: ConnectionState }
  | { type: 'turn_start' }
  | { type: 'turn_end'; ok: boolean; reason?: string }
  | { type: 'message_start'; role: 'user' | 'assistant' | 'tool'; id: string; meta?: string }
  | { type: 'message_delta'; id: string; text: string; channel?: 'content' | 'reasoning' }
  | { type: 'message_end'; id: string }
  | { type: 'tool_call'; id: string; name: string; arguments: string; intercepted: boolean }
  | { type: 'tool_result'; id: string; ok: boolean; content?: string };

export interface ChatViewListener {
  onSend(content: string): void;
  onCancel(): void;
  onCancelConnect(): void;
  onPickModel(): void;
  onLogin(): void;
  onReconnect(): void;
}

/**
 * Sidebar webview. Owns connection-state UI plus the chat surface (message
 * list, streaming append, input). Streaming messages are addressed by id so
 * the controller can pipe content_delta directly to the right bubble.
 */
export class ChatViewProvider implements vscode.WebviewViewProvider {
  static readonly viewType = 'tanrenai.chat';

  private view?: vscode.WebviewView;
  private state: ConnectionState = { status: 'idle' };
  private listener?: ChatViewListener;
  // Buffered messages for when the webview hasn't mounted yet.
  private pending: WebviewOutbound[] = [];

  constructor(private readonly extensionUri: vscode.Uri) {}

  setListener(listener: ChatViewListener): void {
    this.listener = listener;
  }

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, 'media')],
    };
    view.webview.html = this.renderHtml(view.webview);

    view.webview.onDidReceiveMessage((msg: WebviewInbound) => {
      switch (msg.type) {
        case 'send':
          this.listener?.onSend(msg.content);
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
        case 'login':
          this.listener?.onLogin();
          break;
        case 'reconnect':
          this.listener?.onReconnect();
          break;
      }
    });

    // Flush buffered messages.
    this.post({ type: 'state', state: this.state });
    for (const msg of this.pending) {
      this.post(msg);
    }
    this.pending = [];
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
      :root { color-scheme: light dark; }
      html, body { height: 100%; margin: 0; }
      body {
        font-family: var(--vscode-font-family);
        font-size: var(--vscode-font-size);
        color: var(--vscode-foreground);
        background: var(--vscode-sideBar-background);
        display: flex;
        flex-direction: column;
      }
      .header {
        padding: 0.5rem 0.75rem;
        font-size: 0.75rem;
        opacity: 0.7;
        border-bottom: 1px solid var(--vscode-panel-border, transparent);
        flex: 0 0 auto;
      }
      .header .ok { color: var(--vscode-charts-green); }
      .header .err { color: var(--vscode-charts-red); }
      .messages {
        flex: 1 1 auto;
        overflow-y: auto;
        padding: 0.5rem;
      }
      .msg {
        margin: 0.5rem 0;
        line-height: 1.4;
        white-space: pre-wrap;
        word-wrap: break-word;
      }
      .msg.user {
        background: var(--vscode-input-background);
        padding: 0.5rem 0.75rem;
        border-radius: 4px;
      }
      .msg.assistant { padding: 0 0.25rem; }
      .msg.reasoning { opacity: 0.6; font-size: 0.85em; padding: 0 0.25rem; }
      .role { font-size: 0.7rem; opacity: 0.6; text-transform: uppercase; margin-bottom: 0.25rem; }
      .tool {
        margin: 0.5rem 0;
        padding: 0.5rem 0.75rem;
        border-left: 2px solid var(--vscode-charts-blue);
        background: var(--vscode-textBlockQuote-background);
        font-family: var(--vscode-editor-font-family);
        font-size: 0.85em;
      }
      .tool.error { border-left-color: var(--vscode-charts-red); }
      .tool .name { font-weight: 600; }
      .tool .args { opacity: 0.7; word-break: break-all; }
      .tool details { margin-top: 0.25rem; }
      .tool details summary { cursor: pointer; opacity: 0.7; font-size: 0.85em; }
      pre {
        background: var(--vscode-textCodeBlock-background);
        padding: 0.5rem;
        border-radius: 4px;
        overflow-x: auto;
        font-family: var(--vscode-editor-font-family);
        font-size: 0.9em;
      }
      code { font-family: var(--vscode-editor-font-family); }
      .input-row {
        flex: 0 0 auto;
        padding: 0.5rem;
        border-top: 1px solid var(--vscode-panel-border, transparent);
      }
      textarea {
        width: 100%;
        box-sizing: border-box;
        background: var(--vscode-input-background);
        color: var(--vscode-input-foreground);
        border: 1px solid var(--vscode-input-border, transparent);
        padding: 0.5rem;
        font-family: var(--vscode-font-family);
        font-size: var(--vscode-font-size);
        resize: vertical;
        min-height: 2.5rem;
      }
      .actions { display: flex; gap: 0.5rem; margin-top: 0.5rem; }
      button {
        background: var(--vscode-button-background);
        color: var(--vscode-button-foreground);
        border: none;
        padding: 0.4rem 0.9rem;
        border-radius: 2px;
        cursor: pointer;
        font: inherit;
      }
      button:hover { background: var(--vscode-button-hoverBackground); }
      button:disabled { opacity: 0.5; cursor: default; }
      button.secondary {
        background: transparent;
        color: var(--vscode-foreground);
        border: 1px solid var(--vscode-panel-border, currentColor);
      }
      button.secondary:hover { background: var(--vscode-list-hoverBackground); }
      .status-panel {
        flex: 1 1 auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        padding: 2rem 1rem;
        text-align: center;
      }
      .status-panel .label { opacity: 0.7; }
      .status-panel.error .label { color: var(--vscode-charts-red); }
    </style>
  </head>
  <body>
    <div id="header" class="header" hidden></div>
    <div id="messages" class="messages" hidden></div>
    <div id="status" class="status-panel"></div>
    <div id="input-row" class="input-row" hidden>
      <textarea id="input" rows="2" placeholder="Ask Tanrenai…"></textarea>
      <div class="actions">
        <button id="send">Send</button>
        <button id="cancel" class="secondary" hidden>Cancel</button>
      </div>
    </div>

    <script>
      const vscode = acquireVsCodeApi();
      const headerEl = document.getElementById('header');
      const messagesEl = document.getElementById('messages');
      const statusEl = document.getElementById('status');
      const inputRow = document.getElementById('input-row');
      const inputEl = document.getElementById('input');
      const sendBtn = document.getElementById('send');
      const cancelBtn = document.getElementById('cancel');

      let connected = false;
      let turnRunning = false;
      // id → DOM node for streaming append
      const messageNodes = new Map();

      function escape(s) {
        return String(s)
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;');
      }

      function showConnected(state) {
        connected = true;
        headerEl.hidden = false;
        headerEl.innerHTML =
          '<span class="ok">●</span> ' +
          '<a href="#" id="hdrModel" style="color:inherit;text-decoration:underline dotted;">' +
          escape(state.model) + '</a>' +
          ' · ' + state.toolCount + ' tools';
        document.getElementById('hdrModel').addEventListener('click', (e) => {
          e.preventDefault();
          vscode.postMessage({ type: 'pick_model' });
        });
        messagesEl.hidden = false;
        inputRow.hidden = false;
        statusEl.innerHTML = '';
        statusEl.className = 'status-panel';
        statusEl.style.display = 'none';
      }

      function showStatus(state) {
        connected = false;
        headerEl.hidden = true;
        messagesEl.hidden = true;
        inputRow.hidden = true;
        statusEl.style.display = '';

        switch (state.status) {
          case 'idle':
            statusEl.className = 'status-panel';
            statusEl.innerHTML = '<div class="label">Initialising…</div>';
            break;
          case 'connecting':
            statusEl.className = 'status-panel';
            const progress = state.progress
              ? '<div class="label" style="margin-top:0.25rem; font-size:0.85em; opacity:0.8;' +
                (state.warn ? ' color: var(--vscode-charts-yellow);' : '') +
                '">' + escape(state.progress) + '</div>'
              : '';
            statusEl.innerHTML =
              '<div class="label">Connecting…</div>' + progress +
              '<div style="display:flex; gap:0.5rem; margin-top:0.75rem;">' +
              '<button id="cancelConnect" class="secondary">Cancel</button>' +
              '<button id="pickModel" class="secondary">Change Model…</button>' +
              '</div>';
            document.getElementById('cancelConnect').addEventListener('click', () => {
              vscode.postMessage({ type: 'cancel_connect' });
            });
            document.getElementById('pickModel').addEventListener('click', () => {
              vscode.postMessage({ type: 'pick_model' });
            });
            break;
          case 'no_credentials':
            statusEl.className = 'status-panel';
            statusEl.innerHTML = '<div class="label">Not signed in.</div>' +
              '<button id="login">Sign in to Tanrenai</button>';
            document.getElementById('login').addEventListener('click', () => {
              vscode.postMessage({ type: 'login' });
            });
            break;
          case 'error':
            statusEl.className = 'status-panel error';
            statusEl.innerHTML = '<div class="label">Error: ' + escape(state.message) + '</div>' +
              '<button id="retry">Retry</button>';
            document.getElementById('retry').addEventListener('click', () => {
              vscode.postMessage({ type: 'reconnect' });
            });
            break;
        }
      }

      function startMessage(id, role, meta) {
        const wrap = document.createElement('div');
        wrap.className = 'msg ' + role;
        wrap.dataset.id = id;
        const roleEl = document.createElement('div');
        roleEl.className = 'role';
        roleEl.textContent = meta ? role + ' · ' + meta : role;
        const body = document.createElement('div');
        body.className = 'body';
        wrap.appendChild(roleEl);
        wrap.appendChild(body);
        messagesEl.appendChild(wrap);
        messageNodes.set(id, body);
        scrollToBottom();
      }

      function appendDelta(id, text, channel) {
        let node = messageNodes.get(id);
        if (!node) {
          // First delta with no preceding message_start: synthesize one.
          startMessage(id, channel === 'reasoning' ? 'reasoning' : 'assistant');
          node = messageNodes.get(id);
        }
        node.appendChild(document.createTextNode(text));
        scrollToBottom();
      }

      function endMessage(id) {
        messageNodes.delete(id);
      }

      function renderToolCall(id, name, args, intercepted) {
        const card = document.createElement('div');
        card.className = 'tool';
        card.dataset.toolId = id;
        const tag = intercepted ? ' (editor)' : '';
        card.innerHTML =
          '<div class="name">' + escape(name) + escape(tag) + '</div>' +
          '<div class="args">' + escape(truncate(args, 200)) + '</div>' +
          '<details><summary>Result</summary>' +
          '<div class="result">running…</div></details>';
        messagesEl.appendChild(card);
        scrollToBottom();
      }

      function renderToolResult(id, ok, content) {
        const card = messagesEl.querySelector('[data-tool-id="' + id + '"]');
        if (!card) return;
        if (!ok) card.classList.add('error');
        const result = card.querySelector('.result');
        if (result) {
          result.textContent = content || (ok ? 'ok' : '(error)');
        }
      }

      function truncate(s, n) {
        return s.length > n ? s.slice(0, n) + '…' : s;
      }

      function scrollToBottom() {
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }

      function setTurnRunning(running) {
        turnRunning = running;
        cancelBtn.hidden = !running;
        sendBtn.disabled = running;
      }

      function send() {
        const text = inputEl.value.trim();
        if (!text || turnRunning || !connected) return;
        inputEl.value = '';
        const userId = 'u_' + Date.now();
        startMessage(userId, 'user');
        appendDelta(userId, text);
        endMessage(userId);
        vscode.postMessage({ type: 'send', content: text });
      }

      sendBtn.addEventListener('click', send);
      cancelBtn.addEventListener('click', () => vscode.postMessage({ type: 'cancel' }));
      inputEl.addEventListener('keydown', (e) => {
        // Enter sends; Shift+Enter inserts newline.
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          send();
        }
      });

      window.addEventListener('message', (event) => {
        const msg = event.data;
        switch (msg.type) {
          case 'state':
            if (msg.state.status === 'connected') {
              showConnected(msg.state);
            } else {
              showStatus(msg.state);
            }
            break;
          case 'turn_start':
            setTurnRunning(true);
            break;
          case 'turn_end':
            setTurnRunning(false);
            if (!msg.ok && msg.reason) {
              const wrap = document.createElement('div');
              wrap.className = 'msg reasoning';
              wrap.textContent = '[error] ' + msg.reason;
              messagesEl.appendChild(wrap);
              scrollToBottom();
            }
            break;
          case 'message_start':
            startMessage(msg.id, msg.role, msg.meta);
            break;
          case 'message_delta':
            appendDelta(msg.id, msg.text, msg.channel);
            break;
          case 'message_end':
            endMessage(msg.id);
            break;
          case 'tool_call':
            renderToolCall(msg.id, msg.name, msg.arguments, msg.intercepted);
            break;
          case 'tool_result':
            renderToolResult(msg.id, msg.ok, msg.content);
            break;
        }
      });
    </script>
  </body>
</html>`;
  }
}
