import * as vscode from 'vscode';

export interface ProgressLine {
  message: string;
  level: 'info' | 'warn';
}

export type ConnectionState =
  | { status: 'idle' }
  | { status: 'no_credentials' }
  | { status: 'connecting'; progress: ProgressLine[] }
  | { status: 'connected'; model: string; toolCount: number }
  | { status: 'error'; message: string };

export type WebviewInbound =
  | { type: 'send'; content: string }
  | { type: 'cancel' }
  | { type: 'cancel_connect' }
  | { type: 'pick_model' }
  | { type: 'clear_chat' }
  | { type: 'set_mode'; mode: 'chat' | 'agent' | 'swarm' }
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
  | { type: 'tool_result'; id: string; ok: boolean; content?: string }
  | { type: 'clear_chat' }
  | { type: 'mode'; mode: 'chat' | 'agent' | 'swarm' };

export interface ChatViewListener {
  onSend(content: string): void;
  onCancel(): void;
  onCancelConnect(): void;
  onPickModel(): void;
  onClearChat(): void;
  onSetMode(mode: 'chat' | 'agent' | 'swarm'): void;
  onLogin(): void;
  onReconnect(): void;
  /** Called after the webview has finished mounting (or remounting). */
  onMounted(): void;
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
        case 'clear_chat':
          this.listener?.onClearChat();
          break;
        case 'set_mode':
          this.listener?.onSetMode(msg.mode);
          break;
        case 'login':
          this.listener?.onLogin();
          break;
        case 'reconnect':
          this.listener?.onReconnect();
          break;
      }
    });

    // Push the connection state immediately, then flush any buffered messages
    // (typically empty), then notify the controller so it can replay the
    // transcript — the webview was likely just remounted after being hidden.
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
        padding: 0.4rem 0.5rem;
        font-size: 0.75rem;
        opacity: 0.85;
        border-bottom: 1px solid var(--vscode-panel-border, transparent);
        flex: 0 0 auto;
        display: flex;
        align-items: center;
        gap: 0.4rem;
      }
      .header .meta { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .header .ok { color: var(--vscode-charts-green); }
      .header .err { color: var(--vscode-charts-red); }
      .modes {
        display: flex;
        gap: 1px;
        background: var(--vscode-input-background);
        border-radius: 4px;
        padding: 1px;
        flex: 0 0 auto;
      }
      .modes button {
        background: transparent;
        color: var(--vscode-foreground);
        border: none;
        padding: 0.2rem 0.5rem;
        font-size: 0.7rem;
        border-radius: 3px;
        cursor: pointer;
      }
      .modes button.active {
        background: var(--vscode-button-background);
        color: var(--vscode-button-foreground);
      }
      .modes button:not(.active):hover { background: var(--vscode-list-hoverBackground); }
      .icon-btn {
        background: transparent;
        color: var(--vscode-foreground);
        border: none;
        padding: 0.25rem 0.4rem;
        font-size: 0.85rem;
        border-radius: 3px;
        cursor: pointer;
        opacity: 0.7;
      }
      .icon-btn:hover { background: var(--vscode-list-hoverBackground); opacity: 1; }
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
      .progress-log {
        margin-top: 0.75rem;
        padding: 0.5rem 0.75rem;
        max-height: 240px;
        width: 100%;
        max-width: 28rem;
        overflow-y: auto;
        background: var(--vscode-textCodeBlock-background);
        border-radius: 4px;
        font-family: var(--vscode-editor-font-family);
        font-size: 0.8em;
        text-align: left;
        line-height: 1.45;
        box-sizing: border-box;
      }
      .progress-log .line { white-space: pre-wrap; opacity: 0.85; }
      .progress-log .line.warn { color: var(--vscode-charts-yellow); }
      .progress-log .line.info { opacity: 0.85; }
      .spinner {
        display: inline-block;
        width: 0.6em;
        height: 0.6em;
        margin-right: 0.4em;
        border: 2px solid currentColor;
        border-right-color: transparent;
        border-radius: 50%;
        vertical-align: middle;
        animation: spin 0.9s linear infinite;
      }
      @keyframes spin { to { transform: rotate(360deg); } }
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
          '<span class="ok">●</span>' +
          '<span class="meta"><a href="#" id="hdrModel" style="color:inherit;text-decoration:underline dotted;">' +
          escape(state.model) + '</a> · ' + state.toolCount + ' tools</span>' +
          '<div class="modes" role="tablist">' +
          '<button data-mode="chat" id="m_chat" title="Chat — no tools">Chat</button>' +
          '<button data-mode="agent" id="m_agent" title="Agent — single agent with tools">Agent</button>' +
          '<button data-mode="swarm" id="m_swarm" title="Swarm — multi-agent orchestrator">Swarm</button>' +
          '</div>' +
          '<button class="icon-btn" id="hdrClear" title="Clear chat">✕</button>';
        document.getElementById('hdrModel').addEventListener('click', (e) => {
          e.preventDefault();
          vscode.postMessage({ type: 'pick_model' });
        });
        document.getElementById('hdrClear').addEventListener('click', () => {
          vscode.postMessage({ type: 'clear_chat' });
        });
        ['chat','agent','swarm'].forEach((m) => {
          document.getElementById('m_' + m).addEventListener('click', () => {
            vscode.postMessage({ type: 'set_mode', mode: m });
          });
        });
        applyModeActive();
        messagesEl.hidden = false;
        inputRow.hidden = false;
        statusEl.innerHTML = '';
        statusEl.className = 'status-panel';
        statusEl.style.display = 'none';
      }

      let currentMode = 'agent';
      function applyModeActive() {
        ['chat','agent','swarm'].forEach((m) => {
          const el = document.getElementById('m_' + m);
          if (el) el.classList.toggle('active', m === currentMode);
        });
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
            const lines = (state.progress || [])
              .map(p => '<div class="line ' + (p.level === 'warn' ? 'warn' : 'info') + '">' +
                escape(p.message) + '</div>')
              .join('');
            const log = lines
              ? '<div class="progress-log" id="progressLog">' + lines + '</div>'
              : '';
            statusEl.innerHTML =
              '<div class="label"><span class="spinner"></span>Connecting…</div>' +
              log +
              '<div style="display:flex; gap:0.5rem; margin-top:0.75rem;">' +
              '<button id="cancelConnect" class="secondary">Cancel</button>' +
              '<button id="pickModel" class="secondary">Change Model…</button>' +
              '</div>';
            const logEl = document.getElementById('progressLog');
            if (logEl) logEl.scrollTop = logEl.scrollHeight;
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
        if (!text || !connected) return;

        // Slash-command shortcuts. These don't go to the model — they map
        // to the same UI actions as the buttons in the header.
        if (text.startsWith('/')) {
          const handled = handleSlashCommand(text);
          if (handled) {
            inputEl.value = '';

            return;
          }
        }
        if (turnRunning) return;

        inputEl.value = '';
        vscode.postMessage({ type: 'send', content: text });
      }

      function handleSlashCommand(text) {
        const parts = text.trim().split(/\s+/);
        const cmd = parts[0].toLowerCase();
        switch (cmd) {
          case '/clear':
            vscode.postMessage({ type: 'clear_chat' });

            return true;
          case '/chat':
            vscode.postMessage({ type: 'set_mode', mode: 'chat' });

            return true;
          case '/agent':
            vscode.postMessage({ type: 'set_mode', mode: 'agent' });

            return true;
          case '/swarm':
            if (parts[1] === 'off') {
              vscode.postMessage({ type: 'set_mode', mode: 'agent' });
            } else {
              vscode.postMessage({ type: 'set_mode', mode: 'swarm' });
            }

            return true;
          case '/model':
            vscode.postMessage({ type: 'pick_model' });

            return true;
          case '/help':
            renderHelp();

            return true;
          default:
            return false;
        }
      }

      function renderHelp() {
        const wrap = document.createElement('div');
        wrap.className = 'msg reasoning';
        wrap.textContent =
          'Slash commands:\n' +
          '  /clear              Clear chat\n' +
          '  /chat               Switch to plain chat\n' +
          '  /agent              Switch to agent mode\n' +
          '  /swarm [off]        Toggle swarm mode\n' +
          '  /model              Choose model\n' +
          '  /help               Show this list';
        messagesEl.appendChild(wrap);
        scrollToBottom();
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
          case 'mode':
            currentMode = msg.mode;
            applyModeActive();
            break;
          case 'clear_chat':
            messagesEl.innerHTML = '';
            messageNodes.clear();
            break;
        }
      });
    </script>
  </body>
</html>`;
  }
}
