import * as vscode from 'vscode';
import { loadCredentials, deleteCredentials, saveCredentials } from './auth/credentials';
import { runLoginFlow } from './auth/login';
import { ChatViewProvider, ChatViewListener } from './chatViewProvider';
import { resolveCliPath } from './rpc/cliPath';
import { RPCClient } from './rpc/client';
import {
  ConnectingProgressMsg,
  ContentDeltaMsg,
  ReasoningDeltaMsg,
  ToolCallMsg,
  ToolCallRequestMsg,
  ToolResultLocalMsg,
  TurnDoneMsg,
  ErrorMsg,
} from './rpc/messages';
import { ProposedContentProvider } from './diff/proposedProvider';
import * as platform from './platform';
import { readSettings } from './settings';
import { dispatchTool, interceptedToolNames } from './tools/registry';
import { ApproveEditOpts } from './tools/types';
import { showModelPicker } from './ui/modelPicker';

const DEFAULT_MODEL = 'Qwen3.6-35B-A3B-UD-Q4_K_M';

type Mode = 'chat' | 'agent' | 'swarm';

type TranscriptEntry =
  | { kind: 'user'; id: string; content: string }
  | {
      kind: 'assistant';
      id: string;
      content: string;
      reasoning: string;
    }
  | {
      kind: 'tool';
      id: string;
      name: string;
      args: string;
      intercepted: boolean;
      result?: { ok: boolean; content?: string };
    }
  | {
      kind: 'approval';
      id: string;
      name: string;
      args: string;
      resolved: boolean;
    };

// Web frontend — handles the OAuth flow.
const TANRENAI_WEB_URL = 'https://dev.tanrenai.com';
// Backend API — handles model loading, completions, memory, etc. Separate
// host from the web frontend (the SvelteKit app at TANRENAI_WEB_URL
// serves /cli-login but not /api/*).
const TANRENAI_API_URL = 'https://api.dev.tanrenai.com';
const INTERCEPTED_TOOLS = interceptedToolNames;

export class Controller implements ChatViewListener {
  private rpc?: RPCClient;
  private logChannel: vscode.OutputChannel;
  private currentAssistantId?: string;
  private currentReasoningId?: string;
  private turnRunning = false;
  // Reentry guard so rapid Reconnect clicks don't spawn duplicates.
  private connecting = false;
  // Set true during disconnect() so the subprocess exit handler knows the
  // shutdown was intentional and shouldn't surface as an error.
  private intentionalDisconnect = false;
  // Accumulating chronological log surfaced in the connecting state — same
  // shape as the TUI's startup log (Allocating GPU…, Downloading model…, etc.)
  private progressLines: { message: string; level: 'info' | 'warn' }[] = [];
  // Source of truth for the rendered chat. The webview's DOM is disposed
  // when the sidebar view is hidden, so we replay this on every remount.
  private transcript: TranscriptEntry[] = [];
  // Current open assistant entry receiving deltas (one per channel).
  private openContentId?: string;
  private openReasoningId?: string;
  // Mode to send with each user_message. Persisted to settings on change.
  private mode: Mode = 'chat';
  // Pending edit approvals — resolved when the user clicks Allow/Deny on
  // the inline approval card for a proposed file change.
  private pendingEditApprovals = new Map<string, () => void>();
  private pendingEditDenials = new Map<string, () => void>();

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly view: ChatViewProvider,
    private readonly proposed: ProposedContentProvider,
  ) {
    this.logChannel = vscode.window.createOutputChannel('Tanrenai');
    context.subscriptions.push(this.logChannel);
    view.setListener(this);
  }

  async connect(): Promise<void> {
    if (this.connecting) {
      this.log('connect() ignored — already connecting');

      return;
    }
    this.connecting = true;

    try {
      await this.doConnect();
    } finally {
      this.connecting = false;
    }
  }

  private async doConnect(): Promise<void> {
    await this.disconnect();

    this.progressLines = [];
    this.view.setState({ status: 'connecting', progress: [] });

    const creds = await loadCredentials();
    if (!creds || !creds.access_token) {
      this.view.setState({ status: 'no_credentials' });

      return;
    }

    await this.maybeShowFirstRunModelPicker();

    const settings = readSettings();
    this.mode = settings.mode;
    this.view.send({ type: 'mode', mode: this.mode });
    this.appendProgress({ message: `Loading model ${settings.model}…`, level: 'info' });
    // If the stored server_url is the web frontend (the SvelteKit app that
    // serves /cli-login but not /api/*), fall back to the API host. This
    // self-heals credentials written by older versions of the extension
    // that didn't distinguish the two.
    let serverUrl = settings.serverUrlOverride || creds.server_url;
    if (!settings.serverUrlOverride && isLikelyWebFrontend(serverUrl)) {
      this.log(`server_url ${serverUrl} looks like the web frontend — using ${TANRENAI_API_URL} instead`);
      serverUrl = TANRENAI_API_URL;
    }

    const cliPath = resolveCliPath(this.context.extensionUri.fsPath, settings.cliPathOverride);
    this.log(`spawning ${cliPath} agent-rpc`);

    const rpc = new RPCClient({
      cliPath,
      env: {
        ...process.env,
        TANRENAI_SERVER_URL: serverUrl,
      },
      cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
    });

    rpc.on('stderr', (chunk: string) => this.logChannel.append(chunk));
    rpc.on('exit', (code: number | null, signal: NodeJS.Signals | null) => {
      this.log(`agent-rpc exited (code=${code}, signal=${signal})`);
      if (this.rpc !== rpc) {
        return; // not the active subprocess
      }
      this.rpc = undefined;
      if (this.intentionalDisconnect) {
        return; // we asked it to stop
      }
      const tail = rpc.recentStderr();
      const detail = tail
        ? `CLI subprocess exited unexpectedly. Last log:\n${truncate(tail, 400)}`
        : 'CLI subprocess exited unexpectedly. Use Reconnect.';
      this.view.setState({ status: 'error', message: detail });
    });

    rpc.on('connecting_progress', (m: ConnectingProgressMsg) => {
      this.appendProgress({ message: m.message, level: m.level });
    });
    rpc.on('history_cleared', () => {
      this.log('history cleared');
    });
    rpc.on('content_delta', (m: ContentDeltaMsg) => this.handleContentDelta(m.text));
    rpc.on('reasoning_delta', (m: ReasoningDeltaMsg) => this.handleReasoningDelta(m.text));
    rpc.on('tool_call', (m: ToolCallMsg) => this.handleToolCall(m, false));
    rpc.on('tool_call_request', (m: ToolCallRequestMsg) => this.handleToolCall(m, true));
    rpc.on('tool_result_local', (m: ToolResultLocalMsg) => this.handleToolResult(m));
    rpc.on('approval_required', (m: { id: string; name: string; arguments: string }) =>
      this.handleApprovalRequired(m),
    );
    rpc.on('turn_done', (m: TurnDoneMsg) => this.handleTurnDone(m));
    rpc.on('iteration_start', () => this.handleIterationStart());
    rpc.on('error', (m: ErrorMsg) => {
      this.log(`error: ${m.message}`);
      if (m.fatal) {
        this.view.setState({ status: 'error', message: m.message });
      }
    });

    try {
      const ready = await rpc.start({
        model: settings.model,
        agentMode: this.mode !== 'chat',
        swarmMode: this.mode === 'swarm',
        interceptedTools: INTERCEPTED_TOOLS,
        workspaceRoot: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? '',
      });
      this.rpc = rpc;
      this.view.setState({
        status: 'connected',
        model: ready.model,
        toolCount: ready.tools.length,
      });
      this.log(`ready: model=${ready.model}, tools=${ready.tools.length}`);
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      const tail = rpc.recentStderr();
      let message: string;
      if (code === 'ENOENT') {
        message =
          `Could not find the tanrenai CLI at "${cliPath}". ` +
          'Run `make vscode-bundle` from the repo root, or set "tanrenai.cliPath" to your CLI binary.';
      } else if (looksLikeHtmlResponseError(tail) || looksLikeHtmlResponseError((err as Error).message)) {
        message =
          `The backend at ${serverUrl} responded with an HTML page instead of API data.\n\n` +
          `That usually means tanrenai.serverUrl is pointing at the web frontend, not your ` +
          `tanrenai-server. Update it in VS Code settings to your backend URL ` +
          `(e.g. http://100.64.x.x:8080 for a vast.ai + tailnet setup).`;
      } else if (tail) {
        message = `${(err as Error).message}\n${truncate(tail, 400)}`;
      } else {
        message = (err as Error).message;
      }
      this.log(`startup failed: ${message}`);
      this.view.setState({ status: 'error', message });
      await rpc.dispose();
    }
  }

  async disconnect(): Promise<void> {
    if (this.rpc) {
      this.intentionalDisconnect = true;
      try {
        await this.rpc.dispose();
      } finally {
        this.intentionalDisconnect = false;
      }
      this.rpc = undefined;
    }
    this.turnRunning = false;
    this.currentAssistantId = undefined;
    this.currentReasoningId = undefined;
    this.openContentId = undefined;
    this.openReasoningId = undefined;
    this.progressLines = [];
    this.transcript = [];
  }

  // ── ChatViewListener ──────────────────────────────────────────────

  /**
   * Capture the active editor's selection (or the active line if there's
   * no selection) and forward it to the webview as an attachment proposal.
   */
  onAttachRequest(): void {
    const sel = captureActiveSelection({ acceptCursorLine: true });
    if (!sel) {
      void vscode.window.showInformationMessage('Tanrenai: open a file and select some code first.');

      return;
    }
    this.view.send({ type: 'attach_selection', selection: sel });
  }

  /**
   * Watches editor selection changes and pushes the current selection to the
   * webview as a live preview. The webview renders a hint above the composer
   * so the user knows their highlight is "seen" — clicking attaches it.
   */
  watchEditorSelection(): vscode.Disposable {
    const push = () => {
      const sel = captureActiveSelection({ acceptCursorLine: false });
      this.view.send({ type: 'available_selection', selection: sel });
    };
    const disposables: vscode.Disposable[] = [
      vscode.window.onDidChangeTextEditorSelection(() => push()),
      vscode.window.onDidChangeActiveTextEditor(() => push()),
    ];
    // Push the current state immediately too — covers the first mount.
    push();

    return vscode.Disposable.from(...disposables);
  }

  onSend(
    content: string,
    attachments?: import('./protocol').SelectionAttachment[],
    images?: import('./protocol').ImageAttachment[],
  ): void {
    if (!this.rpc) {
      this.view.send({ type: 'turn_end', ok: false, reason: 'not connected' });

      return;
    }
    if (this.turnRunning) {
      return;
    }
    this.turnRunning = true;
    this.currentAssistantId = undefined;
    this.currentReasoningId = undefined;
    this.openContentId = undefined;
    this.openReasoningId = undefined;

    // Fold any attached selections into the user content as fenced blocks
    // before either the model or the transcript see it.
    let fullContent = composeWithAttachments(content, attachments);
    // Append an attachment list for any image so the visible bubble (and
    // the chat history sent to the model) records what images were sent.
    if (images && images.length > 0) {
      const note = images.map((i) => `[image: ${i.label}]`).join(' ');
      fullContent = fullContent ? `${fullContent}\n\n${note}` : note;
    }

    // Record + render the user's turn.
    const userId = `u_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    this.transcript.push({ kind: 'user', id: userId, content: fullContent });
    this.view.send({ type: 'message_start', role: 'user', id: userId });
    this.view.send({ type: 'message_delta', id: userId, text: fullContent });
    this.view.send({ type: 'message_end', id: userId });
    this.view.send({ type: 'turn_start' });

    try {
      this.rpc.send({
        type: 'user_message',
        content: fullContent,
        mode: this.mode,
        images: images && images.length > 0 ? images.map((i) => i.dataUrl) : undefined,
      });
    } catch (err) {
      this.turnRunning = false;
      this.view.send({ type: 'turn_end', ok: false, reason: (err as Error).message });
    }
  }

  onSetMode(mode: Mode): void {
    if (this.mode === mode) {
      return;
    }
    this.mode = mode;
    void vscode.workspace
      .getConfiguration('tanrenai')
      .update('mode', mode, vscode.ConfigurationTarget.Global);
    this.view.send({ type: 'mode', mode });
    this.log(`mode → ${mode}`);
  }

  onClearChat(): void {
    this.transcript = [];
    this.openContentId = undefined;
    this.openReasoningId = undefined;
    this.view.send({ type: 'clear_chat' });
    if (this.rpc) {
      try {
        this.rpc.send({ type: 'clear_history' });
      } catch {
        // best-effort
      }
    }
  }

  onMounted(): void {
    // The webview was just resolved (initial mount or remount after being
    // hidden). Replay the transcript so chat history isn't lost.
    this.view.send({ type: 'mode', mode: this.mode });
    for (const entry of this.transcript) {
      this.replayEntry(entry);
    }
    if (this.turnRunning) {
      this.view.send({ type: 'turn_start' });
    }
  }

  private replayEntry(entry: TranscriptEntry): void {
    switch (entry.kind) {
      case 'user':
        this.view.send({ type: 'message_start', role: 'user', id: entry.id });
        this.view.send({ type: 'message_delta', id: entry.id, text: entry.content });
        this.view.send({ type: 'message_end', id: entry.id });
        break;
      case 'assistant':
        if (entry.reasoning) {
          this.view.send({
            type: 'message_start',
            role: 'assistant',
            id: entry.id + '_r',
            meta: 'thinking',
          });
          this.view.send({
            type: 'message_delta',
            id: entry.id + '_r',
            text: entry.reasoning,
            channel: 'reasoning',
          });
          this.view.send({ type: 'message_end', id: entry.id + '_r' });
        }
        if (entry.content) {
          this.view.send({ type: 'message_start', role: 'assistant', id: entry.id });
          this.view.send({ type: 'message_delta', id: entry.id, text: entry.content });
          this.view.send({ type: 'message_end', id: entry.id });
        }
        break;
      case 'tool':
        this.view.send({
          type: 'tool_call',
          id: entry.id,
          name: entry.name,
          arguments: entry.args,
          intercepted: entry.intercepted,
        });
        if (entry.result) {
          this.view.send({
            type: 'tool_result',
            id: entry.id,
            ok: entry.result.ok,
            content: entry.result.content,
          });
        }
        break;
      case 'approval':
        this.view.send({
          type: 'approval_required',
          id: entry.id,
          name: entry.name,
          arguments: entry.args,
        });
        if (entry.resolved) {
          this.view.send({ type: 'approval_resolved', id: entry.id });
        }
        break;
    }
  }

  onCancel(): void {
    if (this.rpc && this.turnRunning) {
      try {
        this.rpc.send({ type: 'cancel' });
      } catch {
        // best-effort
      }
    }
  }

  /** Called from the "Cancel" button shown during long connecting waits. */
  onCancelConnect(): void {
    this.log('user cancelled connect');
    void this.disconnect().then(() => {
      this.view.setState({ status: 'idle' });
    });
  }

  /** Called from the "Change Model…" link in the sidebar. */
  onPickModel(): void {
    void this.pickModel();
  }

  onLogin(): void {
    void this.login();
  }

  onLogout(): void {
    void this.logout();
  }

  onReconnect(): void {
    void this.connect();
  }

  async onStopGpu(): Promise<void> {
    const choice = await vscode.window.showWarningMessage(
      'Stop the GPU instance? This pauses it (vast.ai may still bill while paused). ' +
        'Your chat is preserved — the next message will wake it back up.',
      { modal: true },
      'Stop',
    );
    if (choice !== 'Stop') {
      return;
    }
    // Note: NOT disconnecting the CLI subprocess. deps.mgr (the canonical
    // conversation) stays alive. When the user sends the next message,
    // the withStreamGPURetry wrapper sees the provisioning error and
    // automatically re-wakes the instance with the full prior context.
    try {
      await platform.instanceStop();
      void vscode.window.showInformationMessage('Tanrenai: GPU stopped. Chat is preserved.');
    } catch (err) {
      const message = (err as Error).message;
      void vscode.window.showErrorMessage(`Tanrenai: stop failed — ${message}`);
    }
  }

  async onDestroyGpu(): Promise<void> {
    const choice = await vscode.window.showWarningMessage(
      'Destroy the GPU instance? This deletes it on vast.ai. ' +
        'Use this if instances are being spawned repeatedly without being torn down.',
      {
        modal: true,
        detail:
          'Chat is preserved. The next message will spin up a fresh GPU and the model ' +
          'will see the prior conversation. Cached models on the destroyed instance are lost ' +
          'and may need to re-pull.',
      },
      'Destroy',
    );
    if (choice !== 'Destroy') {
      return;
    }
    // Same as stop: don't kill the CLI subprocess. mgr survives;
    // next turn triggers a fresh spawn with full context.
    try {
      await platform.instanceDestroy();
      void vscode.window.showInformationMessage('Tanrenai: GPU destroyed. Chat is preserved.');
    } catch (err) {
      const message = (err as Error).message;
      void vscode.window.showErrorMessage(`Tanrenai: destroy failed — ${message}`);
    }
  }

  async onShowGpuStatus(): Promise<void> {
    try {
      const status = await platform.instanceStatus();
      this.logChannel.show(true);
      this.log(`GPU status: ${JSON.stringify(status, null, 2)}`);
    } catch (err) {
      const message = (err as Error).message;
      void vscode.window.showErrorMessage(`Tanrenai: status failed — ${message}`);
    }
  }

  // ── Command handlers ──────────────────────────────────────────────

  async login(): Promise<void> {
    const serverUrl = await this.resolveBackendUrl();
    if (!serverUrl) {
      return; // user cancelled the prompt
    }

    try {
      this.log(`login: webUrl=${TANRENAI_WEB_URL}, serverUrl=${serverUrl}`);
      await runLoginFlow({
        webUrl: TANRENAI_WEB_URL,
        serverUrl,
        log: (msg) => this.log(`login: ${msg}`),
      });
      this.log('login: success');
      void vscode.window.showInformationMessage('Tanrenai: signed in.');
      await this.connect();
    } catch (err) {
      const message = (err as Error).message;
      this.log(`login: failed — ${message}`);
      void vscode.window.showErrorMessage(`Tanrenai login failed: ${message}`);
    }
  }

  /**
   * Figure out which URL to use for backend API calls. Priority:
   *   1. `tanrenai.serverUrl` setting
   *   2. existing creds' server_url, IF it isn't the web frontend URL
   *      (which would point at the SvelteKit 404 page for /api/* routes)
   *   3. prompt the user, validate, persist to global settings
   *
   * Returns undefined if the user cancels the prompt.
   */
  private async resolveBackendUrl(): Promise<string | undefined> {
    const settings = readSettings();
    if (settings.serverUrlOverride) {
      return settings.serverUrlOverride;
    }
    const existing = await loadCredentials();
    if (
      existing?.server_url &&
      !isLikelyWebFrontend(existing.server_url)
    ) {
      return existing.server_url;
    }

    // Default: the platform's hosted API. Matches what the terminal CLI's
    // `tanrenai login --platform-url https://api.dev.tanrenai.com` uses.
    // Persist it so future activations don't re-prompt.
    await vscode.workspace
      .getConfiguration('tanrenai')
      .update('serverUrl', TANRENAI_API_URL, vscode.ConfigurationTarget.Global);
    this.log(`login: defaulted tanrenai.serverUrl = ${TANRENAI_API_URL}`);

    return TANRENAI_API_URL;
  }

  async logout(): Promise<void> {
    await this.disconnect();
    // Preserve server_url across logout so a subsequent sign-in knows
    // where the platform lives. Otherwise we'd fall back to the web URL
    // as the API URL (wrong — that's the frontend, behind Cloudflare).
    const existing = await loadCredentials();
    if (existing?.server_url) {
      await saveCredentials({
        server_url: existing.server_url,
        access_token: '',
      });
    } else {
      await deleteCredentials();
    }
    this.view.setState({ status: 'no_credentials' });
    void vscode.window.showInformationMessage('Tanrenai: logged out.');
  }

  async reconnect(): Promise<void> {
    await this.connect();
  }

  /**
   * Show the model picker and persist the choice. Reconnects automatically
   * if the extension is already connected so the change takes effect.
   */
  async pickModel(): Promise<void> {
    const settings = readSettings();
    const choice = await showModelPicker(settings.model);
    if (!choice) {
      return;
    }
    await vscode.workspace
      .getConfiguration('tanrenai')
      .update('model', choice, vscode.ConfigurationTarget.Global);
    void vscode.window.showInformationMessage(`Tanrenai: model set to ${choice}.`);
    if (this.rpc) {
      await this.connect();
    }
  }

  /**
   * On first activation (after login), nudge the user to confirm or change
   * the default model before we kick off a potentially-long auto-pull.
   * Stored in globalState so we never ask twice.
   */
  private async maybeShowFirstRunModelPicker(): Promise<void> {
    const KEY = 'tanrenai.firstRunModelPickerShown';
    if (this.context.globalState.get<boolean>(KEY)) {
      return;
    }
    const cfg = vscode.workspace.getConfiguration('tanrenai');
    const inspected = cfg.inspect<string>('model');
    const userOverridden =
      (inspected?.globalValue !== undefined && inspected.globalValue !== '') ||
      (inspected?.workspaceValue !== undefined && inspected.workspaceValue !== '') ||
      (inspected?.workspaceFolderValue !== undefined && inspected.workspaceFolderValue !== '');
    if (userOverridden) {
      await this.context.globalState.update(KEY, true);

      return;
    }
    const choice = await vscode.window.showInformationMessage(
      `Tanrenai: about to load ${DEFAULT_MODEL}. Pick a different model first?`,
      'Pick Model',
      'Use Default',
    );
    await this.context.globalState.update(KEY, true);
    if (choice === 'Pick Model') {
      await this.pickModel();
    }
  }

  /**
   * Watch the user's tanrenai.* settings and offer to reconnect when one of
   * the load-bearing values changes — model/url/agentMode/cliPath only take
   * effect at the next handshake.
   */
  watchSettings(): vscode.Disposable {
    const reconnectKeys = ['tanrenai.model', 'tanrenai.serverUrl', 'tanrenai.cliPath'];
    const modeKey = 'tanrenai.mode';

    return vscode.workspace.onDidChangeConfiguration(async (event) => {
      // Mode changes are cheap (per-turn override) — absorb without prompting.
      if (event.affectsConfiguration(modeKey)) {
        const settings = readSettings();
        this.mode = settings.mode;
        this.view.send({ type: 'mode', mode: this.mode });
      }
      const changed = reconnectKeys.find((k) => event.affectsConfiguration(k));
      if (!changed) {
        return;
      }
      if (!this.rpc) {
        return;
      }
      const choice = await vscode.window.showInformationMessage(
        `Tanrenai: ${changed} changed. Reconnect to apply?`,
        'Reconnect',
        'Later',
      );
      if (choice === 'Reconnect') {
        await this.connect();
      }
    });
  }

  // ── RPC event handlers ────────────────────────────────────────────

  private handleContentDelta(text: string): void {
    if (!this.currentAssistantId) {
      this.currentAssistantId = `a_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      this.openContentId = this.currentAssistantId;
      this.transcript.push({
        kind: 'assistant',
        id: this.currentAssistantId,
        content: '',
        reasoning: '',
      });
      this.view.send({ type: 'message_start', role: 'assistant', id: this.currentAssistantId });
    }
    this.appendToOpenAssistant('content', text);
    this.view.send({ type: 'message_delta', id: this.currentAssistantId, text, channel: 'content' });
  }

  private handleReasoningDelta(text: string): void {
    if (!this.currentReasoningId) {
      this.currentReasoningId = `r_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      this.openReasoningId = this.currentReasoningId;
      this.transcript.push({
        kind: 'assistant',
        id: this.currentReasoningId,
        content: '',
        reasoning: '',
      });
      this.view.send({
        type: 'message_start',
        role: 'assistant',
        id: this.currentReasoningId,
        meta: 'thinking',
      });
    }
    this.appendToOpenAssistant('reasoning', text);
    this.view.send({
      type: 'message_delta',
      id: this.currentReasoningId,
      text,
      channel: 'reasoning',
    });
  }

  private appendToOpenAssistant(channel: 'content' | 'reasoning', text: string): void {
    const id = channel === 'content' ? this.openContentId : this.openReasoningId;
    if (!id) {
      return;
    }
    for (let i = this.transcript.length - 1; i >= 0; i--) {
      const e = this.transcript[i];
      if (e.kind === 'assistant' && e.id === id) {
        if (channel === 'content') {
          e.content += text;
        } else {
          e.reasoning += text;
        }

        return;
      }
    }
  }

  private handleToolCall(m: { id: string; name: string; arguments: string }, intercepted: boolean): void {
    this.closeOpenAssistantBubbles();
    this.transcript.push({
      kind: 'tool',
      id: m.id,
      name: m.name,
      args: m.arguments,
      intercepted,
    });
    this.view.send({
      type: 'tool_call',
      id: m.id,
      name: m.name,
      arguments: m.arguments,
      intercepted,
    });

    if (intercepted) {
      void this.executeInterceptedTool(m);
    }
  }

  private async executeInterceptedTool(m: {
    id: string;
    name: string;
    arguments: string;
  }): Promise<void> {
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? '';
    const result = await dispatchTool(
      m.name,
      m.arguments,
      workspaceRoot,
      (opts) => this.approveEdit(opts),
    );
    this.recordToolResult(m.id, result.ok, result.ok ? result.content : result.error);
    this.view.send({
      type: 'tool_result',
      id: m.id,
      ok: result.ok,
      content: result.ok ? result.content : result.error,
    });
    this.rpc?.send({
      type: 'tool_result',
      id: m.id,
      ok: result.ok,
      content: result.content,
      error: result.error,
    });
  }

  /**
   * Show the agent's proposed edit in VS Code's native diff editor and
   * post an inline approval card. Resolves with the user's decision.
   * Each pending approval gets a unique virtual URI so concurrent edits
   * don't clobber each other.
   */
  private async approveEdit(opts: ApproveEditOpts): Promise<boolean> {
    const id = `edit_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const previewUri = this.proposed.uriFor(opts.uri, id);
    this.proposed.set(previewUri, opts.proposed);

    try {
      if (opts.original !== undefined) {
        await vscode.commands.executeCommand(
          'vscode.diff',
          opts.uri,
          previewUri,
          `Tanrenai · ${opts.label}`,
          { preserveFocus: true, preview: true } as vscode.TextDocumentShowOptions,
        );
      } else {
        // New file — no left-hand diff base; just show the proposed content.
        await vscode.commands.executeCommand('vscode.open', previewUri, {
          preserveFocus: true,
          preview: true,
        } as vscode.TextDocumentShowOptions);
      }
    } catch (err) {
      this.log(`approveEdit: failed to open preview: ${(err as Error).message}`);
    }

    // Inline approval card in the chat.
    this.closeOpenAssistantBubbles();
    const previewArgs = JSON.stringify({ path: opts.label, summary: opts.summary });
    this.transcript.push({
      kind: 'approval',
      id,
      name: 'apply_edit',
      args: previewArgs,
      resolved: false,
    });
    this.view.send({
      type: 'approval_required',
      id,
      name: 'apply_edit',
      arguments: previewArgs,
    });

    return new Promise<boolean>((resolve) => {
      this.pendingEditApprovals.set(id, () => {
        this.proposed.clear(previewUri);
        resolve(true);
      });
      this.pendingEditDenials.set(id, () => {
        this.proposed.clear(previewUri);
        resolve(false);
      });
    });
  }

  private handleToolResult(m: ToolResultLocalMsg): void {
    this.recordToolResult(m.id, m.ok, m.ok ? m.content : m.error);
    this.view.send({
      type: 'tool_result',
      id: m.id,
      ok: m.ok,
      content: m.ok ? m.content : m.error,
    });
  }

  private recordToolResult(id: string, ok: boolean, content?: string): void {
    for (let i = this.transcript.length - 1; i >= 0; i--) {
      const e = this.transcript[i];
      if (e.kind === 'tool' && e.id === id) {
        e.result = { ok, content };

        return;
      }
    }
  }

  private closeOpenAssistantBubbles(): void {
    if (this.currentAssistantId) {
      this.view.send({ type: 'message_end', id: this.currentAssistantId });
      this.currentAssistantId = undefined;
      this.openContentId = undefined;
    }
    if (this.currentReasoningId) {
      this.view.send({ type: 'message_end', id: this.currentReasoningId });
      this.currentReasoningId = undefined;
      this.openReasoningId = undefined;
    }
  }

  private handleIterationStart(): void {
    this.closeOpenAssistantBubbles();
  }

  private handleApprovalRequired(m: { id: string; name: string; arguments: string }): void {
    // Surface inline in the chat instead of a VS Code modal — less jarring,
    // and the user can see the request alongside the conversation context.
    this.closeOpenAssistantBubbles();
    this.transcript.push({
      kind: 'approval',
      id: m.id,
      name: m.name,
      args: m.arguments,
      resolved: false,
    });
    this.view.send({
      type: 'approval_required',
      id: m.id,
      name: m.name,
      arguments: m.arguments,
    });
  }

  /** Called from the inline Allow/Always/Deny buttons. */
  onApprovalDecision(id: string, action: 'allow' | 'deny' | 'always'): void {
    // Mark the entry resolved so the buttons disappear.
    for (const entry of this.transcript) {
      if (entry.kind === 'approval' && entry.id === id) {
        entry.resolved = true;
        break;
      }
    }
    this.view.send({ type: 'approval_resolved', id });

    // Edit approvals are resolved locally in the extension (TS shim is
    // awaiting the promise). CLI-driven approvals (shell_exec, etc.) are
    // forwarded to the agent-rpc subprocess.
    const allow = action !== 'deny';
    const editAllow = this.pendingEditApprovals.get(id);
    const editDeny = this.pendingEditDenials.get(id);
    if (editAllow || editDeny) {
      this.pendingEditApprovals.delete(id);
      this.pendingEditDenials.delete(id);
      if (allow) {
        editAllow?.();
      } else {
        editDeny?.();
      }

      return;
    }

    this.rpc?.send({ type: 'approval_response', id, action });
  }

  private handleTurnDone(m: TurnDoneMsg): void {
    this.closeOpenAssistantBubbles();
    this.turnRunning = false;
    this.view.send({ type: 'turn_end', ok: m.ok, reason: m.reason });
  }

  private log(line: string): void {
    this.logChannel.appendLine(`[tanrenai] ${line}`);
  }

  /**
   * Append a line to the connecting-state progress log and re-render. Skips
   * consecutive duplicates so the polling loop's repeated "Downloading
   * model (40%)…" emits don't pile up. Caps the buffer so a runaway emitter
   * can't grow it without bound.
   */
  private appendProgress(line: { message: string; level: 'info' | 'warn' }): void {
    const last = this.progressLines[this.progressLines.length - 1];
    if (last && last.message === line.message && last.level === line.level) {
      return;
    }
    this.progressLines.push(line);
    if (this.progressLines.length > 200) {
      this.progressLines.splice(0, this.progressLines.length - 200);
    }
    this.log(`progress${line.level === 'warn' ? ' (warn)' : ''}: ${line.message}`);
    this.view.setState({ status: 'connecting', progress: [...this.progressLines] });
  }
}

/**
 * Read the active editor's selection. With `acceptCursorLine`, an empty
 * selection (just a cursor position) returns the current line; otherwise
 * it returns null. Returns null when no editor is active.
 */
function captureActiveSelection(opts: { acceptCursorLine: boolean }):
  | import('./protocol').SelectionAttachment
  | null {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return null;
  const sel = editor.selection;
  if (sel.isEmpty && !opts.acceptCursorLine) return null;
  const range = sel.isEmpty
    ? editor.document.lineAt(sel.active.line).range
    : new vscode.Range(sel.start, sel.end);
  const text = editor.document.getText(range);
  if (!text.trim()) return null;
  const path = vscode.workspace.asRelativePath(editor.document.uri, false);
  const startLine = range.start.line + 1;
  const endLine = range.end.line + 1;
  const label = startLine === endLine ? `${path}:${startLine}` : `${path}:${startLine}-${endLine}`;

  return {
    label,
    path,
    languageId: editor.document.languageId,
    startLine,
    endLine,
    text,
  };
}

/**
 * Prepend each selection attachment to the user's typed content as a fenced
 * code block with a path/range header. The fenced language tag uses the
 * editor's languageId so the model gets a syntax hint.
 */
function composeWithAttachments(
  content: string,
  attachments?: import('./protocol').SelectionAttachment[],
): string {
  if (!attachments || attachments.length === 0) {
    return content;
  }
  const blocks = attachments.map((a) => {
    const fence = '```' + (a.languageId || '');

    return `From \`${a.label}\`:\n${fence}\n${a.text}\n\`\`\``;
  });
  if (!content.trim()) {
    return blocks.join('\n\n');
  }

  return `${blocks.join('\n\n')}\n\n${content}`;
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + '…' : s;
}

/**
 * Heuristic: is this URL the Tanrenai web frontend rather than the API
 * backend? Web frontend lives at tanrenai.com / *.tanrenai.com and serves
 * the SvelteKit app — its /api/* routes 404 from the CLI's perspective.
 * The real backend is wherever the user runs `tanrenai-server`.
 */
function isLikelyWebFrontend(url: string): boolean {
  try {
    const host = new URL(url).host.toLowerCase();

    return host === 'tanrenai.com' || host.endsWith('.tanrenai.com');
  } catch {
    return false;
  }
}

/**
 * Detect responses that look like an HTML page (web frontend) instead of
 * the JSON the API should return. Triggers a clearer error message.
 */
function looksLikeHtmlResponseError(s: string | undefined): boolean {
  if (!s) return false;
  const lower = s.toLowerCase();

  return lower.includes('<!doctype html') || lower.includes('<html');
}
