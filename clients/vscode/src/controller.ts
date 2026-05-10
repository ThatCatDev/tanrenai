import * as vscode from 'vscode';
import { loadCredentials, deleteCredentials } from './auth/credentials';
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
import { readSettings } from './settings';
import { dispatchTool, interceptedToolNames } from './tools/registry';
import { showModelPicker } from './ui/modelPicker';

const DEFAULT_MODEL = 'Qwen3.6-35B-A3B-UD-Q4_K_M';

const TANRENAI_WEB_URL = 'https://dev.tanrenai.com';
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

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly view: ChatViewProvider,
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

    this.view.setState({ status: 'connecting' });

    const creds = await loadCredentials();
    if (!creds || !creds.access_token) {
      this.view.setState({ status: 'no_credentials' });

      return;
    }

    await this.maybeShowFirstRunModelPicker();

    const settings = readSettings();
    const serverUrl = settings.serverUrlOverride || creds.server_url;

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
      this.log(`progress: ${m.message}`);
      this.view.setState({
        status: 'connecting',
        progress: m.message,
        warn: m.level === 'warn',
      });
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
        agentMode: settings.agentMode,
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
  }

  // ── ChatViewListener ──────────────────────────────────────────────

  onSend(content: string): void {
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
    this.view.send({ type: 'turn_start' });
    try {
      this.rpc.send({ type: 'user_message', content });
    } catch (err) {
      this.turnRunning = false;
      this.view.send({ type: 'turn_end', ok: false, reason: (err as Error).message });
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

  onReconnect(): void {
    void this.connect();
  }

  // ── Command handlers ──────────────────────────────────────────────

  async login(): Promise<void> {
    const settings = readSettings();
    const existingCreds = await loadCredentials();
    const serverUrl = settings.serverUrlOverride || existingCreds?.server_url || TANRENAI_WEB_URL;

    try {
      await runLoginFlow({ webUrl: TANRENAI_WEB_URL, serverUrl });
      void vscode.window.showInformationMessage('Tanrenai: signed in.');
      await this.connect();
    } catch (err) {
      const message = (err as Error).message;
      void vscode.window.showErrorMessage(`Tanrenai login failed: ${message}`);
    }
  }

  async logout(): Promise<void> {
    await this.disconnect();
    await deleteCredentials();
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
    const reconnectKeys = ['tanrenai.model', 'tanrenai.serverUrl', 'tanrenai.agentMode', 'tanrenai.cliPath'];

    return vscode.workspace.onDidChangeConfiguration(async (event) => {
      const changed = reconnectKeys.find((k) => event.affectsConfiguration(k));
      if (!changed) {
        return;
      }
      // Quiet path: not connected yet → just absorb the change for next connect.
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
      this.view.send({ type: 'message_start', role: 'assistant', id: this.currentAssistantId });
    }
    this.view.send({ type: 'message_delta', id: this.currentAssistantId, text, channel: 'content' });
  }

  private handleReasoningDelta(text: string): void {
    if (!this.currentReasoningId) {
      this.currentReasoningId = `r_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      this.view.send({
        type: 'message_start',
        role: 'assistant',
        id: this.currentReasoningId,
        meta: 'thinking',
      });
    }
    this.view.send({
      type: 'message_delta',
      id: this.currentReasoningId,
      text,
      channel: 'reasoning',
    });
  }

  private handleToolCall(m: { id: string; name: string; arguments: string }, intercepted: boolean): void {
    if (this.currentAssistantId) {
      this.view.send({ type: 'message_end', id: this.currentAssistantId });
      this.currentAssistantId = undefined;
    }
    if (this.currentReasoningId) {
      this.view.send({ type: 'message_end', id: this.currentReasoningId });
      this.currentReasoningId = undefined;
    }
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
    const result = await dispatchTool(m.name, m.arguments, workspaceRoot);
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

  private handleToolResult(m: ToolResultLocalMsg): void {
    this.view.send({
      type: 'tool_result',
      id: m.id,
      ok: m.ok,
      content: m.ok ? m.content : m.error,
    });
  }

  private handleIterationStart(): void {
    if (this.currentAssistantId) {
      this.view.send({ type: 'message_end', id: this.currentAssistantId });
      this.currentAssistantId = undefined;
    }
    if (this.currentReasoningId) {
      this.view.send({ type: 'message_end', id: this.currentReasoningId });
      this.currentReasoningId = undefined;
    }
  }

  private handleApprovalRequired(m: { id: string; name: string; arguments: string }): void {
    void vscode.window
      .showWarningMessage(
        `Tanrenai wants to run ${m.name}: ${truncate(m.arguments, 200)}`,
        { modal: true },
        'Allow once',
        'Always allow',
        'Deny',
      )
      .then((choice) => {
        const action =
          choice === 'Always allow' ? 'always' : choice === 'Allow once' ? 'allow' : 'deny';
        this.rpc?.send({ type: 'approval_response', id: m.id, action });
      });
  }

  private handleTurnDone(m: TurnDoneMsg): void {
    if (this.currentAssistantId) {
      this.view.send({ type: 'message_end', id: this.currentAssistantId });
      this.currentAssistantId = undefined;
    }
    if (this.currentReasoningId) {
      this.view.send({ type: 'message_end', id: this.currentReasoningId });
      this.currentReasoningId = undefined;
    }
    this.turnRunning = false;
    this.view.send({ type: 'turn_end', ok: m.ok, reason: m.reason });
  }

  private log(line: string): void {
    this.logChannel.appendLine(`[tanrenai] ${line}`);
  }
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + '…' : s;
}
