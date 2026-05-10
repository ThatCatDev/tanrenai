import * as vscode from 'vscode';
import { loadCredentials, deleteCredentials } from './auth/credentials';
import { runLoginFlow } from './auth/login';
import { ChatViewProvider, ChatViewListener } from './chatViewProvider';
import { resolveCliPath } from './rpc/cliPath';
import { RPCClient } from './rpc/client';
import {
  ContentDeltaMsg,
  ReasoningDeltaMsg,
  ToolCallMsg,
  ToolCallRequestMsg,
  ToolResultLocalMsg,
  TurnDoneMsg,
  ErrorMsg,
} from './rpc/messages';
import { readSettings } from './settings';

const TANRENAI_WEB_URL = 'https://dev.tanrenai.com';
// Tools the extension intercepts. Real implementations land in the next
// commit; this commit stubs the result so the agent doesn't hang.
const INTERCEPTED_TOOLS = ['file_read', 'file_write', 'patch_file'];

export class Controller implements ChatViewListener {
  private rpc?: RPCClient;
  private logChannel: vscode.OutputChannel;
  private currentAssistantId?: string;
  private currentReasoningId?: string;
  private turnRunning = false;

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly view: ChatViewProvider,
  ) {
    this.logChannel = vscode.window.createOutputChannel('Tanrenai');
    context.subscriptions.push(this.logChannel);
    view.setListener(this);
  }

  async connect(): Promise<void> {
    await this.disconnect();

    this.view.setState({ status: 'connecting' });

    const creds = await loadCredentials();
    if (!creds || !creds.access_token) {
      this.view.setState({ status: 'no_credentials' });

      return;
    }

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
      if (this.rpc === rpc) {
        this.rpc = undefined;
        this.view.setState({
          status: 'error',
          message: 'CLI subprocess exited. Use Reconnect.',
        });
      }
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
      const message = (err as Error).message;
      this.log(`startup failed: ${message}`);
      this.view.setState({ status: 'error', message });
      await rpc.dispose();
    }
  }

  async disconnect(): Promise<void> {
    if (this.rpc) {
      await this.rpc.dispose();
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

    // Real intercepted-tool execution lands in the next commit. For now
    // stub the reply so the agent doesn't hang.
    if (intercepted) {
      this.rpc?.send({
        type: 'tool_result',
        id: m.id,
        ok: false,
        error: `tool ${m.name} interception not yet wired (Phase 4)`,
      });
    }
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
