import * as vscode from 'vscode';
import { loadCredentials, deleteCredentials } from './auth/credentials';
import { runLoginFlow } from './auth/login';
import { ChatViewProvider } from './chatViewProvider';
import { resolveCliPath } from './rpc/cliPath';
import { RPCClient } from './rpc/client';
import { readSettings } from './settings';

const TANRENAI_WEB_URL = 'https://dev.tanrenai.com';
// Tools the extension intercepts so they see editor buffers / use diff
// previews. Real implementations land in a later commit; for now the
// extension just advertises the names so the CLI knows which to forward.
const INTERCEPTED_TOOLS = ['file_read', 'file_write', 'patch_file'];

export class Controller {
  private rpc?: RPCClient;
  private logChannel: vscode.OutputChannel;

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly view: ChatViewProvider,
  ) {
    this.logChannel = vscode.window.createOutputChannel('Tanrenai');
    context.subscriptions.push(this.logChannel);
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
  }

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

  private log(line: string): void {
    this.logChannel.appendLine(`[tanrenai] ${line}`);
  }
}
