import * as vscode from 'vscode';
import { ChatViewProvider } from './chatViewProvider';
import { Controller } from './controller';

export function activate(context: vscode.ExtensionContext): void {
  const view = new ChatViewProvider(context.extensionUri);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(ChatViewProvider.viewType, view),
  );

  const controller = new Controller(context, view);

  context.subscriptions.push(
    vscode.commands.registerCommand('tanrenai.login', () => controller.login()),
    vscode.commands.registerCommand('tanrenai.logout', () => controller.logout()),
    vscode.commands.registerCommand('tanrenai.reconnect', () => controller.reconnect()),
    controller.watchSettings(),
  );

  // Auto-connect on activation if credentials exist.
  void controller.connect();
}

export function deactivate(): void {
  // Subprocess cleanup happens via context.subscriptions disposers and
  // controller.disconnect() through the `exit` listener.
}
