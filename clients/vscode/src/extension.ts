import * as vscode from 'vscode';
import { ChatViewProvider } from './chatViewProvider';
import { Controller } from './controller';
import { ProposedContentProvider } from './diff/proposedProvider';

export function activate(context: vscode.ExtensionContext): void {
  const view = new ChatViewProvider(context.extensionUri);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(ChatViewProvider.viewType, view),
  );

  const proposed = new ProposedContentProvider();
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(ProposedContentProvider.scheme, proposed),
  );

  const controller = new Controller(context, view, proposed);

  context.subscriptions.push(
    vscode.commands.registerCommand('tanrenai.login', () => controller.login()),
    vscode.commands.registerCommand('tanrenai.logout', () => controller.logout()),
    vscode.commands.registerCommand('tanrenai.reconnect', () => controller.reconnect()),
    vscode.commands.registerCommand('tanrenai.pickModel', () => controller.pickModel()),
    vscode.commands.registerCommand('tanrenai.addSelection', () => {
      controller.onAttachRequest();
      // Reveal the chat so the user sees their selection landed.
      void vscode.commands.executeCommand('tanrenai.chat.focus');
    }),
    controller.watchSettings(),
    controller.watchEditorSelection(),
  );

  // Auto-connect on activation if credentials exist.
  void controller.connect();
}

export function deactivate(): void {
  // Subprocess cleanup happens via context.subscriptions disposers and
  // controller.disconnect() through the `exit` listener.
}
