import * as vscode from 'vscode';

export interface ResolvedSettings {
  serverUrlOverride: string;
  model: string;
  agentMode: boolean;
  cliPathOverride: string;
}

export function readSettings(): ResolvedSettings {
  const cfg = vscode.workspace.getConfiguration('tanrenai');

  return {
    serverUrlOverride: cfg.get<string>('serverUrl', '').trim(),
    model: cfg.get<string>('model', 'Qwen3.6-35B-A3B-UD-Q4_K_M'),
    agentMode: cfg.get<boolean>('agentMode', true),
    cliPathOverride: cfg.get<string>('cliPath', '').trim(),
  };
}
