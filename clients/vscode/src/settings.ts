import * as vscode from 'vscode';

export type Mode = 'chat' | 'agent' | 'swarm';

export interface ResolvedSettings {
  serverUrlOverride: string;
  model: string;
  mode: Mode;
  cliPathOverride: string;
}

export function readSettings(): ResolvedSettings {
  const cfg = vscode.workspace.getConfiguration('tanrenai');

  // tanrenai.mode is the new canonical setting; tanrenai.agentMode is the
  // legacy boolean. If mode is unset, derive from agentMode for backwards
  // compatibility.
  const explicitMode = cfg.get<string>('mode', '').trim();
  const legacyAgentMode = cfg.get<boolean>('agentMode', true);
  let mode: Mode;
  if (explicitMode === 'chat' || explicitMode === 'agent' || explicitMode === 'swarm') {
    mode = explicitMode;
  } else {
    mode = legacyAgentMode ? 'agent' : 'chat';
  }

  return {
    serverUrlOverride: cfg.get<string>('serverUrl', '').trim(),
    model: cfg.get<string>('model', 'Qwen3.6-35B-A3B-UD-Q4_K_M'),
    mode,
    cliPathOverride: cfg.get<string>('cliPath', '').trim(),
  };
}

