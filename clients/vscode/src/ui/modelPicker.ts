import * as vscode from 'vscode';

interface ModelChoice {
  label: string;
  description: string;
  detail?: string;
  value: string;
}

/**
 * Curated list of models likely to be on the user's backend. Not
 * exhaustive — there's a "Custom…" escape hatch at the bottom of the
 * picker for HF URIs or any other identifier the CLI accepts.
 */
const CURATED: ModelChoice[] = [
  {
    label: 'Qwen3.6-27B (dense coder)',
    description: 'UD-Q4_K_M · ~17 GB',
    detail: 'Best coder per benchmarks. Dense — no MoE shortcuts.',
    value: 'Qwen3.6-27B-UD-Q4_K_M',
  },
  {
    label: 'Qwen3.6-35B-A3B (MoE)',
    description: 'UD-Q4_K_M · ~21 GB',
    detail: '3B active params per token. Efficient on smaller GPUs.',
    value: 'Qwen3.6-35B-A3B-UD-Q4_K_M',
  },
  {
    label: 'Qwen3.5-9B',
    description: 'Q4_K_M · ~6 GB',
    detail: 'Lighter — fastest cold-start.',
    value: 'Qwen3.5-9B-Q4_K_M',
  },
  {
    label: 'Qwen3.5-27B',
    description: 'Q4_K_M · ~17 GB',
    detail: 'Older but reliable. Likely already cached on the backend.',
    value: 'Qwen3.5-27B-Q4_K_M',
  },
  {
    label: 'Qwen3.5-35B-A3B (MoE)',
    description: 'Q4_K_M · ~21 GB',
    value: 'Qwen3.5-35B-A3B-Q4_K_M',
  },
];

/**
 * Show a quickPick with curated model options plus a "Custom…" escape
 * hatch. Returns the chosen identifier, or undefined if cancelled.
 */
export async function showModelPicker(currentModel: string): Promise<string | undefined> {
  const picks: (vscode.QuickPickItem & { value?: string; isCustom?: boolean })[] = [
    ...CURATED.map((m) => ({
      label: m.label + (m.value === currentModel ? '  $(check)' : ''),
      description: m.description,
      detail: m.detail,
      value: m.value,
    })),
    { label: '$(edit) Custom…', description: 'Enter a model name or hf:// URI', isCustom: true },
  ];

  const choice = await vscode.window.showQuickPick(picks, {
    title: 'Tanrenai — pick a model',
    placeHolder: `Currently: ${currentModel || '(none)'}`,
    matchOnDescription: true,
    matchOnDetail: true,
  });
  if (!choice) {
    return undefined;
  }
  if (choice.isCustom) {
    const custom = await vscode.window.showInputBox({
      title: 'Custom model identifier',
      prompt: 'Bare name (e.g. Qwen3.5-9B-Q4_K_M) or hf:// URI',
      value: currentModel,
      ignoreFocusOut: true,
    });

    return custom?.trim() || undefined;
  }

  return choice.value;
}
