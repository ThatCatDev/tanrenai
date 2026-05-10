import type { Mode } from '../../src/protocol';
import { send } from '../host';

interface Props {
  model: string;
  toolCount: number;
  mode: Mode;
}

export function Header({ model, toolCount, mode }: Props) {
  return (
    <div class="header">
      <span class="mark">Tanrenai</span>
      <span class="meta" title={model}>
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault();
            send({ type: 'pick_model' });
          }}
        >
          {shortenModel(model)}
        </a>
        <span class="sep">·</span>
        {toolCount} tools
      </span>
      <ModePicker mode={mode} />
      <button class="icon-btn" title="Clear chat" onClick={() => send({ type: 'clear_chat' })}>
        Clear
      </button>
    </div>
  );
}

function ModePicker({ mode }: { mode: Mode }) {
  const items: { key: Mode; label: string; tooltip: string }[] = [
    { key: 'chat', label: 'Chat', tooltip: 'Chat — no tools' },
    { key: 'agent', label: 'Agent', tooltip: 'Agent — single agent with tools' },
    { key: 'swarm', label: 'Swarm', tooltip: 'Swarm — multi-agent orchestrator' },
  ];

  return (
    <div class="modes" role="tablist">
      {items.map((item) => (
        <button
          key={item.key}
          class={mode === item.key ? 'active' : ''}
          title={item.tooltip}
          onClick={() => send({ type: 'set_mode', mode: item.key })}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

// In a 250-px sidebar, "Qwen3.6-35B-A3B-UD-Q4_K_M" is ten times as long as
// the visible area. Trim to a recognisable head — the tooltip carries the
// full string for anyone who needs it.
function shortenModel(model: string): string {
  if (model.length <= 24) {
    return model;
  }

  return model.slice(0, 22) + '…';
}
