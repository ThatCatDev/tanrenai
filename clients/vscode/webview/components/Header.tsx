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
      <span class="ok">●</span>
      <span class="meta">
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault();
            send({ type: 'pick_model' });
          }}
          title="Change model"
        >
          {model}
        </a>{' '}
        · {toolCount} tools
      </span>
      <ModePicker mode={mode} />
      <button
        class="icon-btn"
        title="Clear chat"
        onClick={() => send({ type: 'clear_chat' })}
      >
        ✕
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
