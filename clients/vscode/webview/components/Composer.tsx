import { useState } from 'preact/hooks';
import type { Mode } from '../../src/protocol';
import { send } from '../host';

interface Props {
  turnRunning: boolean;
  mode: Mode;
}

export function Composer({ turnRunning, mode }: Props) {
  const [value, setValue] = useState('');

  const submit = () => {
    const text = value.trim();
    if (!text) return;
    if (text.startsWith('/') && handleSlashCommand(text)) {
      setValue('');

      return;
    }
    if (turnRunning) return;
    setValue('');
    send({ type: 'send', content: text });
  };

  return (
    <div class="input-row">
      <textarea
        rows={2}
        placeholder={placeholderForMode(mode)}
        value={value}
        onInput={(e) => setValue((e.currentTarget as HTMLTextAreaElement).value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
      />
      <div class="actions">
        <span class="hint">{turnRunning ? 'Streaming' : '⏎ to send · ⇧⏎ for newline'}</span>
        {turnRunning ? (
          <button class="secondary" onClick={() => send({ type: 'cancel' })}>
            Cancel
          </button>
        ) : (
          <button onClick={submit} disabled={!value.trim()}>
            Send
          </button>
        )}
      </div>
    </div>
  );
}

function placeholderForMode(mode: Mode): string {
  if (mode === 'swarm') return 'Brief the swarm…';
  if (mode === 'agent') return 'Ask Tanrenai…';

  return 'Chat with Tanrenai (no tools)…';
}

function handleSlashCommand(text: string): boolean {
  const parts = text.trim().split(/\s+/);
  const cmd = parts[0].toLowerCase();
  switch (cmd) {
    case '/clear':
      send({ type: 'clear_chat' });

      return true;
    case '/chat':
      send({ type: 'set_mode', mode: 'chat' });

      return true;
    case '/agent':
      send({ type: 'set_mode', mode: 'agent' });

      return true;
    case '/swarm':
      send({ type: 'set_mode', mode: parts[1] === 'off' ? 'agent' : 'swarm' });

      return true;
    case '/model':
      send({ type: 'pick_model' });

      return true;
    default:
      return false;
  }
}
