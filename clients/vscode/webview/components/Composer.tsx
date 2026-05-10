import { useState } from 'preact/hooks';
import type { Mode, SelectionAttachment } from '../../src/protocol';
import { send } from '../host';
import type { Action } from '../state';

interface Props {
  turnRunning: boolean;
  mode: Mode;
  attachments: SelectionAttachment[];
  availableSelection: SelectionAttachment | null;
  dispatch: (a: Action) => void;
}

export function Composer({ turnRunning, mode, attachments, availableSelection, dispatch }: Props) {
  const [value, setValue] = useState('');

  // Suppress the live indicator when this exact selection is already
  // attached — avoids showing "Selection available" alongside an
  // identical chip.
  const liveSelection =
    availableSelection &&
    !attachments.some(
      (a) => a.path === availableSelection.path && a.text === availableSelection.text,
    )
      ? availableSelection
      : null;
  const liveLines =
    liveSelection
      ? Math.max(1, liveSelection.endLine - liveSelection.startLine + 1)
      : 0;

  const submit = () => {
    const text = value.trim();
    if (!text && attachments.length === 0) return;
    if (text.startsWith('/') && handleSlashCommand(text)) {
      setValue('');

      return;
    }
    if (turnRunning) return;
    setValue('');
    send({ type: 'send', content: text, attachments });
    dispatch({ type: 'attach_clear_pending' });
  };

  return (
    <div class="input-row">
      {liveSelection && (
        <button
          class="selection-hint"
          title={`Add ${liveSelection.label} to chat`}
          onClick={() => send({ type: 'attach_request' })}
        >
          <span class="selection-hint-dot" aria-hidden="true" />
          <span class="selection-hint-label">
            {liveLines} line{liveLines === 1 ? '' : 's'} selected · {liveSelection.label}
          </span>
          <span class="selection-hint-action">+ Add</span>
        </button>
      )}
      {attachments.length > 0 && (
        <div class="attachments">
          {attachments.map((a, i) => (
            <span key={`${a.path}-${a.startLine}-${i}`} class="chip" title={a.text}>
              <span class="chip-label">
                <span class="chip-glyph">⟦</span>
                {a.label}
                <span class="chip-glyph">⟧</span>
              </span>
              <button
                class="chip-x"
                title="Remove"
                onClick={() => dispatch({ type: 'attach_remove', index: i })}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
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
        <button
          class="secondary attach-btn"
          title="Attach editor selection"
          onClick={() => send({ type: 'attach_request' })}
        >
          + Attach
        </button>
        <span class="hint">{turnRunning ? 'Streaming' : '⏎ to send · ⇧⏎ for newline'}</span>
        {turnRunning ? (
          <button class="secondary" onClick={() => send({ type: 'cancel' })}>
            Cancel
          </button>
        ) : (
          <button onClick={submit} disabled={!value.trim() && attachments.length === 0}>
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
