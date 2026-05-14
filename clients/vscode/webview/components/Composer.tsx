import { useRef, useState } from 'preact/hooks';
import type {
  ImageAttachment,
  Mode,
  SelectionAttachment,
} from '../../src/protocol';
import { send } from '../host';
import type { Action } from '../state';

interface Props {
  turnRunning: boolean;
  mode: Mode;
  attachments: SelectionAttachment[];
  images: ImageAttachment[];
  availableSelection: SelectionAttachment | null;
  dispatch: (a: Action) => void;
}

/** Cap individual image size — guards against pasting 20MB screenshots that
 *  blow past the model's input limits or the IPC channel. */
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;

export function Composer({
  turnRunning,
  mode,
  attachments,
  images,
  availableSelection,
  dispatch,
}: Props) {
  const [value, setValue] = useState('');
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);

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
    if (!text && attachments.length === 0 && images.length === 0) return;
    if (text.startsWith('/') && handleSlashCommand(text)) {
      setValue('');

      return;
    }
    if (turnRunning) return;
    setValue('');
    send({
      type: 'send',
      content: text,
      attachments: attachments.length > 0 ? attachments : undefined,
      images: images.length > 0 ? images : undefined,
    });
    dispatch({ type: 'attach_clear_pending' });
    dispatch({ type: 'image_clear_pending' });
  };

  /** Read a File into an ImageAttachment (data URL) and dispatch it. */
  const ingestFile = (file: File) => {
    if (!file.type.startsWith('image/')) return;
    if (file.size > MAX_IMAGE_BYTES) {
      // Surface to the user — toast won't work in a webview, so render an
      // inline notice the next time we render. Simplest: insert a chip with
      // an error label and let them remove it.
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result ?? '');
      if (!dataUrl.startsWith('data:')) return;
      dispatch({
        type: 'image_attach',
        image: {
          label: file.name || 'image',
          mimeType: file.type,
          dataUrl,
          size: file.size,
        },
      });
    };
    reader.readAsDataURL(file);
  };

  const onPaste = (e: ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    let consumedAny = false;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) {
          ingestFile(file);
          consumedAny = true;
        }
      }
    }
    if (consumedAny) {
      e.preventDefault();
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const files = e.dataTransfer?.files;
    if (!files) return;
    for (let i = 0; i < files.length; i++) {
      ingestFile(files[i]);
    }
  };

  const onFilesPicked = (e: Event) => {
    const target = e.target as HTMLInputElement;
    if (!target.files) return;
    for (let i = 0; i < target.files.length; i++) {
      ingestFile(target.files[i]);
    }
    target.value = ''; // allow re-picking the same file
  };

  return (
    <div
      class={`input-row${dragOver ? ' drag-over' : ''}`}
      onDragEnter={(e) => {
        if (e.dataTransfer?.types.includes('Files')) {
          e.preventDefault();
          setDragOver(true);
        }
      }}
      onDragOver={(e) => {
        if (e.dataTransfer?.types.includes('Files')) {
          e.preventDefault();
        }
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragOver(false);
      }}
      onDrop={onDrop}
    >
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
      {(attachments.length > 0 || images.length > 0) && (
        <div class="attachments">
          {attachments.map((a, i) => (
            <span key={`sel-${a.path}-${a.startLine}-${i}`} class="chip" title={a.text}>
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
          {images.map((img, i) => (
            <span key={`img-${i}`} class="chip image-chip" title={img.label}>
              <img class="chip-thumb" src={img.dataUrl} alt="" />
              <span class="chip-label">{img.label}</span>
              <button
                class="chip-x"
                title="Remove"
                onClick={() => dispatch({ type: 'image_remove', index: i })}
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
        onPaste={onPaste}
      />
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={onFilesPicked}
      />
      <div class="actions">
        <button
          class="secondary attach-btn"
          title="Attach editor selection"
          onClick={() => send({ type: 'attach_request' })}
        >
          + Sel
        </button>
        <button
          class="secondary attach-btn"
          title="Attach image (paste or drag also works)"
          onClick={() => fileInputRef.current?.click()}
        >
          + Img
        </button>
        <span class="hint">
          {turnRunning ? 'Streaming' : '⏎ to send · paste / drag image'}
        </span>
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
