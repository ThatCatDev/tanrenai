import { useEffect, useRef, useState } from 'preact/hooks';
import type { Mode } from '../../src/protocol';
import { send } from '../host';

interface Props {
  signedIn: boolean;
  mode: Mode;
}

export function Footer({ signedIn, mode }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  // Click-outside to dismiss.
  useEffect(() => {
    if (!open) return undefined;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDoc);

    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  const fire = (cb: () => void) => {
    setOpen(false);
    cb();
  };

  return (
    <div class="footer">
      <span class="footer-status">
        <span class="footer-dot" aria-hidden="true" />
        <span class="footer-mode">{mode}</span>
      </span>
      <div class="footer-menu" ref={ref}>
        <button
          class="footer-trigger"
          aria-haspopup="menu"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          title="More"
        >
          ⋯
        </button>
        {open && (
          <div class="footer-menu-panel" role="menu">
            <button role="menuitem" onClick={() => fire(() => send({ type: 'pick_model' }))}>
              Choose model
            </button>
            <button role="menuitem" onClick={() => fire(() => send({ type: 'reconnect' }))}>
              Reconnect
            </button>
            <button role="menuitem" onClick={() => fire(() => send({ type: 'clear_chat' }))}>
              Clear chat
            </button>
            <div class="footer-menu-divider" />
            <button role="menuitem" onClick={() => fire(() => send({ type: 'show_gpu_status' }))}>
              GPU status
            </button>
            <button role="menuitem" onClick={() => fire(() => send({ type: 'stop_gpu' }))}>
              Stop GPU
            </button>
            <button
              role="menuitem"
              class="menu-danger"
              onClick={() => fire(() => send({ type: 'destroy_gpu' }))}
            >
              Destroy GPU…
            </button>
            <div class="footer-menu-divider" />
            {signedIn ? (
              <button role="menuitem" onClick={() => fire(() => send({ type: 'logout' }))}>
                Sign out
              </button>
            ) : (
              <button role="menuitem" onClick={() => fire(() => send({ type: 'login' }))}>
                Sign in
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
