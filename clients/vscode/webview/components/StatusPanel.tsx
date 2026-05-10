import { useEffect, useRef } from 'preact/hooks';
import type { ConnectionState } from '../../src/protocol';
import { send } from '../host';

export function StatusPanel({ connection }: { connection: ConnectionState }) {
  switch (connection.status) {
    case 'idle':
      return (
        <div class="status-panel">
          <div class="label">Initialising…</div>
        </div>
      );
    case 'connecting':
      return <Connecting progress={connection.progress} />;
    case 'no_credentials':
      return (
        <div class="status-panel">
          <div class="label">Not signed in.</div>
          <button onClick={() => send({ type: 'login' })}>Sign in to Tanrenai</button>
        </div>
      );
    case 'error':
      return (
        <div class="status-panel error">
          <div class="label">Error: {connection.message}</div>
          <button onClick={() => send({ type: 'reconnect' })}>Retry</button>
        </div>
      );
    case 'connected':
      return null;
  }
}

function Connecting({ progress }: { progress: { message: string; level: 'info' | 'warn' }[] }) {
  const logRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [progress.length]);

  return (
    <div class="status-panel">
      <div class="label">
        <span class="spinner" /> Connecting…
      </div>
      {progress.length > 0 && (
        <div class="progress-log" ref={logRef}>
          {progress.map((p, i) => (
            <div key={i} class={`line ${p.level}`}>
              {p.message}
            </div>
          ))}
        </div>
      )}
      <div class="status-actions">
        <button class="secondary" onClick={() => send({ type: 'cancel_connect' })}>
          Cancel
        </button>
        <button class="secondary" onClick={() => send({ type: 'pick_model' })}>
          Change Model…
        </button>
      </div>
    </div>
  );
}
