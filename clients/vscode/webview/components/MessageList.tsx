import { useEffect, useRef } from 'preact/hooks';
import type { Entry } from '../state';

interface Props {
  entries: Entry[];
}

export function MessageList({ entries }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  // Auto-scroll to bottom on new entries / streaming.
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  });

  return (
    <div class="messages" ref={ref}>
      {entries.map((e) => (
        <EntryView key={e.id} entry={e} />
      ))}
    </div>
  );
}

function EntryView({ entry }: { entry: Entry }) {
  switch (entry.kind) {
    case 'user':
      return (
        <div class="msg user">
          <div class="role">user</div>
          <div class="body">{entry.content}</div>
        </div>
      );
    case 'assistant':
      return (
        <>
          {entry.reasoning && (
            <div class="msg reasoning">
              <div class="role">thinking</div>
              <div class="body">{entry.reasoning}</div>
            </div>
          )}
          {entry.content && (
            <div class="msg assistant">
              <div class="body">{entry.content}</div>
            </div>
          )}
        </>
      );
    case 'tool':
      return <ToolCard entry={entry} />;
    case 'error':
      return (
        <div class="msg reasoning" style="color: var(--vscode-charts-red);">
          [error] {entry.text}
        </div>
      );
  }
}

function ToolCard({ entry }: { entry: Extract<Entry, { kind: 'tool' }> }) {
  const failed = entry.result && !entry.result.ok;
  const argPreview = entry.args.length > 200 ? entry.args.slice(0, 200) + '…' : entry.args;

  return (
    <div class={`tool${failed ? ' error' : ''}`}>
      <div class="name">
        {entry.name}
        {entry.intercepted && <span class="tag"> (editor)</span>}
      </div>
      <div class="args">{argPreview}</div>
      <details>
        <summary>{entry.result ? (entry.result.ok ? 'Result' : 'Error') : 'Running…'}</summary>
        <div class="result">{entry.result?.content ?? 'running…'}</div>
      </details>
    </div>
  );
}
