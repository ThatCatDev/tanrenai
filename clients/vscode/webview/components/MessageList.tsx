import { useEffect, useRef } from 'preact/hooks';
import type { Activity, Entry } from '../state';

interface Props {
  entries: Entry[];
  activity: Activity;
}

export function MessageList({ entries, activity }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  // Auto-scroll to bottom on new entries / streaming.
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  });

  return (
    <div class="messages" ref={ref}>
      {entries.map((e) => (
        <EntryView key={e.id} entry={e} activity={activity} />
      ))}
    </div>
  );
}

function EntryView({ entry, activity }: { entry: Entry; activity: Activity }) {
  switch (entry.kind) {
    case 'user':
      return (
        <div class="msg user">
          <div class="role">user</div>
          <div class="body">{entry.content}</div>
        </div>
      );
    case 'assistant': {
      // Pulse the bubble whose channel matches the current activity.
      const reasoningStreaming = entry.open && activity.kind === 'thinking';
      const contentStreaming = entry.open && activity.kind === 'generating';

      return (
        <>
          {entry.reasoning && (
            <div class={`msg reasoning${reasoningStreaming ? ' streaming' : ''}`}>
              <div class="role">
                thinking
                {reasoningStreaming && <span class="pulse" />}
              </div>
              <div class="body">{entry.reasoning}</div>
            </div>
          )}
          {entry.content && (
            <div class={`msg assistant${contentStreaming ? ' streaming' : ''}`}>
              <div class="body">{entry.content}</div>
              {contentStreaming && <span class="caret" />}
            </div>
          )}
        </>
      );
    }
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
  const running = !entry.result;
  const argPreview = entry.args.length > 200 ? entry.args.slice(0, 200) + '…' : entry.args;

  return (
    <div class={`tool${failed ? ' error' : ''}${running ? ' running' : ''}`}>
      <div class="name">
        {running && <span class="tool-spinner" />}
        {entry.name}
        {entry.intercepted && <span class="tag"> (editor)</span>}
      </div>
      <div class="args">{argPreview}</div>
      <details>
        <summary>
          {entry.result ? (entry.result.ok ? 'Result' : 'Error') : 'Running…'}
        </summary>
        <div class="result">{entry.result?.content ?? 'running…'}</div>
      </details>
    </div>
  );
}
