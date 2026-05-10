import { useEffect, useRef } from 'preact/hooks';
import { send } from '../host';
import type { Activity, Entry } from '../state';

interface Props {
  entries: Entry[];
  activity: Activity;
}

export function MessageList({ entries, activity }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  // Auto-scroll on new entries / streaming.
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
          <div class="role">You</div>
          <div class="body">{entry.content}</div>
        </div>
      );
    case 'assistant': {
      const reasoningStreaming = entry.open && activity.kind === 'thinking';
      const contentStreaming = entry.open && activity.kind === 'generating';

      return (
        <>
          {entry.reasoning && (
            <div class={`msg reasoning${reasoningStreaming ? ' streaming' : ''}`}>
              <div class="role">
                Thinking
                {reasoningStreaming && <span class="pulse" />}
              </div>
              <div class="body">{entry.reasoning}</div>
            </div>
          )}
          {entry.content && (
            <div class={`msg assistant${contentStreaming ? ' streaming' : ''}`}>
              <div class="role">Tanrenai</div>
              <div class="body">
                {entry.content}
                {contentStreaming && <span class="caret" />}
              </div>
            </div>
          )}
        </>
      );
    }
    case 'tool':
      return <ToolCard entry={entry} />;
    case 'approval':
      return <ApprovalCard entry={entry} />;
    case 'error':
      return (
        <div class="msg reasoning" style="color: var(--vscode-charts-red);">
          <div class="role">Error</div>
          <div class="body">{entry.text}</div>
        </div>
      );
  }
}

function ApprovalCard({ entry }: { entry: Extract<Entry, { kind: 'approval' }> }) {
  const argPreview = entry.args.length > 240 ? entry.args.slice(0, 240) + '…' : entry.args;
  const decide = (action: 'allow' | 'deny' | 'always') => {
    send({ type: 'approval_decision', id: entry.id, action });
  };

  return (
    <div class={`approval${entry.resolved ? ' resolved' : ''}`}>
      <div class="approval-head">
        <span class="bracket">〔</span>
        <span>Approval</span>
        <span class="approval-title">
          run <code>{entry.name}</code>
        </span>
        <span class="bracket" style="margin-left:auto">〕</span>
      </div>
      <div class="approval-args">{argPreview}</div>
      {!entry.resolved ? (
        <div class="approval-actions">
          <button onClick={() => decide('allow')}>Allow once</button>
          <button onClick={() => decide('always')}>Always</button>
          <button onClick={() => decide('deny')}>Deny</button>
        </div>
      ) : (
        <div class="approval-resolved-label">Resolved</div>
      )}
    </div>
  );
}

function ToolCard({ entry }: { entry: Extract<Entry, { kind: 'tool' }> }) {
  const failed = !!(entry.result && !entry.result.ok);
  const running = !entry.result;
  const argPreview = entry.args.length > 240 ? entry.args.slice(0, 240) + '…' : entry.args;

  return (
    <div class={`tool${failed ? ' error' : ''}${running ? ' running' : ''}`}>
      <div class="name">
        {running && <span class="tool-spinner" />}
        <code>{entry.name}</code>
        {entry.intercepted && <span class="tag">editor</span>}
      </div>
      <div class="args">{argPreview}</div>
      <details>
        <summary>{entry.result ? (entry.result.ok ? 'Result' : 'Error') : 'Running…'}</summary>
        <div class="result">{entry.result?.content ?? 'running…'}</div>
      </details>
    </div>
  );
}
