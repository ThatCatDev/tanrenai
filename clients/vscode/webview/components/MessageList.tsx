import { useCallback, useLayoutEffect, useRef, useState } from 'preact/hooks';
import { send } from '../host';
import type { Activity, Entry } from '../state';

interface Props {
  entries: Entry[];
  activity: Activity;
}

// Scroll is "pinned to bottom" when the user is within this many pixels of
// the bottom. Streaming content auto-scrolls only while pinned.
const PIN_THRESHOLD = 80;

/**
 * Build a fingerprint that changes whenever any rendered text grows or a
 * new entry arrives. Used as the dep for the auto-scroll effect — without
 * it, useLayoutEffect with no deps would still run too eagerly during
 * React's reconciliation, but with it we trigger exactly when content
 * has changed in a way that affects scroll height.
 */
function contentSignature(entries: Entry[]): string {
  let sig = `${entries.length}`;
  for (const e of entries) {
    if (e.kind === 'user') sig += `|u${e.content.length}`;
    else if (e.kind === 'assistant') sig += `|a${e.content.length}+${e.reasoning.length}`;
    else if (e.kind === 'tool') sig += `|t${e.result ? '!' : '?'}${e.args.length}`;
    else if (e.kind === 'approval') sig += `|p${e.resolved ? '!' : '?'}`;
    else if (e.kind === 'error') sig += `|e`;
  }

  return sig;
}

export function MessageList({ entries, activity }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  // Default to pinned. The user must scroll up explicitly to break it.
  const pinnedRef = useRef(true);
  const [unreadGrowth, setUnreadGrowth] = useState(false);
  // Bump-counter so scroll-driven pin toggles cause a re-render — refs
  // alone don't trigger Preact's reconciliation.
  const [, setForceTick] = useState(0);
  // Tracks whether we're currently scrolling programmatically — so the
  // resulting scroll event doesn't accidentally toggle pinned.
  const programmaticScrollRef = useRef(false);

  const scrollToBottom = useCallback((smooth = false) => {
    const el = ref.current;
    if (!el) return;
    programmaticScrollRef.current = true;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
    // Reset the flag on the next tick — by then the scroll event has fired.
    requestAnimationFrame(() => {
      programmaticScrollRef.current = false;
    });
  }, []);

  // Latest user id — submit forces re-pin and scroll.
  const lastUserId = (() => {
    for (let i = entries.length - 1; i >= 0; i--) {
      if (entries[i].kind === 'user') return entries[i].id;
    }

    return undefined;
  })();

  // useLayoutEffect runs after DOM commit, before paint — so scrollHeight
  // is correct and the user never sees the un-scrolled state flash.
  useLayoutEffect(() => {
    if (lastUserId) {
      pinnedRef.current = true;
      setUnreadGrowth(false);
      scrollToBottom();
    }
  }, [lastUserId, scrollToBottom]);

  const sig = contentSignature(entries);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (pinnedRef.current) {
      scrollToBottom();
      if (unreadGrowth) setUnreadGrowth(false);
    } else {
      // Content changed while unpinned — surface the badge.
      setUnreadGrowth(true);
    }
    // We deliberately depend on `sig` only — pinnedRef is a ref, not state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

  const onScroll = () => {
    if (programmaticScrollRef.current) return;
    const el = ref.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    const nowPinned = distance < PIN_THRESHOLD;
    if (nowPinned !== pinnedRef.current) {
      pinnedRef.current = nowPinned;
      // Force a re-render so the jump button shows/hides correctly.
      if (nowPinned && unreadGrowth) {
        setUnreadGrowth(false);
      } else if (!nowPinned) {
        // re-render via state poke — toggling unreadGrowth would be wrong here.
        setForceTick((t) => t + 1);
      }
    }
  };

  const pinned = pinnedRef.current;

  return (
    <div class="messages-wrap">
      <div class="messages" ref={ref} onScroll={onScroll}>
        {entries.map((e) => (
          <EntryView key={e.id} entry={e} activity={activity} />
        ))}
      </div>
      {!pinned && unreadGrowth && (
        <button
          class="jump-to-latest"
          onClick={() => {
            pinnedRef.current = true;
            setUnreadGrowth(false);
            setForceTick((t) => t + 1);
            scrollToBottom(true);
          }}
        >
          ↓ New messages
        </button>
      )}
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
