import { useState } from 'preact/hooks';
import type { SwarmActivityMsg, SwarmStep } from '../state';

interface Props {
  swarm: SwarmActivityMsg;
  /** Parent-depth swarms in depth order (0..swarm.depth-1). Empty when
   *  the active swarm is the root. Used to render a breadcrumb so users
   *  retain context when the agent decomposes a step into a sub-swarm
   *  (the "11 became 2" situation — actually 2 sub-steps of step N of
   *  the outer 11). */
  ancestors: SwarmActivityMsg[];
}

/**
 * Always-visible status for the active swarm. Sits above the composer
 * so the user can track what the agent is doing without scrolling — the
 * full SwarmActivityCard remains in the chat as the historical record.
 *
 * Two display modes:
 *  - Collapsed (default): one row showing progress + current step. Tiny
 *    visual footprint, just enough to answer "what's it on right now?"
 *  - Expanded: full step list, click anywhere on the row to toggle.
 *
 * Drill-down: when there are ancestor crumbs, each is clickable. The
 * dock locally remembers a "focused depth" so the user can inspect any
 * level of the hierarchy without losing the live status — the
 * breadcrumb stays visible while the body shows the focused depth's
 * steps. Reverts to the active swarm's depth on next live event.
 *
 * Closes itself when there's nothing to show; the caller is expected
 * to render conditionally.
 */
export function SwarmDock({ swarm, ancestors }: Props) {
  const [open, setOpen] = useState(false);
  // null = follow the active swarm (default). When the user clicks an
  // ancestor crumb we pin to that depth until the live tracker
  // advances (a new worker_start at a deeper depth) — at that point
  // the user almost certainly wants the live thing back. We approximate
  // that by resetting focus whenever the active swarm's identity
  // changes; tracked via the swarm prop's depth as the comparison key.
  const [focusedDepth, setFocusedDepth] = useState<number | null>(null);
  // Pick which swarm's body to show. Defaults to the live active swarm
  // (when focusedDepth is null OR the focused depth no longer exists).
  const all = [...ancestors, swarm];
  const focused =
    focusedDepth === null
      ? swarm
      : all.find((s) => s.depth === focusedDepth) ?? swarm;

  const total = focused.steps.length;
  const done = focused.steps.filter((s) => s.status === 'done').length;
  const running = focused.steps.find((s) => s.status === 'running');
  // Current step priority: actively running > first pending > last done.
  // If everything's done the dock shows the final step so users can read
  // the result of the most recent work while the chat sits idle.
  const current =
    running ??
    focused.steps.find((s) => s.status === 'pending') ??
    focused.steps[focused.steps.length - 1];

  const phase: 'running' | 'done' | 'pending' | 'mixed' =
    running ? 'running'
    : done === total && total > 0 ? 'done'
    : done > 0 ? 'mixed'
    : 'pending';

  const isPinned = focusedDepth !== null && focusedDepth !== swarm.depth;

  return (
    <div class={`swarm-dock ${phase}${isPinned ? ' pinned' : ''}`} role="status" aria-live="polite">
      {ancestors.length > 0 && (
        <ol class="swarm-dock-breadcrumb" aria-label="Swarm hierarchy">
          {ancestors.map((parent) => {
            const pDone = parent.steps.filter((s) => s.status === 'done').length;
            const pTotal = parent.steps.length;
            // Surface the step the parent was running when it spawned
            // the child — best proxy for "this is the step we're in".
            // Falls back to the last-touched step if none is currently
            // running (the parent paused while the child works).
            const pCurrent = currentParentStep(parent);
            const isFocused = focused.depth === parent.depth;
            return (
              <li key={parent.depth} class={`swarm-dock-crumb${isFocused ? ' is-focused' : ''}`}>
                <button
                  class="swarm-dock-crumb-btn"
                  onClick={() => {
                    setFocusedDepth(parent.depth);
                    setOpen(true);
                  }}
                  title={`Focus depth ${parent.depth}`}
                >
                  <span class="swarm-dock-crumb-depth">d{parent.depth}</span>
                  <span class="swarm-dock-crumb-count">
                    {pDone}/{pTotal}
                  </span>
                  {pCurrent && (
                    <span class="swarm-dock-crumb-step" title={pCurrent.description}>
                      ▸ {pCurrent.description}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ol>
      )}
      <button
        class="swarm-dock-summary"
        onClick={() => {
          if (isPinned) {
            // Pinned to an ancestor — clicking the main row pops back
            // to the live active swarm (and stays expanded if it was).
            setFocusedDepth(null);
          } else {
            setOpen((v) => !v);
          }
        }}
        aria-expanded={open}
        title={
          isPinned
            ? 'Return to live status'
            : open
              ? 'Collapse swarm status'
              : 'Expand swarm status'
        }
      >
        <span class="swarm-dock-marker" aria-hidden="true">
          {phase === 'running' ? '◐' : phase === 'done' ? '✓' : '·'}
        </span>
        <span class="swarm-dock-count">
          {done}/{total}
        </span>
        {focused.depth > 0 && <span class="swarm-dock-depth">d{focused.depth}</span>}
        {current && (
          <span class="swarm-dock-current">
            <span class="swarm-dock-verb">
              {isPinned ? 'viewing' : phaseLabel(phase, current.status)}
            </span>
            <span class="swarm-dock-desc">{current.description || '—'}</span>
          </span>
        )}
        {focused.verifying && <span class="swarm-dock-verify">verifying…</span>}
        <span class="swarm-dock-toggle" aria-hidden="true">
          {isPinned ? '↺' : open ? '▴' : '▾'}
        </span>
      </button>
      {open && (
        <ol class="swarm-dock-steps">
          {focused.steps.map((s) => (
            <li
              key={s.index}
              class={`swarm-dock-step ${s.status}${s === current ? ' is-current' : ''}`}
            >
              <span class="swarm-dock-step-marker" aria-hidden="true">
                {stepGlyph(s.status)}
              </span>
              <span class="swarm-dock-step-desc">{s.description}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

/** Pick the verb that describes the current step's relationship to the
 *  user right now. "running" is happening NOW; "pending" means the
 *  agent will get to it; "done" means it just finished. */
function phaseLabel(
  phase: 'running' | 'done' | 'pending' | 'mixed',
  stepStatus: string,
): string {
  if (stepStatus === 'running') return 'running';
  if (stepStatus === 'error') return 'failed';
  if (phase === 'done') return 'done';
  if (stepStatus === 'done') return 'last';
  return 'next';
}

/** Pick the step on a parent swarm that's "where we are" — i.e. the
 *  one the parent was working on when it spawned this child. Running
 *  wins; otherwise the last done; otherwise the first step (parent
 *  may have just started). */
function currentParentStep(parent: SwarmActivityMsg): SwarmStep | undefined {
  const running = parent.steps.find((s) => s.status === 'running');
  if (running) return running;
  // Walk backwards to find the most recently completed step — that's
  // typically the one whose worker spawned the child.
  for (let i = parent.steps.length - 1; i >= 0; i--) {
    if (parent.steps[i].status === 'done') return parent.steps[i];
  }
  return parent.steps[0];
}

function stepGlyph(status: string): string {
  switch (status) {
    case 'done':
      return '✓';
    case 'error':
      return '✗';
    case 'running':
      return '◐';
    default:
      return '·';
  }
}
