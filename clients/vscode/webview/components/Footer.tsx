import { useEffect, useRef, useState } from 'preact/hooks';
import type { Mode } from '../../src/protocol';
import type { ContextUsage, TokenRate } from '../state';
import { send } from '../host';

interface Props {
  signedIn: boolean;
  mode: Mode;
  /** Generation throughput for the current/most-recent turn. Null until
   *  the first delta arrives. The footer renders it next to the mode so
   *  there's a quiet always-visible meter without taking dedicated space. */
  tokenRate: TokenRate | null;
  /** Prompt-budget snapshot for the running session. Null until the CLI's
   *  first `context_usage` lands (right after `ready`). When present the
   *  footer shows `38% / 8k` and exposes a breakdown popover. */
  contextUsage?: ContextUsage | null;
}

export function Footer({ signedIn, mode, tokenRate, contextUsage }: Props) {
  const [open, setOpen] = useState(false);
  const [usageOpen, setUsageOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  const usageRef = useRef<HTMLDivElement | null>(null);

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

  useEffect(() => {
    if (!usageOpen) return undefined;
    const onDoc = (e: MouseEvent) => {
      if (usageRef.current && !usageRef.current.contains(e.target as Node)) {
        setUsageOpen(false);
      }
    };
    document.addEventListener('mousedown', onDoc);

    return () => document.removeEventListener('mousedown', onDoc);
  }, [usageOpen]);

  const fire = (cb: () => void) => {
    setOpen(false);
    cb();
  };

  const usage = contextUsage ?? null;
  const pct = usage && usage.total > 0
    ? Math.min(100, Math.round(((usage.total - usage.available) / usage.total) * 100))
    : null;

  return (
    <div class="footer">
      <span class="footer-status">
        <span class="footer-dot" aria-hidden="true" />
        <span class="footer-mode">{mode}</span>
        {tokenRate && (
          <span class="footer-rate" title={`${tokenRate.tokens} tokens generated`}>
            {tokenRate.tps.toFixed(0)} t/s
          </span>
        )}
        {usage && pct !== null && (
          <div class="footer-usage" ref={usageRef}>
            <button
              class="footer-usage-trigger"
              aria-haspopup="dialog"
              aria-expanded={usageOpen}
              title={`${usage.total - usage.available} / ${usage.total} tokens used — click for breakdown`}
              onClick={() => setUsageOpen((v) => !v)}
            >
              {pct}% / {formatTokens(usage.total)}
            </button>
            {usageOpen && <UsagePanel usage={usage} pct={pct} />}
          </div>
        )}
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
            <button role="menuitem" onClick={() => fire(() => send({ type: 'compact_now' }))}>
              Compact now
            </button>
            <button
              role="menuitem"
              onClick={() => fire(() => send({ type: 'context_files_open' }))}
            >
              Context files…
            </button>
            <button role="menuitem" onClick={() => fire(() => send({ type: 'memories_open' }))}>
              Memories…
            </button>
            <button role="menuitem" onClick={() => fire(() => send({ type: 'scrolls_open' }))}>
              Scrolls…
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

function UsagePanel({ usage, pct }: { usage: ContextUsage; pct: number }) {
  const used = usage.total - usage.available;
  const rows: [string, number][] = [
    ['System / pinned', usage.system],
    ['Context files', usage.scrolls],
    ['Memory', usage.memory],
    ['Summary', usage.summary],
    ['History', usage.history],
  ];

  return (
    <div class="footer-usage-panel" role="dialog" aria-label="Context usage">
      <div class="footer-usage-head">
        <span class="footer-usage-pct">{pct}% used</span>
        <span class="footer-usage-tokens">
          {used.toLocaleString()} / {usage.total.toLocaleString()} tokens
        </span>
      </div>
      <div class="footer-usage-bar" aria-hidden="true">
        <div class="footer-usage-bar-fill" style={`width: ${pct}%`} />
      </div>
      <table class="footer-usage-table">
        <tbody>
          {rows.map(([label, v]) => (
            <tr key={label}>
              <td>{label}</td>
              <td>{v.toLocaleString()}</td>
            </tr>
          ))}
          <tr class="footer-usage-available">
            <td>Available</td>
            <td>{usage.available.toLocaleString()}</td>
          </tr>
        </tbody>
      </table>
      <div class="footer-usage-foot">
        {usage.historyCount} of {usage.totalHistory} history msgs in window
      </div>
    </div>
  );
}

/** Compact token totals: 1234 → "1.2k", 18432 → "18k". Keeps the footer
 *  one line on narrow sidebars; the popover shows the exact value. */
function formatTokens(n: number): string {
  if (n < 1000) return String(n);
  if (n < 10_000) return `${(n / 1000).toFixed(1)}k`;

  return `${Math.round(n / 1000)}k`;
}
