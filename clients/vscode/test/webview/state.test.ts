import { describe, expect, it } from 'vitest';
import { initialState, reduce, type AppState, type Entry } from '../../webview/state';

const seed = (overrides: Partial<AppState> = {}): AppState => ({ ...initialState, ...overrides });

const assistantEntry = (entries: Entry[], id: string) =>
  entries.find((e): e is Extract<Entry, { kind: 'assistant' }> => e.kind === 'assistant' && e.id === id);

describe('reduce', () => {
  it('updates connection on state', () => {
    const next = reduce(seed(), {
      type: 'state',
      state: { status: 'connected', model: 'X', toolCount: 3 },
    });
    expect(next.connection).toEqual({ status: 'connected', model: 'X', toolCount: 3 });
  });

  it('updates mode', () => {
    const next = reduce(seed(), { type: 'mode', mode: 'swarm' });
    expect(next.mode).toBe('swarm');
  });

  it('toggles turn flag on turn_start / turn_end', () => {
    const a = reduce(seed(), { type: 'turn_start' });
    expect(a.turnRunning).toBe(true);
    const b = reduce(a, { type: 'turn_end', ok: true });
    expect(b.turnRunning).toBe(false);
  });

  it('appends an error entry when a turn ends with reason', () => {
    const next = reduce(seed({ turnRunning: true }), {
      type: 'turn_end',
      ok: false,
      reason: 'broken',
    });
    expect(next.entries).toHaveLength(1);
    expect(next.entries[0].kind).toBe('error');
    expect((next.entries[0] as Extract<Entry, { kind: 'error' }>).text).toBe('broken');
  });

  it('opens a user entry on message_start with role=user', () => {
    const next = reduce(seed(), { type: 'message_start', role: 'user', id: 'u1' });
    expect(next.entries).toHaveLength(1);
    expect(next.entries[0]).toEqual({ kind: 'user', id: 'u1', content: '' });
  });

  it('appends user content via message_delta', () => {
    let s: AppState = seed();
    s = reduce(s, { type: 'message_start', role: 'user', id: 'u1' });
    s = reduce(s, { type: 'message_delta', id: 'u1', text: 'hi ' });
    s = reduce(s, { type: 'message_delta', id: 'u1', text: 'world' });
    expect(s.entries[0]).toMatchObject({ kind: 'user', content: 'hi world' });
  });

  it('streams assistant content + reasoning into separate channels', () => {
    let s: AppState = seed();
    s = reduce(s, { type: 'message_start', role: 'assistant', id: 'a1' });
    s = reduce(s, { type: 'message_delta', id: 'a1', text: 'hello', channel: 'content' });
    s = reduce(s, { type: 'message_delta', id: 'a1', text: 'thinking…', channel: 'reasoning' });
    const a = assistantEntry(s.entries, 'a1');
    expect(a?.content).toBe('hello');
    expect(a?.reasoning).toBe('thinking…');
    expect(a?.open).toBe(true);
  });

  it('closes assistant entries on message_end', () => {
    let s: AppState = seed();
    s = reduce(s, { type: 'message_start', role: 'assistant', id: 'a1' });
    s = reduce(s, { type: 'message_end', id: 'a1' });
    expect(assistantEntry(s.entries, 'a1')?.open).toBe(false);
  });

  it('synthesises an assistant entry when a delta arrives without start', () => {
    const next = reduce(seed(), {
      type: 'message_delta',
      id: 'a1',
      text: 'hi',
      channel: 'content',
    });
    expect(next.entries).toHaveLength(1);
    expect(assistantEntry(next.entries, 'a1')).toMatchObject({ content: 'hi', reasoning: '' });
  });

  it('appends a tool entry on tool_call and attaches result on tool_result', () => {
    let s: AppState = seed();
    s = reduce(s, {
      type: 'tool_call',
      id: 't1',
      name: 'file_read',
      arguments: '{}',
      intercepted: true,
    });
    s = reduce(s, { type: 'tool_result', id: 't1', ok: true, content: 'data' });
    expect(s.entries).toHaveLength(1);
    const tool = s.entries[0];
    expect(tool.kind).toBe('tool');
    if (tool.kind === 'tool') {
      expect(tool.name).toBe('file_read');
      expect(tool.intercepted).toBe(true);
      expect(tool.result).toEqual({ ok: true, content: 'data' });
    }
  });

  it('clears all entries on clear_chat', () => {
    let s: AppState = seed();
    s = reduce(s, { type: 'message_start', role: 'user', id: 'u1' });
    s = reduce(s, { type: 'tool_call', id: 't1', name: 'x', arguments: '{}', intercepted: false });
    expect(s.entries).toHaveLength(2);
    s = reduce(s, { type: 'clear_chat' });
    expect(s.entries).toHaveLength(0);
  });

  it('replays a full assistant turn idempotently when ids match', () => {
    // This exercises the host-replay path: same events delivered twice
    // (e.g. live, then on remount) should not duplicate the entry.
    let s: AppState = seed();
    const events = [
      { type: 'message_start', role: 'assistant', id: 'a1' } as const,
      { type: 'message_delta', id: 'a1', text: 'hi', channel: 'content' } as const,
      { type: 'message_end', id: 'a1' } as const,
    ];
    for (const e of events) s = reduce(s, e);
    const before = s.entries.length;
    for (const e of events) s = reduce(s, e);
    // message_start updates in place by id (appendOrUpdate). The follow-up
    // delta would re-append since the entry was reset to '' — capture the
    // observable behaviour: still exactly one entry, content reflects last
    // applied delta.
    expect(s.entries).toHaveLength(before);
    expect(assistantEntry(s.entries, 'a1')?.content).toBe('hi');
  });

  it('updates tokenRate on token_rate', () => {
    let s: AppState = seed({ turnRunning: true });
    s = reduce(s, { type: 'token_rate', tokens: 42, tps: 18.7 });
    expect(s.tokenRate).toEqual({ tokens: 42, tps: 18.7 });

    // Subsequent updates overwrite — the panel always shows the latest.
    s = reduce(s, { type: 'token_rate', tokens: 113, tps: 22.4 });
    expect(s.tokenRate).toEqual({ tokens: 113, tps: 22.4 });
  });

  it('clears tokenRate on turn_start so a new turn does not show stale numbers', () => {
    let s: AppState = seed();
    s = reduce(s, { type: 'token_rate', tokens: 42, tps: 18.7 });
    expect(s.tokenRate).not.toBeNull();
    s = reduce(s, { type: 'turn_start' });
    expect(s.tokenRate).toBeNull();
  });

  it('preserves tokenRate across turn_end so the final reading stays visible', () => {
    let s: AppState = seed({ turnRunning: true });
    s = reduce(s, { type: 'token_rate', tokens: 113, tps: 22.4 });
    s = reduce(s, { type: 'turn_end', ok: true });
    // After the turn closes the footer still shows the final t/s — it
    // only resets on the next turn_start.
    expect(s.tokenRate).toEqual({ tokens: 113, tps: 22.4 });
  });

  it('ignores deltas for unknown ids on entries that already exist for other ids', () => {
    let s: AppState = seed();
    s = reduce(s, { type: 'message_start', role: 'user', id: 'u1' });
    s = reduce(s, { type: 'message_delta', id: 'u1', text: 'a' });
    // Delta for unknown id should synthesise a new assistant.
    const before = s.entries.length;
    s = reduce(s, { type: 'message_delta', id: 'a1', text: 'b' });
    expect(s.entries).toHaveLength(before + 1);
  });
});
