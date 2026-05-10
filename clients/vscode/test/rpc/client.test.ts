import { describe, expect, it } from 'vitest';
import { RPCClient } from '../../src/rpc/client';

/**
 * Fake CLI: runs node with a one-shot script that:
 *   1. reads NDJSON from stdin
 *   2. waits for `init`
 *   3. echoes a `ready` (or whatever the test wants)
 *   4. ignores subsequent input until shutdown
 *
 * `script` is the body of the IIFE that runs on the spawned node process —
 * each test customises the response shape.
 */
function fakeCli(script: string): string {
  return `(async () => {
    let buf = '';
    const out = (msg) => process.stdout.write(JSON.stringify(msg) + '\\n');
    const onLine = (line) => { ${script} };
    process.stdin.on('data', (chunk) => {
      buf += chunk.toString();
      let i;
      while ((i = buf.indexOf('\\n')) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 1);
        if (line) try { onLine(JSON.parse(line)); } catch {}
      }
    });
    process.stdin.on('end', () => process.exit(0));
  })();`;
}

const startWith = async (script: string) => {
  const client = new RPCClient({
    cliPath: process.execPath, // node
    args: ['-e', fakeCli(script)],
    subcommand: '',
  });

  return await client.start({
    model: 'X',
    agentMode: true,
    interceptedTools: [],
    workspaceRoot: '',
  });
};

describe('RPCClient', () => {
  it('completes the init → ready handshake', async () => {
    const ready = await startWith(`
      if (line.type === 'init') {
        out({ type: 'ready', protocolVersion: 1, model: line.model, tools: [
          { name: 'file_read', description: 'reads', schema: {} },
        ]});
      }
    `);
    expect(ready.type).toBe('ready');
    expect(ready.model).toBe('X');
    expect(ready.tools).toHaveLength(1);
    expect(ready.tools[0].name).toBe('file_read');
  });

  it('rejects start() when ready arrives with mismatched protocol version', async () => {
    await expect(
      startWith(`
        if (line.type === 'init') {
          out({ type: 'ready', protocolVersion: 99, model: 'X', tools: [] });
        }
      `),
    ).rejects.toThrow(/protocol version mismatch/);
  });

  it('rejects start() if the subprocess exits before ready', async () => {
    await expect(
      startWith(`
        if (line.type === 'init') {
          process.exit(1);
        }
      `),
    ).rejects.toThrow(/exited.*before ready/);
  });

  it('emits inbound events to listeners', async () => {
    const client = new RPCClient({
      cliPath: process.execPath,
      subcommand: '',
      args: [
        '-e',
        fakeCli(`
          if (line.type === 'init') {
            out({ type: 'ready', protocolVersion: 1, model: 'X', tools: [] });
            out({ type: 'connecting_progress', level: 'info', message: 'Loading…' });
            out({ type: 'turn_done', ok: true });
          }
        `),
      ],
    });

    const progress: unknown[] = [];
    const turnDone: unknown[] = [];
    client.on('connecting_progress', (m) => progress.push(m));
    client.on('turn_done', (m) => turnDone.push(m));

    await client.start({
      model: 'X',
      agentMode: true,
      interceptedTools: [],
      workspaceRoot: '',
    });

    // Give the fake process a beat to emit the post-ready events.
    await new Promise((r) => setTimeout(r, 50));

    expect(progress).toHaveLength(1);
    expect((progress[0] as { message: string }).message).toBe('Loading…');
    expect(turnDone).toHaveLength(1);

    await client.dispose();
  });

  it('dispose() cleanly stops the subprocess', async () => {
    const client = new RPCClient({
      cliPath: process.execPath,
      subcommand: '',
      args: [
        '-e',
        fakeCli(`
          if (line.type === 'init') out({ type: 'ready', protocolVersion: 1, model: 'X', tools: [] });
        `),
      ],
    });

    await client.start({
      model: 'X',
      agentMode: true,
      interceptedTools: [],
      workspaceRoot: '',
    });

    const exited = new Promise((resolve) => client.on('exit', resolve));
    await client.dispose();
    await exited;
  });
});
