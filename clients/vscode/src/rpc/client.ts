import { spawn, ChildProcessWithoutNullStreams } from 'node:child_process';
import { EventEmitter } from 'node:events';
import * as readline from 'node:readline';
import {
  InboundMsg,
  InitMsg,
  OutboundMsg,
  PROTOCOL_VERSION,
  ReadyMsg,
} from './messages';

export interface RPCClientOptions {
  cliPath: string;
  args?: string[];
  env?: NodeJS.ProcessEnv;
  cwd?: string;
}

export interface RPCStartOptions {
  model: string;
  agentMode: boolean;
  swarmMode?: boolean;
  enableMemory?: boolean;
  enableScrolls?: boolean;
  interceptedTools?: string[];
  workspaceRoot?: string;
  maxIterations?: number;
  systemPrompt?: string;
}

/**
 * RPCClient owns the CLI subprocess. It exposes typed events for every
 * inbound message and a `send()` for outbound. Lifecycle: construct → start()
 * (returns a ReadyMsg) → use → dispose().
 */
export class RPCClient extends EventEmitter {
  private proc?: ChildProcessWithoutNullStreams;
  private startupResolver?: (msg: ReadyMsg) => void;
  private startupRejecter?: (err: Error) => void;

  constructor(private readonly options: RPCClientOptions) {
    super();
  }

  /**
   * Spawn the CLI, send `init`, await `ready`. Throws on protocol mismatch
   * or process exit before handshake completes.
   */
  start(opts: RPCStartOptions): Promise<ReadyMsg> {
    if (this.proc) {
      throw new Error('RPCClient.start() called twice');
    }

    return new Promise<ReadyMsg>((resolve, reject) => {
      this.startupResolver = resolve;
      this.startupRejecter = reject;

      const args = ['agent-rpc', ...(this.options.args ?? [])];
      this.proc = spawn(this.options.cliPath, args, {
        env: this.options.env ?? process.env,
        cwd: this.options.cwd,
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      const rl = readline.createInterface({ input: this.proc.stdout });
      rl.on('line', (line) => this.handleLine(line));

      // Stderr goes to the extension log channel (set up by the controller).
      this.proc.stderr.on('data', (chunk: Buffer) => {
        this.emit('stderr', chunk.toString('utf8'));
      });

      this.proc.on('exit', (code, signal) => {
        const reason = signal ? `signal ${signal}` : `exit code ${code}`;
        if (this.startupRejecter) {
          this.startupRejecter(new Error(`tanrenai agent-rpc exited (${reason}) before ready`));
          this.startupResolver = undefined;
          this.startupRejecter = undefined;
        }
        this.emit('exit', code, signal);
      });

      this.proc.on('error', (err) => {
        if (this.startupRejecter) {
          this.startupRejecter(err);
          this.startupResolver = undefined;
          this.startupRejecter = undefined;
        }
        this.emit('error', err);
      });

      // Send init.
      const init: InitMsg = {
        type: 'init',
        protocolVersion: PROTOCOL_VERSION,
        model: opts.model,
        agentMode: opts.agentMode,
        swarmMode: opts.swarmMode ?? false,
        enableMemory: opts.enableMemory ?? false,
        enableScrolls: opts.enableScrolls ?? false,
        interceptedTools: opts.interceptedTools ?? [],
        workspaceRoot: opts.workspaceRoot ?? '',
        maxIterations: opts.maxIterations ?? 0,
        systemPrompt: opts.systemPrompt ?? '',
      };
      this.write(init);
    });
  }

  /** Send any outbound message. */
  send(msg: OutboundMsg): void {
    this.write(msg);
  }

  /** Stop the subprocess gracefully (sends shutdown then closes stdin). */
  async dispose(): Promise<void> {
    if (!this.proc) {
      return;
    }
    try {
      this.write({ type: 'shutdown' });
    } catch {
      // proc already gone
    }
    this.proc.stdin.end();
    await new Promise<void>((resolve) => {
      const t = setTimeout(() => {
        this.proc?.kill('SIGTERM');
        resolve();
      }, 1000);
      this.proc?.once('exit', () => {
        clearTimeout(t);
        resolve();
      });
    });
    this.proc = undefined;
  }

  private write(msg: OutboundMsg): void {
    if (!this.proc || !this.proc.stdin.writable) {
      throw new Error('RPCClient: subprocess not running');
    }
    this.proc.stdin.write(JSON.stringify(msg) + '\n');
  }

  private handleLine(line: string): void {
    if (!line.trim()) {
      return;
    }
    let msg: InboundMsg;
    try {
      msg = JSON.parse(line) as InboundMsg;
    } catch (err) {
      this.emit('parse_error', { line, err });

      return;
    }

    if (msg.type === 'ready' && this.startupResolver) {
      if (msg.protocolVersion !== PROTOCOL_VERSION) {
        const err = new Error(
          `protocol version mismatch (extension=${PROTOCOL_VERSION}, cli=${msg.protocolVersion})`,
        );
        this.startupRejecter?.(err);
      } else {
        this.startupResolver(msg);
      }
      this.startupResolver = undefined;
      this.startupRejecter = undefined;
    }

    this.emit(msg.type, msg);
    this.emit('message', msg);
  }
}
