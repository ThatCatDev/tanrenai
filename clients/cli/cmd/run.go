package cmd

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/rivo/tview"

	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
	gpuserve "github.com/ThatCatDev/tanrenai-gpu/pkg/serve"
	"github.com/ThatCatDev/tanrenai/shared/agent"
	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/scrolls"
	"github.com/ThatCatDev/tanrenai/shared/tools"
	"github.com/spf13/cobra"
)

// startupLog posts status lines to the TUI (or stdout if tui is nil).
type startupLog struct {
	tui *tuiApp
	// emit, when set, replaces the default stdout/stderr behaviour. It's
	// mutually exclusive with tui — used by agent-rpc to route progress
	// through NDJSON instead of corrupting stdout.
	emit func(level, msg string)
}

func (s *startupLog) Info(msg string) {
	switch {
	case s.tui != nil:
		s.tui.appendLogLine("[gray::-]  " + tview.Escape(msg) + "[-:-:-]")
	case s.emit != nil:
		s.emit("info", msg)
	default:
		_, _ = fmt.Fprintf(os.Stdout, "%s\n", msg)
	}
}

func (s *startupLog) Warn(msg string) {
	switch {
	case s.tui != nil:
		s.tui.appendLogLine("[yellow::-]  " + tview.Escape(msg) + "[-:-:-]")
	case s.emit != nil:
		s.emit("warn", msg)
	default:
		fmt.Fprintf(os.Stderr, "Warning: %s\n", msg)
	}
}

const (
	defaultEmbeddingModel    = "nomic-embed-text-v1.5.Q4_K_M"
	defaultEmbeddingModelURL = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf"
)

// tanrenaiDataDir returns the tanrenai data directory, matching the convention
// used by the GPU and server modules: ~/.local/share/tanrenai (or TANRENAI_DATA_DIR).
func tanrenaiDataDir() string {
	if dir := os.Getenv("TANRENAI_DATA_DIR"); dir != "" {
		return dir
	}
	if runtime.GOOS == "windows" {
		return filepath.Join(os.Getenv("LOCALAPPDATA"), "tanrenai")
	}
	home, _ := os.UserHomeDir()

	return filepath.Join(home, ".local", "share", "tanrenai")
}

// projectDirName is the conventional folder name created in cwd for
// project-local tanrenai state (architect spec, scrolls, permissions).
const projectDirName = ".tanrenai"

// ensureProjectDir creates `.tanrenai/` in the current working directory if
// it doesn't already exist. Failures are logged but not fatal — a read-only
// workspace should still be able to chat.
func ensureProjectDir(log *startupLog) {
	if err := os.MkdirAll(projectDirName, 0o755); err != nil {
		log.Warn(fmt.Sprintf("could not create %s/ in current directory: %v", projectDirName, err))
	}
}

// projectMemoryDir returns a project-scoped memory directory based on a hash of the working directory.
func projectMemoryDir() string {
	wd, err := os.Getwd()
	if err != nil {
		wd = "default"
	}
	h := sha256.Sum256([]byte(wd))
	hash16 := fmt.Sprintf("%x", h[:8])

	return filepath.Join(tanrenaiDataDir(), "memory", hash16)
}

// ensureEmbeddingModel checks if the embedding model exists and downloads it if not.
func ensureEmbeddingModel(log *startupLog) error {
	if _, err := gpuserve.ResolveModel(defaultEmbeddingModel); err == nil {
		return nil // already exists
	}

	log.Info("Downloading embedding model " + defaultEmbeddingModel + "...")
	destDir := gpuserve.ModelsDir()
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("create models dir: %w", err)
	}

	var lastPct int
	_, err := gpuserve.DownloadModel(defaultEmbeddingModelURL, destDir, func(downloaded, total int64) {
		if total <= 0 {
			return
		}
		pct := int(downloaded * 100 / total)
		if pct >= lastPct+5 {
			lastPct = pct
			log.Info(fmt.Sprintf("Downloading embedding model... %d%% (%d/%d MB)",
				pct, downloaded/(1024*1024), total/(1024*1024)))
		}
	})
	if err != nil {
		return fmt.Errorf("download embedding model: %w", err)
	}
	log.Info("Embedding model downloaded successfully")

	return nil
}

// defaultAgentSystemPrompt is the agent's system prompt, kept in
// agent_system_prompt.md (next to this file) and embedded at build time so it's
// easy to find and edit without touching Go. TrimSpace drops the file's
// trailing newline so the value is byte-identical to an inline string.
//
//go:embed agent_system_prompt.md
var defaultAgentSystemPromptRaw string

var defaultAgentSystemPrompt = strings.TrimSpace(defaultAgentSystemPromptRaw)

// parseRunParams reads the common flags from a cobra command into a runParams struct.
func parseRunParams(cmd *cobra.Command, model string) (runParams, error) {
	systemPrompt, _ := cmd.Flags().GetString("system")
	systemFile, _ := cmd.Flags().GetString("system-file")
	if systemFile != "" {
		data, err := os.ReadFile(systemFile)
		if err != nil {
			return runParams{}, fmt.Errorf("failed to read system file: %w", err)
		}
		systemPrompt = string(data)
	}

	agentMode, _ := cmd.Flags().GetBool("agent")
	ctxSize, _ := cmd.Flags().GetInt("ctx-size")
	responseBudget, _ := cmd.Flags().GetInt("response-budget")
	contextFiles, _ := cmd.Flags().GetStringSlice("context-file")
	memoryEnabled, _ := cmd.Flags().GetBool("memory")
	maxIterations, _ := cmd.Flags().GetInt("max-iterations")
	maxTokens, _ := cmd.Flags().GetInt("max-tokens")
	noScrolls, _ := cmd.Flags().GetBool("no-scrolls")
	thinking, _ := cmd.Flags().GetBool("thinking")
	local, _ := cmd.Flags().GetBool("local")
	gpuLayers, _ := cmd.Flags().GetInt("gpu-layers")
	flashAttn, _ := cmd.Flags().GetBool("flash-attn")
	swarmMode, _ := cmd.Flags().GetBool("swarm")
	cpuMoE, _ := cmd.Flags().GetBool("cpu-moe")
	noKVOffload, _ := cmd.Flags().GetBool("no-kv-offload")
	fitVRAM, _ := cmd.Flags().GetBool("fit")
	pipeFormat, _ := cmd.Flags().GetString("format")

	// Swarm mode implies agent mode.
	if swarmMode {
		agentMode = true
	}

	// Validate format value early so a typo errors out before we spin up
	// llama-server and a backend connection.
	switch pipeFormat {
	case "", "text", "json":
		// ok
	default:
		return runParams{}, fmt.Errorf("--format: must be 'text' or 'json', got %q", pipeFormat)
	}

	return runParams{
		model:          model,
		systemPrompt:   systemPrompt,
		agentMode:      agentMode,
		ctxSize:        ctxSize,
		ctxSizeChanged: cmd.Flags().Changed("ctx-size"),
		responseBudget: responseBudget,
		contextFiles:   contextFiles,
		memoryEnabled:  memoryEnabled,
		maxIterations:  maxIterations,
		maxTokens:      maxTokens,
		noScrolls:      noScrolls,
		thinking:       thinking,
		swarmMode:      swarmMode,
		local:          local,
		gpuLayers:      gpuLayers,
		flashAttn:      flashAttn,
		cpuMoE:         cpuMoE,
		noKVOffload:    noKVOffload,
		fitVRAM:        fitVRAM,
		pipeFormat:     pipeFormat,
	}, nil
}

var runCmd = &cobra.Command{
	Use:   "run [model]",
	Short: "Start an interactive chat",
	Long: "Start an interactive chat.\n\n" +
		"Hosted (default when logged in): the model is the service's — just run `tanrenai run`.\n" +
		"Local (--local): pass a model to run on your own machine, e.g. `tanrenai run <model> --local`.",
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		model := ""
		if len(args) > 0 {
			model = args[0]
		}
		p, err := parseRunParams(cmd, model)
		if err != nil {
			return err
		}

		// Hosted sessions use the service's configured model; local sessions
		// need one. Treat as remote when logged in and not forced local.
		_, credErr := loadCredentials()
		remote := !p.local && credErr == nil
		if remote && p.model != "" {
			fmt.Fprintf(os.Stderr, "Note: the hosted service runs its own model — ignoring %q. Just run `tanrenai run`.\n", p.model)
			p.model = ""
		}
		if !remote && p.model == "" {
			return fmt.Errorf("a model is required to run locally:\n  tanrenai run <model> --local")
		}

		display := p.model
		if display == "" {
			display = "hosted model"
		}

		pipeMode, _ := cmd.Flags().GetBool("pipe")
		if pipeMode {
			return startPipe(cmd.Context(), p)
		}
		return startTUI(display, func(t *tuiApp, log *startupLog) error {
			deps, err := setupSession(cmd.Context(), p, log)
			if err != nil {
				return err
			}
			assignToTUI(t, deps)
			return nil
		})
	},
}

var chatCmd = &cobra.Command{
	Use:   "chat",
	Short: "Interactive chat with a loaded model",
	RunE: func(cmd *cobra.Command, args []string) error {
		model, _ := cmd.Flags().GetString("model")
		if model == "" {
			return fmt.Errorf("specify a model with --model")
		}
		p, err := parseRunParams(cmd, model)
		if err != nil {
			return err
		}
		pipeMode, _ := cmd.Flags().GetBool("pipe")
		if pipeMode {
			return startPipe(cmd.Context(), p)
		}
		return startTUI(p.model, func(t *tuiApp, log *startupLog) error {
			deps, err := setupSession(cmd.Context(), p, log)
			if err != nil {
				return err
			}
			assignToTUI(t, deps)
			return nil
		})
	},
}

func startTUI(model string, startup func(t *tuiApp, log *startupLog) error) error {
	t := newTuiApp(model)
	t.loading = true
	t.startLoadingAnimation()

	// Set up file + TUI logging.
	tuiHandler := &tuiSlogHandler{tui: t}
	logFile, logErr := openLogFile()
	if logErr != nil {
		slog.SetDefault(slog.New(tuiHandler))
	} else {
		fileHandler := slog.NewTextHandler(logFile, &slog.HandlerOptions{Level: slog.LevelDebug})
		slog.SetDefault(slog.New(&multiHandler{handlers: []slog.Handler{tuiHandler, fileHandler}}))
	}

	go func() {
		log := &startupLog{tui: t}
		err := startup(t, log)

		// After startup, stop showing logs in the TUI but keep writing to the file.
		if logFile != nil {
			fileHandler := slog.NewTextHandler(logFile, &slog.HandlerOptions{Level: slog.LevelDebug})
			slog.SetDefault(slog.New(fileHandler))
		} else {
			slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
		}

		if err != nil {
			t.app.QueueUpdateDraw(func() {
				t.stopLoadingAnimation()
				t.addLine(fmt.Sprintf("[red::-]  Error: %v[-:-:-]", err))
				t.addLine("[gray::-]  Press Ctrl+C to exit.[-:-:-]")
				t.refreshChatView()
			})

			return
		}

		t.app.QueueUpdateDraw(func() {
			t.stopLoadingAnimation()
			t.loading = false
			t.inputArea.SetPlaceholder("")
			t.addLine("")
			t.refreshChatView()
		})
	}()

	err := t.run()
	if logFile != nil {
		_ = logFile.Close()
	}

	return err
}

// tuiSlogHandler routes slog records to the TUI chat view.
type tuiSlogHandler struct {
	tui   *tuiApp
	attrs []slog.Attr
}

func (h *tuiSlogHandler) Enabled(_ context.Context, _ slog.Level) bool { return true }

func (h *tuiSlogHandler) Handle(_ context.Context, r slog.Record) error {
	// Collect attrs into a map for easy lookup.
	attrs := make(map[string]string)
	for _, a := range h.attrs {
		attrs[a.Key] = a.Value.String()
	}
	r.Attrs(func(a slog.Attr) bool {
		attrs[a.Key] = a.Value.String()

		return true
	})

	// Filter and rewrite messages to be user-friendly.
	var msg string
	color := "gray"
	switch {
	case r.Level >= slog.LevelError:
		color = "red"
		msg = r.Message
	case r.Level >= slog.LevelWarn:
		color = "yellow"
		msg = r.Message
	case strings.HasPrefix(r.Message, "subprocess still loading"):
		if s := attrs["elapsed_s"]; s != "" {
			msg = fmt.Sprintf("Loading model... (%ss)", s)
		}
	case r.Message == "subprocess starting":
		msg = "Starting inference server..."
	case strings.HasPrefix(r.Message, "Using GPU acceleration"):
		color = "green"
		msg = r.Message
	case strings.HasPrefix(r.Message, "No GPU detected"):
		msg = r.Message
	default:
		return nil // skip noisy INFO messages
	}

	if msg == "" {
		return nil
	}
	line := fmt.Sprintf("[%s::-]  %s[-:-:-]", color, tview.Escape(msg))
	h.tui.appendLogLine(line)

	return nil
}

func (h *tuiSlogHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	combined := make([]slog.Attr, len(h.attrs)+len(attrs))
	copy(combined, h.attrs)
	copy(combined[len(h.attrs):], attrs)

	return &tuiSlogHandler{tui: h.tui, attrs: combined}
}

func (h *tuiSlogHandler) WithGroup(_ string) slog.Handler {
	return h
}

// ── File Logging ─────────────────────────────────────────────────────

const (
	maxLogSize  = 5 * 1024 * 1024 // 5 MB per log file
	maxLogFiles = 3               // keep current + 2 rotated
)

// logDir returns the XDG-compliant state directory for logs.
func logDir() string {
	if dir := os.Getenv("XDG_STATE_HOME"); dir != "" {
		return filepath.Join(dir, "tanrenai")
	}
	if runtime.GOOS == "windows" {
		return filepath.Join(os.Getenv("LOCALAPPDATA"), "tanrenai", "logs")
	}
	home, _ := os.UserHomeDir()

	return filepath.Join(home, ".local", "state", "tanrenai")
}

// openLogFile opens the log file, rotating if it exceeds maxLogSize.
func openLogFile() (*os.File, error) {
	dir := logDir()
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	logPath := filepath.Join(dir, "tanrenai.log")

	// Rotate if current log is too large.
	if info, err := os.Stat(logPath); err == nil && info.Size() > maxLogSize {
		rotated := filepath.Join(dir, fmt.Sprintf("tanrenai.%d.log", time.Now().Unix()))
		_ = os.Rename(logPath, rotated)
		cleanupOldLogs(dir)
	}

	return os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
}

// cleanupOldLogs removes rotated logs beyond maxLogFiles.
func cleanupOldLogs(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	var rotated []string
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, "tanrenai.") && strings.HasSuffix(name, ".log") && name != "tanrenai.log" {
			rotated = append(rotated, name)
		}
	}
	if len(rotated) <= maxLogFiles-1 {
		return
	}
	sort.Strings(rotated)
	for _, name := range rotated[:len(rotated)-(maxLogFiles-1)] {
		_ = os.Remove(filepath.Join(dir, name))
	}
}

// multiHandler fans out slog records to multiple handlers.
type multiHandler struct {
	handlers []slog.Handler
}

func (m *multiHandler) Enabled(ctx context.Context, level slog.Level) bool {
	for _, h := range m.handlers {
		if h.Enabled(ctx, level) {
			return true
		}
	}

	return false
}

func (m *multiHandler) Handle(ctx context.Context, r slog.Record) error {
	for _, h := range m.handlers {
		if h.Enabled(ctx, r.Level) {
			_ = h.Handle(ctx, r)
		}
	}

	return nil
}

func (m *multiHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	handlers := make([]slog.Handler, len(m.handlers))
	for i, h := range m.handlers {
		handlers[i] = h.WithAttrs(attrs)
	}

	return &multiHandler{handlers: handlers}
}

func (m *multiHandler) WithGroup(name string) slog.Handler {
	handlers := make([]slog.Handler, len(m.handlers))
	for i, h := range m.handlers {
		handlers[i] = h.WithGroup(name)
	}

	return &multiHandler{handlers: handlers}
}

func calibrateEstimator(client *apiclient.Client, estimator *chatctx.TokenEstimator, log *startupLog) {
	tokenizeFn := func(text string) (int, error) {
		return client.Tokenize(context.Background(), text)
	}
	if err := estimator.Calibrate(tokenizeFn); err != nil {
		log.Warn("Token estimation using default ratio (calibration unavailable)")
	}
}

func loadContextFile(mgr *chatctx.Manager, path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	mgr.AddContextFile(path, string(data))

	return fmt.Sprintf("Loaded context file: %s (%d bytes)", path, len(data)), nil
}

func handleREPLCommand(w io.Writer, input string, mgr *chatctx.Manager, client *apiclient.Client, memoryEnabled bool) bool {
	switch {
	case input == "/clear":
		return handleClearCommand(w, mgr)
	case input == "/tokens":
		return handleTokensCommand(w, mgr)
	case input == "/context list", input == "/context clear":
		return handleContextCommand(w, input, mgr)
	case strings.HasPrefix(input, "/context add "):
		return handleContextCommand(w, input, mgr)
	case input == "/memory" || input == "/memory list":
		return handleMemoryListCommand(w, client, memoryEnabled)
	case strings.HasPrefix(input, "/memory search "):
		return handleMemorySearchCommand(w, input, client, memoryEnabled)
	case strings.HasPrefix(input, "/memory forget "):
		return handleMemoryForgetCommand(w, input, client, memoryEnabled)
	case input == "/memory clear":
		return handleMemoryClearCommand(w, client, memoryEnabled)
	case input == "/help":
		return handleHelpCommand(w)
	}

	return false
}

func handleClearCommand(w io.Writer, mgr *chatctx.Manager) bool {
	mgr.Clear()
	_, _ = fmt.Fprintln(w, "History cleared. System prompt and context files preserved.")

	return true
}

func handleTokensCommand(w io.Writer, mgr *chatctx.Manager) bool {
	budget := mgr.Budget()
	_, _ = fmt.Fprintf(w, "Token budget:\n")
	_, _ = fmt.Fprintf(w, "  Total context:  %d\n", budget.Total)
	_, _ = fmt.Fprintf(w, "  System/pinned:  %d\n", budget.System)
	_, _ = fmt.Fprintf(w, "  Scrolls:        %d\n", budget.Scrolls)
	_, _ = fmt.Fprintf(w, "  Memory:         %d\n", budget.Memory)
	_, _ = fmt.Fprintf(w, "  Summary:        %d\n", budget.Summary)
	_, _ = fmt.Fprintf(w, "  History:        %d (%d messages, %d total)\n", budget.History, budget.HistoryCount, budget.TotalHistory)
	_, _ = fmt.Fprintf(w, "  Available:      %d\n", budget.Available)

	return true
}

func handleContextCommand(w io.Writer, input string, mgr *chatctx.Manager) bool {
	switch input {
	case "/context list":
		files := mgr.ContextFiles()
		if len(files) == 0 {
			_, _ = fmt.Fprintln(w, "No context files loaded.")
		} else {
			_, _ = fmt.Fprintln(w, "Context files:")
			for _, f := range files {
				_, _ = fmt.Fprintf(w, "  - %s\n", f)
			}
		}

		return true

	case "/context clear":
		mgr.ClearContextFiles()
		_, _ = fmt.Fprintln(w, "Context files cleared.")

		return true

	default:
		// "/context add <path>"
		path := strings.TrimPrefix(input, "/context add ")
		path = strings.TrimSpace(path)
		if path == "" {
			_, _ = fmt.Fprintln(w, "Usage: /context add <file-path>")

			return true
		}
		msg, err := loadContextFile(mgr, path)
		if err != nil {
			_, _ = fmt.Fprintf(w, "Error: %v\n", err)
		} else {
			_, _ = fmt.Fprintln(w, msg)
		}

		return true
	}
}

func handleMemoryListCommand(w io.Writer, client *apiclient.Client, memoryEnabled bool) bool {
	if !memoryEnabled {
		_, _ = fmt.Fprintln(w, "Memory is not enabled. Use --memory flag to enable.")

		return true
	}
	resp, err := client.MemoryList(context.Background(), 10)
	if err != nil {
		_, _ = fmt.Fprintf(w, "Error listing memories: %v\n", err)

		return true
	}
	_, _ = fmt.Fprintf(w, "Memories (%d total):\n", resp.Total)
	for _, e := range resp.Entries {
		_, _ = fmt.Fprintf(w, "  [%s] %s — %s\n", e.ID[:8], e.Timestamp.Format("2006-01-02 15:04"), truncate(e.UserMsg, 80))
	}

	return true
}

func handleMemorySearchCommand(w io.Writer, input string, client *apiclient.Client, memoryEnabled bool) bool {
	if !memoryEnabled {
		_, _ = fmt.Fprintln(w, "Memory is not enabled. Use --memory flag to enable.")

		return true
	}
	query := strings.TrimPrefix(input, "/memory search ")
	query = strings.TrimSpace(query)
	if query == "" {
		_, _ = fmt.Fprintln(w, "Usage: /memory search <query>")

		return true
	}
	resp, err := client.MemorySearch(context.Background(), query, 5)
	if err != nil {
		_, _ = fmt.Fprintf(w, "Error searching memories: %v\n", err)

		return true
	}
	if len(resp.Results) == 0 {
		_, _ = fmt.Fprintln(w, "No matching memories found.")

		return true
	}
	_, _ = fmt.Fprintf(w, "Search results (%d):\n", len(resp.Results))
	for _, r := range resp.Results {
		_, _ = fmt.Fprintf(w, "  [%s] score=%.3f (sem=%.3f kw=%.3f) %s\n",
			r.Entry.ID[:8], r.CombinedScore, r.SemanticScore, r.KeywordScore,
			truncate(r.Entry.UserMsg, 70))
	}

	return true
}

func handleMemoryForgetCommand(w io.Writer, input string, client *apiclient.Client, memoryEnabled bool) bool {
	if !memoryEnabled {
		_, _ = fmt.Fprintln(w, "Memory is not enabled. Use --memory flag to enable.")

		return true
	}
	idPrefix := strings.TrimPrefix(input, "/memory forget ")
	idPrefix = strings.TrimSpace(idPrefix)
	if idPrefix == "" {
		_, _ = fmt.Fprintln(w, "Usage: /memory forget <id-prefix>")

		return true
	}
	resp, err := client.MemoryList(context.Background(), 0)
	if err != nil {
		_, _ = fmt.Fprintf(w, "Error: %v\n", err)

		return true
	}
	for _, e := range resp.Entries {
		if strings.HasPrefix(e.ID, idPrefix) {
			if err := client.MemoryDelete(context.Background(), e.ID); err != nil {
				_, _ = fmt.Fprintf(w, "Error deleting memory: %v\n", err)
			} else {
				_, _ = fmt.Fprintf(w, "Deleted memory %s\n", e.ID[:8])
			}

			return true
		}
	}
	_, _ = fmt.Fprintf(w, "No memory found with prefix %q\n", idPrefix)

	return true
}

func handleMemoryClearCommand(w io.Writer, client *apiclient.Client, memoryEnabled bool) bool {
	if !memoryEnabled {
		_, _ = fmt.Fprintln(w, "Memory is not enabled. Use --memory flag to enable.")

		return true
	}
	if err := client.MemoryClear(context.Background()); err != nil {
		_, _ = fmt.Fprintf(w, "Error clearing memories: %v\n", err)
	} else {
		_, _ = fmt.Fprintln(w, "All memories cleared.")
	}

	return true
}

func handleHelpCommand(w io.Writer) bool {
	_, _ = fmt.Fprintln(w, "Commands:")
	_, _ = fmt.Fprintln(w, "  /clear                        - Clear conversation history")
	_, _ = fmt.Fprintln(w, "  /compact                      - Summarize conversation to free context")
	_, _ = fmt.Fprintln(w, "  /tokens                       - Show token budget breakdown")
	_, _ = fmt.Fprintln(w, "  /context add <path>           - Load file into context")
	_, _ = fmt.Fprintln(w, "  /context list                 - Show loaded context files")
	_, _ = fmt.Fprintln(w, "  /context clear                - Remove all context files")
	_, _ = fmt.Fprintln(w, "  /memory                       - List recent memories")
	_, _ = fmt.Fprintln(w, "  /memory search <q>            - Search memories")
	_, _ = fmt.Fprintln(w, "  /memory forget <id>           - Delete a memory by ID prefix")
	_, _ = fmt.Fprintln(w, "  /memory clear                 - Clear all memories")
	_, _ = fmt.Fprintln(w, "  /scrolls                      - List loaded scrolls")
	_, _ = fmt.Fprintln(w, "  /scrolls show <name>          - Show a scroll's content")
	_, _ = fmt.Fprintln(w, "  /swarm [on|off]               - Toggle multi-agent swarm mode")
	_, _ = fmt.Fprintln(w, "  /quit, /exit                  - Exit")

	return true
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}

	return s[:max] + "..."
}

// chatStreamHooks are callbacks for the shared simple-chat streaming loop.
type chatStreamHooks struct {
	OnThinking     func()
	OnThinkingDone func()
	OnContentDelta func(delta string)
}

// streamSimpleChat runs a single streaming chat completion and invokes hooks
// for thinking and content deltas. Returns the accumulated assistant content.
// Both TUI and pipe mode call this so the logic lives in one place.
func streamSimpleChat(events <-chan apiclient.StreamEvent, hooks chatStreamHooks) (string, error) {
	var full strings.Builder
	thinking := false
	for ev := range events {
		if ev.Err != nil {
			return full.String(), ev.Err
		}
		if ev.Done {
			break
		}
		if ev.Chunk == nil {
			continue
		}
		for _, choice := range ev.Chunk.Choices {
			if choice.Delta.ReasoningContent != "" && !thinking {
				thinking = true
				if hooks.OnThinking != nil {
					hooks.OnThinking()
				}
			}
			if choice.Delta.Content != "" {
				if thinking {
					thinking = false
					if hooks.OnThinkingDone != nil {
						hooks.OnThinkingDone()
					}
				}
				full.WriteString(choice.Delta.Content)
				if hooks.OnContentDelta != nil {
					hooks.OnContentDelta(choice.Delta.Content)
				}
			}
		}
	}
	return full.String(), nil
}

// runParams captures all parsed flags shared by the run and chat commands.
type runParams struct {
	model          string
	systemPrompt   string
	agentMode      bool
	ctxSize        int
	ctxSizeChanged bool
	responseBudget int
	contextFiles   []string
	memoryEnabled  bool
	maxIterations  int
	maxTokens      int
	noScrolls      bool
	thinking       bool
	swarmMode      bool
	local          bool
	gpuLayers      int
	flashAttn      bool
	cpuMoE         bool
	noKVOffload    bool
	fitVRAM        bool
	// pipeFormat controls how pipe mode serialises output: "text" (default,
	// human-friendly with ---END--- delimiters and bracketed stderr status)
	// or "json" (JSONL events on stdout for programmatic consumers like
	// editor integrations, IDE extensions, and agent hosts).
	pipeFormat string
}

// sessionDeps holds the initialised resources for a chat/agent session.
type sessionDeps struct {
	client         *apiclient.Client
	mgr            *chatctx.Manager
	registry       *tools.Registry
	procMgr        *tools.ProcessManager
	memoryEnabled  bool
	allScrolls     []scrolls.Scroll
	scrollsEnabled bool
	maxIterations  int
	maxTokens      int
	enableThinking bool
	agentMode      bool
	swarmMode      bool
	completeFn     agent.CompletionFunc
	streamFn       agent.StreamingCompletionFunc
	cleanupFn      func()
	modelName      string
	// pipeFormat propagates the --format flag into the pipe hook builders
	// so they can route output to JSONL stdout (and stderr-only status)
	// for programmatic consumers, instead of the default text-mode mix.
	// Only consulted from pipe.go — TUI sessions ignore it.
	pipeFormat string
}

// setupSession initialises the backend client, model, context manager,
// tools, scrolls, and memory — everything TUI, pipe, and agent-rpc need.
func setupSession(ctx context.Context, p runParams, log *startupLog) (*sessionDeps, error) {
	deps := &sessionDeps{modelName: p.model}

	// Ensure the project-local `.tanrenai/` exists in cwd. Home for
	// architect.md (swarm), project-scoped scrolls, and permissions.json.
	// Non-fatal — read-only workspaces still get chat.
	ensureProjectDir(log)

	mode := resolveSessionMode(p, log)

	activeURL := serverURL
	if mode == sessionModeLocal {
		opts := localOpts{
			GPULayers:      p.gpuLayers,
			FlashAttention: p.flashAttn,
			MemoryEnabled:  p.memoryEnabled,
			CPUMoE:         p.cpuMoE,
			NoKVOffload:    p.noKVOffload,
			FitVRAM:        p.fitVRAM,
		}
		if p.memoryEnabled {
			if err := ensureEmbeddingModel(log); err != nil {
				return nil, err
			}
			opts.EmbeddingModel = defaultEmbeddingModel
			opts.MemoryDir = projectMemoryDir()
		}
		url, cleanup, err := startLocalServers(ctx, opts, log)
		if err != nil {
			return nil, err
		}
		deps.cleanupFn = cleanup
		activeURL = url
	} else if mode == sessionModeRemote && p.memoryEnabled {
		// Memory features use a local embedding model even in remote mode —
		// one-time download, works offline, avoids an extra network hop.
		if err := ensureEmbeddingModel(log); err != nil {
			return nil, err
		}
	}

	client := newAuthedClient(activeURL, authToken)
	deps.client = client

	modelToLoad := p.model
	if mode == sessionModeRemote && isModelURI(p.model) {
		resolved, err := pullModelForRemote(ctx, client, p.model, log)
		if err != nil {
			return nil, fmt.Errorf("pull model: %w", err)
		}
		modelToLoad = resolved
		p.model = resolved
		deps.modelName = resolved
	}

	log.Info("Loading model " + modelToLoad + "...")

	loadResp, err := loadModelWithProgress(ctx, client, mode, modelToLoad, log)
	if err != nil {
		// A 401 that survived the transport's refresh-and-retry means the
		// stored credentials are dead — "is the backend running?" would be
		// misleading. Tell the user the one thing that fixes it.
		var se *apiclient.StatusError
		if errors.As(err, &se) && se.Code == http.StatusUnauthorized {
			return nil, fmt.Errorf("authentication failed — your session has expired; run `tanrenai login` to sign in again (%w)", err)
		}
		return nil, fmt.Errorf("failed to load model (is the backend running?): %w", err)
	}
	// In hosted mode the user may have passed an empty model so the platform
	// picks GPU_MODEL — /api/load reports the actual name back. Carry it onto
	// deps so the TUI footer and the agent-rpc ready handshake both show
	// what's really loaded instead of a blank.
	if loadResp.Model != "" {
		deps.modelName = loadResp.Model
	}

	ctxSize := p.ctxSize
	if !p.ctxSizeChanged && loadResp.CtxSize > 0 {
		ctxSize = loadResp.CtxSize
		log.Info(fmt.Sprintf("Using model context size: %d tokens", ctxSize))
	}
	if ctxSize == 0 {
		ctxSize = 4096
	}

	estimator := chatctx.NewTokenEstimator()
	calibrateEstimator(client, estimator, log)

	toolsBudget := 0
	if p.agentMode {
		toolsBudget = 4000
	}

	mgr := chatctx.NewManager(chatctx.Config{
		CtxSize:        ctxSize,
		ResponseBudget: p.responseBudget,
		ToolsBudget:    toolsBudget,
	}, estimator)
	deps.mgr = mgr

	for _, path := range p.contextFiles {
		msg, loadErr := loadContextFile(mgr, path)
		if loadErr != nil {
			log.Warn(fmt.Sprintf("Failed to load context file %s: %v", path, loadErr))
		} else {
			log.Info(msg)
		}
	}

	// Load scrolls
	var allScrolls []scrolls.Scroll
	if !p.noScrolls {
		projectScrollsDir := filepath.Join(".tanrenai", "scrolls")
		globalScrollsDir := filepath.Join(tools.GlobalConfigDir(), "scrolls")
		allScrolls, err = scrolls.Load(projectScrollsDir, globalScrollsDir)
		if err != nil {
			log.Warn(fmt.Sprintf("Failed to load scrolls: %v", err))
		} else if len(allScrolls) > 0 {
			log.Info(fmt.Sprintf("Loaded %d scrolls", len(allScrolls)))
		}
	}
	deps.allScrolls = allScrolls
	deps.scrollsEnabled = !p.noScrolls && len(allScrolls) > 0

	memoryEnabled := p.memoryEnabled
	if memoryEnabled && p.agentMode {
		count, memErr := client.MemoryCount(ctx)
		if memErr != nil {
			log.Warn(fmt.Sprintf("Memory not available: %v", memErr))
			memoryEnabled = false
		} else {
			log.Info(fmt.Sprintf("Memory enabled (%d stored memories)", count))
		}
	}
	deps.memoryEnabled = memoryEnabled

	// Configure system prompt
	if p.agentMode {
		agentSystem := defaultAgentSystemPrompt
		if p.systemPrompt != "" {
			agentSystem += "\n\n" + p.systemPrompt
		}
		mgr.SetSystemPrompt(agentSystem)
	} else if p.systemPrompt != "" {
		mgr.SetSystemPrompt(p.systemPrompt)
	}

	if p.agentMode {
		registry := tools.DefaultRegistry()
		procMgr := tools.NewProcessManager()
		if st, ok := registry.Get("shell_exec").(*tools.ShellExecTool); ok {
			st.ProcessManager = procMgr
		}
		deps.registry = registry
		deps.procMgr = procMgr

		// External MCP servers (configured via .tanrenai/mcp.json
		// and/or ~/.config/tanrenai/mcp.json) get connected here and
		// their tools attached to the same registry the agent reads.
		// Per-server failures don't block startup — a flaky MCP server
		// shouldn't make `tanrenai run` unusable. The deps.cleanupFn
		// chain disconnects everything when the session ends.
		attachMCP(ctx, p, deps, log)
	}

	deps.maxIterations = p.maxIterations
	deps.maxTokens = p.maxTokens
	deps.enableThinking = p.thinking
	deps.agentMode = p.agentMode
	deps.swarmMode = p.swarmMode
	deps.pipeFormat = p.pipeFormat

	model := p.model
	deps.completeFn = func(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		req.Model = model
		return client.ChatCompletion(ctx, req)
	}
	deps.streamFn = func(ctx context.Context, req *api.ChatCompletionRequest) (<-chan apiclient.StreamEvent, error) {
		req.Model = model
		return client.StreamCompletion(ctx, req)
	}

	// In remote mode, wrap stream/complete so a transient provisioning
	// error (GPU paused, model unloaded) waits for the platform to bring
	// the GPU back and then retries — the conversation in deps.mgr is
	// untouched, so the model gets the full prior context on retry.
	if mode == sessionModeRemote {
		deps.streamFn = withStreamGPURetry(deps.streamFn, client, model, log)
		deps.completeFn = withCompleteGPURetry(deps.completeFn, client, model, log)
	}

	return deps, nil
}

// withStreamGPURetry wraps a StreamingCompletionFunc so that if the call
// fails before streaming begins with an error indicating the platform is
// provisioning / waking the GPU, it waits for the model to load and
// retries. Once a stream has started successfully, errors mid-stream are
// passed through to the caller — those are handled by the agent loop's
// own retry-with-backoff.
func withStreamGPURetry(
	inner agent.StreamingCompletionFunc,
	client *apiclient.Client,
	model string,
	log *startupLog,
) agent.StreamingCompletionFunc {
	return func(ctx context.Context, req *api.ChatCompletionRequest) (<-chan apiclient.StreamEvent, error) {
		ch, err := inner(ctx, req)
		if err == nil || !isProvisioningInProgress(err) {
			return ch, err
		}
		log.Info("GPU is sleeping — waking it up...")
		if _, lerr := loadModelWithProgress(ctx, client, sessionModeRemote, model, log); lerr != nil {
			return nil, fmt.Errorf("backend did not come back: %w", lerr)
		}

		return inner(ctx, req)
	}
}

// withCompleteGPURetry is the non-streaming equivalent.
func withCompleteGPURetry(
	inner agent.CompletionFunc,
	client *apiclient.Client,
	model string,
	log *startupLog,
) agent.CompletionFunc {
	return func(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		resp, err := inner(ctx, req)
		if err == nil || !isProvisioningInProgress(err) {
			return resp, err
		}
		log.Info("GPU is sleeping — waking it up...")
		if _, lerr := loadModelWithProgress(ctx, client, sessionModeRemote, model, log); lerr != nil {
			return nil, fmt.Errorf("backend did not come back: %w", lerr)
		}

		return inner(ctx, req)
	}
}

// assignToTUI copies sessionDeps fields onto a tuiApp.
func assignToTUI(t *tuiApp, deps *sessionDeps) {
	t.client = deps.client
	t.mgr = deps.mgr
	t.registry = deps.registry
	t.memoryEnabled = deps.memoryEnabled
	t.allScrolls = deps.allScrolls
	t.scrollsEnabled = deps.scrollsEnabled
	t.maxIterations = deps.maxIterations
	t.maxResponseTokens = deps.maxTokens
	t.enableThinking = deps.enableThinking
	t.agentMode = deps.agentMode
	t.swarmMode = deps.swarmMode
	t.completeFn = deps.completeFn
	t.streamFn = deps.streamFn
	if deps.cleanupFn != nil {
		t.cleanupFn = deps.cleanupFn
	}
	if deps.procMgr != nil {
		t.procMgr = deps.procMgr
		deps.procMgr.OnChange = func() {
			t.app.QueueUpdateDraw(func() {
				t.refreshProcessPanel()
			})
		}
	}
}

func addRunFlags(cmd *cobra.Command) {
	cmd.Flags().String("system", "", "system prompt")
	cmd.Flags().String("system-file", "", "read system prompt from file")
	cmd.Flags().Bool("agent", false, "enable agent mode with tool calling")
	cmd.Flags().Int("ctx-size", 0, "context window size in tokens (0 = auto-detect from model)")
	cmd.Flags().Int("response-budget", 512, "tokens reserved for model response")
	cmd.Flags().StringSlice("context-file", nil, "files to load into context")
	cmd.Flags().Bool("memory", false, "enable memory/RAG")
	cmd.Flags().Int("max-iterations", 0, "maximum agent tool-call iterations per turn (0 = unlimited)")
	cmd.Flags().Int("max-tokens", 0, "max tokens per model response (0 = default 16384)")
	cmd.Flags().Bool("no-scrolls", false, "disable automatic scroll injection")
	cmd.Flags().Bool("thinking", true, "enable thinking/reasoning mode (for models that support it)")
	cmd.Flags().Bool("pipe", false, "non-interactive pipe mode: read from stdin, write to stdout")
	cmd.Flags().String("format", "text", "pipe output format: 'text' (default, with ---END--- delimiters and bracketed stderr status) or 'json' (JSONL events on stdout for programmatic consumers)")
	cmd.Flags().Bool("swarm", false, "multi-agent swarm mode: orchestrator plans, workers execute with fresh contexts")
}

var _ = time.Now

func init() {
	addRunFlags(runCmd)
	chatCmd.Flags().String("model", "", "model to chat with")
	addRunFlags(chatCmd)
	rootCmd.AddCommand(runCmd)
	rootCmd.AddCommand(chatCmd)
}
