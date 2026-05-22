package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"

	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/shared/agent"
	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/scrolls"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

// slashCommands defines the available REPL commands for autocomplete.
var slashCommands = []struct {
	cmd  string // command text (trailing space = expects argument)
	desc string
}{
	{"/clear", "Clear conversation history"},
	{"/compact", "Summarize to free context"},
	{"/tokens", "Show token budget"},
	{"/context add ", "Load file into context"},
	{"/context list", "Show loaded files"},
	{"/context clear", "Remove all context files"},
	{"/memory", "List recent memories"},
	{"/memory search ", "Search memories"},
	{"/memory forget ", "Delete a memory"},
	{"/memory clear", "Clear all memories"},
	{"/scrolls", "List loaded scrolls"},
	{"/scrolls show ", "Show a scroll's content"},
	{"/swarm", "Toggle multi-agent swarm mode"},
	{"/help", "Show help"},
	{"/quit", "Exit"},
	{"/exit", "Exit"},
}

type focusTarget int

const (
	focusChat focusTarget = iota
	focusFileViewer
	focusProcessPanel
)

type iterRecord struct {
	inputTokens int
	duration    time.Duration
}

// tuiApp is the single mutable state struct for the tview-based TUI.
type tuiApp struct {
	app      *tview.Application
	pages    *tview.Pages // root container: "main" page + modal overlays
	rootFlex *tview.Flex  // vertical: chatArea + hDiv + inputFlex + hDiv
	chatArea *tview.Flex  // horizontal: chatView [+ vDiv + filePanel]
	chatView *tview.TextView

	// File viewer widgets (created on demand)
	filePanel  *tview.Flex     // vertical: fileHeader + fileView
	fileHeader *tview.TextView // 1-line file path + hints
	fileView   *tview.TextView // scrollable syntax-highlighted content

	// Process panel (created on demand)
	procMgr      *tools.ProcessManager
	processPanel *tview.Table

	inputArea  *tview.TextArea
	statusBar  *tview.TextView
	statusText string

	// Swarm progress panel (above status bar; hidden when not in a swarm turn)
	swarmPanel   *tview.TextView
	swarmPlan    *agent.Plan
	swarmDepth   int
	swarmRunning bool

	mu            sync.Mutex
	lines         []string
	toolResults   map[int]string       // line index -> full tool result
	toolCallLines map[int]api.ToolCall // line index -> original tool call
	expanded      bool                 // Tab toggles full tool output
	filePath      string               // "" = no file viewer open
	focus         focusTarget
	loading       bool // true until startup goroutine completes
	processing    bool
	ctrlCPending  bool
	streaming     strings.Builder
	turnCancel    context.CancelFunc
	cleanupFn     func() // called on app exit (e.g. stop local servers)
	memoryWg      sync.WaitGroup
	liveCtxTokens int // live token count from agent loop (0 = use mgr.Budget())

	// genRate tracks tokens-per-second during the current streaming turn.
	// Reset at iteration start, Record()ed on each content delta, read by
	// the status-bar renderer. Pointer so tui_view.go's renderer can safely
	// pass it around without copying the mutex.
	genRate *apiclient.TokenRateTracker

	// Plan-execute agent mode
	userInputCh chan string // non-nil during planned agent turns
	plannedMode bool        // true when a planned agent turn is running

	// Slash command autocomplete
	acList     *tview.List // popup list (added/removed from pages)
	acActive   bool        // true when autocomplete popup is visible
	acSuppress bool        // temporarily suppress updateAutocomplete

	// Loading animation
	anvilFrame int // current animation frame (-1 = not animating)
	anvilStop  chan struct{}

	// Progress tracking
	iterStartTime     time.Time
	iterHistory       []iterRecord // persists across turns — never reset
	currentIterTokens int          // input tokens for the current iteration
	currentIterOutput int          // output chars accumulated this iteration
	lastInputTokens   int          // input tokens for status bar display
	lastOutputTokens  int          // output tokens for status bar display
	estimatedDur      time.Duration
	progressTicker    *time.Ticker
	progressStop      chan struct{}

	// Dependencies (set by startup goroutine, then immutable)
	client            *apiclient.Client
	modelName         string
	mgr               *chatctx.Manager
	registry          *tools.Registry
	permissions       *tools.Permissions
	memoryEnabled     bool
	allScrolls        []scrolls.Scroll
	scrollsEnabled    bool
	maxIterations     int
	maxResponseTokens int
	enableThinking    bool
	agentMode         bool
	swarmMode         bool
	completeFn        agent.CompletionFunc
	streamFn          agent.StreamingCompletionFunc
}

func newTuiApp(modelName string) *tuiApp {
	t := &tuiApp{
		toolResults:   make(map[int]string),
		toolCallLines: make(map[int]api.ToolCall),
		focus:         focusChat,
		modelName:     modelName,
		permissions:   tools.LoadPermissions(),
		anvilFrame:    -1, // not animating
		genRate:       &apiclient.TokenRateTracker{},
	}

	t.app = tview.NewApplication()

	// Chat view
	t.chatView = tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(true).
		SetWordWrap(true).
		SetChangedFunc(func() { t.app.Draw() })
	t.chatView.SetBorder(false)

	// Swarm progress panel (hidden by default; shown during swarm turns)
	t.swarmPanel = tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(false).
		SetWrap(false)
	t.swarmPanel.SetBorder(false)

	// Status bar (fixed 1-row panel above input)
	t.statusBar = tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(false)
	t.statusBar.SetBorder(false)

	// Input area (multiline)
	t.inputArea = tview.NewTextArea().
		SetLabel("[blue::b] > [-:-:-]").
		SetLabelWidth(4).
		SetPlaceholder("Loading...").
		SetWordWrap(true).
		SetWrap(true)
	t.inputArea.SetBorder(false)
	t.inputArea.SetBackgroundColor(tcell.ColorDefault)
	t.inputArea.SetTextStyle(tcell.StyleDefault.Background(tcell.ColorDefault))
	t.inputArea.SetChangedFunc(func() { t.updateAutocomplete() })

	// Build layout
	t.chatArea = tview.NewFlex().SetDirection(tview.FlexColumn)
	t.chatArea.AddItem(t.chatView, 0, 1, false)

	t.rootFlex = tview.NewFlex().SetDirection(tview.FlexRow)
	t.rootFlex.AddItem(t.chatArea, 0, 1, false)
	t.rootFlex.AddItem(t.swarmPanel, 0, 0, false) // hidden until swarm turn starts
	t.rootFlex.AddItem(t.statusBar, 1, 0, false)
	t.rootFlex.AddItem(newHDivider(), 1, 0, false)
	t.rootFlex.AddItem(t.inputArea, 3, 0, true)
	t.rootFlex.AddItem(newHDivider(), 1, 0, false)

	t.pages = tview.NewPages()
	t.pages.AddPage("main", t.rootFlex, true, true)

	t.setupInputCapture()
	t.setupMouseCapture()

	return t
}

// newHDivider creates a 1-row box that draws a horizontal line.
func newHDivider() *tview.Box {
	box := tview.NewBox()
	box.SetDrawFunc(func(screen tcell.Screen, x, y, width, height int) (int, int, int, int) {
		style := tcell.StyleDefault.Foreground(tcell.ColorDarkGray)
		for cx := x; cx < x+width; cx++ {
			screen.SetContent(cx, y, tcell.RuneHLine, nil, style)
		}

		return x, y, width, height
	})

	return box
}

// newVDivider creates a 1-col box that draws a vertical line.
func newVDivider(focused bool) *tview.Box {
	box := tview.NewBox()
	box.SetDrawFunc(func(screen tcell.Screen, x, y, width, height int) (int, int, int, int) {
		color := tcell.ColorDarkGray
		if focused {
			color = tcell.ColorBlue
		}
		style := tcell.StyleDefault.Foreground(color)
		for cy := y; cy < y+height; cy++ {
			screen.SetContent(x, cy, tcell.RuneVLine, nil, style)
		}

		return x, y, width, height
	})

	return box
}

func (t *tuiApp) run() error {
	err := t.app.SetRoot(t.pages, true).EnableMouse(true).EnablePaste(true).Run()
	// Wait for pending memory stores to finish (up to 2s).
	done := make(chan struct{})
	go func() {
		t.memoryWg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
	}
	if t.procMgr != nil {
		t.procMgr.KillAll()
	}
	if t.cleanupFn != nil {
		t.cleanupFn()
	}

	return err
}

// ── Content Management ──────────────────────────────────────────────────

func (t *tuiApp) addLine(line string) {
	t.lines = append(t.lines, line)
}

// appendLogLine adds a line to the chat view from any goroutine (thread-safe).
func (t *tuiApp) appendLogLine(line string) {
	t.app.QueueUpdateDraw(func() {
		t.lines = append(t.lines, line)
		t.refreshChatView()
	})
}

func (t *tuiApp) updateStreamingLine() {
	content := t.streaming.String()
	formatted := fmt.Sprintf(" [purple::b] * [-:-:-]%s", tview.Escape(content))
	contentLines := strings.Split(formatted, "\n")

	// Find where streaming started (after last user prefix + blank)
	streamStart := len(t.lines)
	for i := len(t.lines) - 1; i >= 0; i-- {
		if strings.Contains(t.lines[i], ">>>") {
			streamStart = i + 2

			break
		}
	}
	if streamStart > len(t.lines) {
		streamStart = len(t.lines)
	}

	t.lines = append(t.lines[:streamStart], contentLines...)
}

func (t *tuiApp) refreshChatView() {
	var parts []string

	// Show braille logo + spinner during loading
	if t.anvilFrame >= 0 {
		parts = append(parts, renderLoadingArt(t.anvilFrame))
	}

	if !t.expanded || len(t.toolResults) == 0 {
		parts = append(parts, strings.Join(t.lines, "\n"))
	} else {
		var built []string
		for i, line := range t.lines {
			if full, ok := t.toolResults[i]; ok {
				for _, fline := range strings.Split(strings.TrimRight(full, "\n"), "\n") {
					built = append(built, colorizeDiffLine(fline))
				}
			} else {
				built = append(built, line)
			}
		}
		parts = append(parts, strings.Join(built, "\n"))
	}

	t.chatView.SetText(strings.Join(parts, "\n"))
	t.chatView.ScrollToEnd()
}

// colorizeDiffLine applies tview color tags to a single line of unified diff output.
// Uses background highlights similar to IDE diff views.
func colorizeDiffLine(line string) string {
	escaped := tview.Escape(line)
	prefix := "      "
	if len(line) == 0 {
		return "[gray::-]" + prefix + "[-:-:-]"
	}
	switch line[0] {
	case '+':
		return "[green:#1a3a1a:-]" + prefix + escaped + "[-:-:-]"
	case '-':
		return "[red:#3a1a1a:-]" + prefix + escaped + "[-:-:-]"
	case '@':
		return "[#6688cc::-]" + prefix + escaped + "[-:-:-]"
	default:
		return "[gray::-]" + prefix + escaped + "[-:-:-]"
	}
}

// ── Helpers ─────────────────────────────────────────────────────────────

func extractFilePath(call api.ToolCall) string {
	var args struct {
		Path string `json:"path"`
	}
	_ = json.Unmarshal([]byte(call.Function.Arguments), &args)

	return args.Path
}

func extractShellCommand(call api.ToolCall) string {
	var args struct {
		Command string `json:"command"`
	}
	_ = json.Unmarshal([]byte(call.Function.Arguments), &args)
	cmd := args.Command
	if len(cmd) > 120 {
		cmd = cmd[:117] + "..."
	}

	return cmd
}
