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

	"github.com/ThatCatDev/tanrenai/client/internal/agent"
	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
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
	{"/help", "Show help"},
	{"/quit", "Exit"},
	{"/exit", "Exit"},
}

type focusTarget int

const (
	focusChat focusTarget = iota
	focusFileViewer
)

type iterRecord struct {
	inputTokens int
	duration    time.Duration
}

// tuiApp is the single mutable state struct for the tview-based TUI.
type tuiApp struct {
	app      *tview.Application
	rootFlex *tview.Flex // vertical: chatArea + hDiv + inputFlex + hDiv
	chatArea *tview.Flex // horizontal: chatView [+ vDiv + filePanel]
	chatView *tview.TextView

	// File viewer widgets (created on demand)
	filePanel  *tview.Flex     // vertical: fileHeader + fileView
	fileHeader *tview.TextView // 1-line file path + hints
	fileView   *tview.TextView // scrollable syntax-highlighted content

	inputField *tview.InputField
	statusBar  *tview.TextView
	statusText string

	mu            sync.Mutex
	lines         []string
	toolResults   map[int]string       // line index -> full tool result
	toolCallLines map[int]api.ToolCall // line index -> original tool call
	expanded      bool                 // Tab toggles full tool output
	filePath             string               // "" = no file viewer open
	focus                focusTarget
	autocompleteActive   bool
	autocompleteMatches  []string
	loading       bool // true until startup goroutine completes
	processing    bool
	ctrlCPending  bool
	streaming     strings.Builder
	turnCancel    context.CancelFunc
	cleanupFn     func() // called on app exit (e.g. stop local servers)

	// Progress tracking
	iterStartTime    time.Time
	iterHistory      []iterRecord // persists across turns — never reset
	currentIterTokens int         // input tokens for the current iteration
	currentIterOutput int         // output chars accumulated this iteration
	lastInputTokens  int         // input tokens for status bar display
	lastOutputTokens int         // output tokens for status bar display
	estimatedDur     time.Duration
	progressTicker   *time.Ticker
	progressStop     chan struct{}

	// Dependencies (set by startup goroutine, then immutable)
	client        *apiclient.Client
	modelName     string
	mgr           *chatctx.Manager
	registry      *tools.Registry
	memoryEnabled bool
	maxIterations int
	agentMode     bool
	completeFn    agent.CompletionFunc
	streamFn      agent.StreamingCompletionFunc
}

func newTuiApp(modelName string) *tuiApp {
	t := &tuiApp{
		toolResults:   make(map[int]string),
		toolCallLines: make(map[int]api.ToolCall),
		focus:         focusChat,
		modelName:     modelName,
	}

	t.app = tview.NewApplication()

	// Chat view
	t.chatView = tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(true).
		SetWordWrap(true).
		SetChangedFunc(func() { t.app.Draw() })
	t.chatView.SetBorder(false)

	// Status bar (fixed 1-row panel above input)
	t.statusBar = tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(false)
	t.statusBar.SetBorder(false)

	// Input field
	t.inputField = tview.NewInputField().
		SetLabel("[blue::b] > [-:-:-]").
		SetLabelWidth(4).
		SetFieldBackgroundColor(tcell.ColorDefault).
		SetPlaceholder("Loading...").
		SetPlaceholderTextColor(tcell.ColorGray)
	t.inputField.SetBorder(false)

	// Slash command autocomplete
	t.inputField.SetAutocompleteFunc(func(currentText string) []string {
		if !strings.HasPrefix(currentText, "/") {
			t.autocompleteActive = false
			t.autocompleteMatches = nil
			return nil
		}
		var entries []string
		t.autocompleteMatches = nil
		for _, sc := range slashCommands {
			if strings.HasPrefix(sc.cmd, currentText) {
				entries = append(entries, fmt.Sprintf("%-20s %s", strings.TrimRight(sc.cmd, " "), sc.desc))
				t.autocompleteMatches = append(t.autocompleteMatches, sc.cmd)
			}
		}
		t.autocompleteActive = len(entries) > 0
		return entries
	})
	t.inputField.SetAutocompletedFunc(func(text string, index int, source int) bool {
		if source == tview.AutocompletedNavigate {
			return false // just highlight, don't select
		}
		if index >= 0 && index < len(t.autocompleteMatches) {
			t.inputField.SetText(t.autocompleteMatches[index])
		}
		t.autocompleteActive = false
		return true
	})
	t.inputField.SetAutocompleteStyles(
		tcell.ColorDarkSlateGray,
		tcell.StyleDefault.Foreground(tcell.ColorWhite).Background(tcell.ColorDarkSlateGray),
		tcell.StyleDefault.Foreground(tcell.ColorWhite).Background(tcell.ColorBlue),
	)

	// Build layout
	t.chatArea = tview.NewFlex().SetDirection(tview.FlexColumn)
	t.chatArea.AddItem(t.chatView, 0, 1, false)

	t.rootFlex = tview.NewFlex().SetDirection(tview.FlexRow)
	t.rootFlex.AddItem(t.chatArea, 0, 1, false)
	t.rootFlex.AddItem(t.statusBar, 1, 0, false)
	t.rootFlex.AddItem(newHDivider(), 1, 0, false)
	t.rootFlex.AddItem(t.inputField, 1, 0, true)
	t.rootFlex.AddItem(newHDivider(), 1, 0, false)

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
	err := t.app.SetRoot(t.rootFlex, true).EnableMouse(true).Run()
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
	var content string
	if !t.expanded || len(t.toolResults) == 0 {
		content = strings.Join(t.lines, "\n")
	} else {
		var built []string
		for i, line := range t.lines {
			if full, ok := t.toolResults[i]; ok {
				for _, fline := range strings.Split(strings.TrimRight(full, "\n"), "\n") {
					built = append(built, "[gray::-]      "+tview.Escape(fline)+"[-:-:-]")
				}
			} else {
				built = append(built, line)
			}
		}
		content = strings.Join(built, "\n")
	}
	t.chatView.SetText(content)
	t.chatView.ScrollToEnd()
}

// ── Helpers ─────────────────────────────────────────────────────────────

func extractFilePath(call api.ToolCall) string {
	var args struct {
		Path string `json:"path"`
	}
	_ = json.Unmarshal([]byte(call.Function.Arguments), &args)
	return args.Path
}
