package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/alecthomas/chroma/v2"
	"github.com/alecthomas/chroma/v2/formatters"
	"github.com/alecthomas/chroma/v2/lexers"
	"github.com/alecthomas/chroma/v2/styles"
	"github.com/charmbracelet/glamour"
	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"

	"github.com/ThatCatDev/tanrenai/client/internal/agent"
	"github.com/ThatCatDev/tanrenai/client/internal/apiclient"
	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/client/internal/tools"
	"github.com/ThatCatDev/tanrenai/client/pkg/api"
)

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
	filePath      string               // "" = no file viewer open
	focus         focusTarget
	processing    bool
	ctrlCPending  bool
	streaming     strings.Builder
	turnCancel    context.CancelFunc

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

	// Dependencies (immutable after construction)
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

func newTuiApp(
	client *apiclient.Client,
	modelName string,
	mgr *chatctx.Manager,
	registry *tools.Registry,
	memoryEnabled bool,
	maxIterations int,
	agentMode bool,
	completeFn agent.CompletionFunc,
	streamFn agent.StreamingCompletionFunc,
) *tuiApp {
	t := &tuiApp{
		toolResults:   make(map[int]string),
		toolCallLines: make(map[int]api.ToolCall),
		focus:         focusChat,
		client:        client,
		modelName:     modelName,
		mgr:           mgr,
		registry:      registry,
		memoryEnabled: memoryEnabled,
		maxIterations: maxIterations,
		agentMode:     agentMode,
		completeFn:    completeFn,
		streamFn:      streamFn,
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
		SetFieldBackgroundColor(tcell.ColorDefault)
	t.inputField.SetBorder(false)

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
	return t.app.SetRoot(t.rootFlex, true).EnableMouse(true).Run()
}

// ── Input Capture ──────────────────────────────────────────────────────

func (t *tuiApp) setupInputCapture() {
	t.app.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		if event.Key() != tcell.KeyCtrlC {
			t.ctrlCPending = false
		}

		switch event.Key() {
		case tcell.KeyCtrlC:
			if t.processing {
				t.mu.Lock()
				if t.turnCancel != nil {
					t.turnCancel()
				}
				t.mu.Unlock()
				return nil
			}
			if t.ctrlCPending {
				t.app.Stop()
				return nil
			}
			t.ctrlCPending = true
			t.addLine("[gray::-]  Press Ctrl+C again to quit.[-:-:-]")
			t.refreshChatView()
			return nil

		case tcell.KeyCtrlD:
			t.app.Stop()
			return nil

		case tcell.KeyEscape:
			if t.filePath != "" {
				t.closeFileViewer()
				return nil
			}

		case tcell.KeyTab:
			if t.filePath != "" {
				if t.focus == focusChat {
					t.focus = focusFileViewer
				} else {
					t.focus = focusChat
				}
				t.rebuildFileViewer()
				return nil
			}
			t.expanded = !t.expanded
			t.refreshChatView()
			return nil

		case tcell.KeyUp:
			t.scrollFocusedPane(-1)
			return nil
		case tcell.KeyDown:
			t.scrollFocusedPane(1)
			return nil
		case tcell.KeyPgUp:
			t.scrollFocusedPane(-10)
			return nil
		case tcell.KeyPgDn:
			t.scrollFocusedPane(10)
			return nil

		case tcell.KeyEnter:
			if t.processing {
				return nil
			}
			text := strings.TrimSpace(t.inputField.GetText())
			if text == "" {
				return nil
			}
			t.inputField.SetText("")
			t.handleEnter(text)
			return nil
		}

		return event
	})
}

func (t *tuiApp) scrollFocusedPane(delta int) {
	var tv *tview.TextView
	if t.filePath != "" && t.focus == focusFileViewer {
		tv = t.fileView
	} else {
		tv = t.chatView
	}
	if tv == nil {
		return
	}
	row, col := tv.GetScrollOffset()
	newRow := row + delta
	if newRow < 0 {
		newRow = 0
	}
	tv.ScrollTo(newRow, col)
}

// ── Mouse Capture ──────────────────────────────────────────────────────

func (t *tuiApp) setupMouseCapture() {
	t.app.SetMouseCapture(func(event *tcell.EventMouse, action tview.MouseAction) (*tcell.EventMouse, tview.MouseAction) {
		mx, my := event.Position()

		switch action {
		case tview.MouseScrollUp, tview.MouseScrollDown:
			delta := 3
			if action == tview.MouseScrollUp {
				delta = -3
			}

			// Determine which pane based on X coordinate
			if t.filePath != "" && t.fileView != nil {
				fx, _, fw, _ := t.fileView.GetRect()
				if mx >= fx && mx < fx+fw {
					row, col := t.fileView.GetScrollOffset()
					newRow := row + delta
					if newRow < 0 {
						newRow = 0
					}
					t.fileView.ScrollTo(newRow, col)
					return nil, 0
				}
			}

			cx, _, cw, _ := t.chatView.GetRect()
			if mx >= cx && mx < cx+cw {
				row, col := t.chatView.GetScrollOffset()
				newRow := row + delta
				if newRow < 0 {
					newRow = 0
				}
				t.chatView.ScrollTo(newRow, col)
				return nil, 0
			}

		case tview.MouseLeftClick:
			// Check if click is in chat area for tool call click-to-open
			cx, cy, cw, ch := t.chatView.GetRect()
			if mx >= cx && mx < cx+cw && my >= cy && my < cy+ch {
				row, _ := t.chatView.GetScrollOffset()
				displayLine := row + (my - cy)
				logicalLine := t.displayLineToLogicalLine(displayLine)
				if logicalLine >= 0 {
					if call, ok := t.toolCallLines[logicalLine]; ok {
						path := extractFilePath(call)
						if path != "" {
							t.focus = focusFileViewer
							go t.loadFileViewer(path)
							return nil, 0
						}
					}
				}
				t.focus = focusChat
			}

			// Click in file viewer area
			if t.filePath != "" && t.fileView != nil {
				fx, fy, fw, fh := t.fileView.GetRect()
				if mx >= fx && mx < fx+fw && my >= fy && my < fy+fh {
					t.focus = focusFileViewer
				}
			}
		}

		return event, action
	})
}

// ── Enter Handler ──────────────────────────────────────────────────────

func (t *tuiApp) handleEnter(text string) {
	if text == "/quit" || text == "/exit" {
		t.app.Stop()
		return
	}

	if t.handleSlashCommand(text) {
		t.refreshChatView()
		return
	}

	t.addLine(fmt.Sprintf(" [blue::b]>>>[white] %s", tview.Escape(text)))
	t.addLine("")
	t.refreshChatView()

	t.processing = true
	t.statusText = "Thinking..."
	t.currentIterTokens = 0
	t.currentIterOutput = 0
	t.lastInputTokens = 0
	t.lastOutputTokens = 0
	t.startProgressTicker()
	t.iterStartTime = time.Now()
	t.estimatedDur = 0
	t.updateStatusBar()
	t.streaming.Reset()

	if t.agentMode {
		go t.startAgentTurn(text)
	} else {
		go t.startChatTurn(text)
	}
}

// ── Slash Commands ──────────────────────────────────────────────────────

func (t *tuiApp) handleSlashCommand(input string) bool {
	switch {
	case input == "/clear":
		t.mgr.Clear()
		t.lines = nil
		t.toolResults = make(map[int]string)
		t.toolCallLines = make(map[int]api.ToolCall)
		t.closeFileViewer()
		t.addLine("[gray::-]  History cleared.[-:-:-]")
		t.addLine("")
		return true

	case input == "/compact":
		if !t.agentMode {
			t.addLine("[gray::-]  /compact is only available in agent mode.[-:-:-]")
			t.addLine("")
			return true
		}
		if t.mgr.NeedsSummary() {
			t.addLine("[gray::-]  [compacting...][-:-:-]")
			if err := t.mgr.Summarize(context.Background(), chatctx.CompletionFunc(t.completeFn)); err != nil {
				t.addLine(fmt.Sprintf("[gray::-]  Compact failed: %v[-:-:-]", err))
			} else {
				budget := t.mgr.Budget()
				t.addLine(fmt.Sprintf("[gray::-]  Compacted. %d tokens free (%d%%)[-:-:-]",
					budget.Available, budget.Available*100/budget.Total))
			}
		} else {
			t.addLine("[gray::-]  Nothing to compact.[-:-:-]")
		}
		t.addLine("")
		return true

	case input == "/help":
		t.addLine("[gray::-]  Commands:[-:-:-]")
		t.addLine("[gray::-]    /clear              Clear conversation history[-:-:-]")
		t.addLine("[gray::-]    /compact            Summarize to free context[-:-:-]")
		t.addLine("[gray::-]    /tokens             Show token budget[-:-:-]")
		t.addLine("[gray::-]    /context add <path> Load file into context[-:-:-]")
		t.addLine("[gray::-]    /context list       Show loaded files[-:-:-]")
		t.addLine("[gray::-]    /context clear      Remove all context files[-:-:-]")
		t.addLine("[gray::-]    /memory             List recent memories[-:-:-]")
		t.addLine("[gray::-]    /memory search <q>  Search memories[-:-:-]")
		t.addLine("[gray::-]    /memory forget <id> Delete a memory[-:-:-]")
		t.addLine("[gray::-]    /memory clear       Clear all memories[-:-:-]")
		t.addLine("[gray::-]    /quit, /exit        Exit[-:-:-]")
		t.addLine("")
		return true
	}

	var buf strings.Builder
	if handleREPLCommand(&buf, input, t.mgr, t.client, t.memoryEnabled) {
		for _, line := range strings.Split(buf.String(), "\n") {
			if line != "" {
				t.addLine("[gray::-]  " + tview.Escape(line) + "[-:-:-]")
			}
		}
		t.addLine("")
		return true
	}

	return false
}

// ── Chat Turn (non-agent, streaming) ────────────────────────────────────

func (t *tuiApp) startChatTurn(input string) {
	t.mgr.Append(api.Message{Role: "user", Content: input})
	windowedMsgs := t.mgr.Messages()

	// Estimate input tokens
	inputTokens := t.mgr.Estimator().EstimateMessages(windowedMsgs)
	t.currentIterTokens = inputTokens
	t.lastInputTokens = inputTokens
	t.currentIterOutput = 0
	t.estimatedDur = t.predictDuration(inputTokens)
	t.app.QueueUpdateDraw(func() { t.updateStatusBar() })

	req := &api.ChatCompletionRequest{
		Model:    t.modelName,
		Messages: windowedMsgs,
		Stream:   true,
	}

	turnCtx, turnCancel := context.WithCancel(context.Background())
	t.mu.Lock()
	t.turnCancel = turnCancel
	t.mu.Unlock()

	events, err := t.client.StreamCompletion(turnCtx, req)
	if err != nil {
		turnCancel()
		t.mu.Lock()
		t.turnCancel = nil
		t.mu.Unlock()
		t.app.QueueUpdateDraw(func() {
			t.handleStreamDone("", err)
		})
		return
	}

	var full strings.Builder
	for ev := range events {
		if ev.Err != nil {
			turnCancel()
			t.mu.Lock()
			t.turnCancel = nil
			t.mu.Unlock()
			content := full.String()
			t.app.QueueUpdateDraw(func() {
				t.handleStreamDone(content, ev.Err)
			})
			return
		}
		if ev.Done {
			break
		}
		if ev.Chunk == nil {
			continue
		}
		for _, choice := range ev.Chunk.Choices {
			if choice.Delta.Content != "" {
				full.WriteString(choice.Delta.Content)
				t.currentIterOutput += len(choice.Delta.Content)
				t.app.QueueUpdateDraw(func() {
					t.streaming.WriteString(choice.Delta.Content)
					t.updateStreamingLine()
					t.refreshChatView()
				})
			}
		}
	}
	turnCancel()
	t.mu.Lock()
	t.turnCancel = nil
	t.mu.Unlock()

	content := full.String()
	t.app.QueueUpdateDraw(func() {
		t.handleStreamDone(content, nil)
	})
}

func (t *tuiApp) handleStreamDone(content string, err error) {
	t.recordIterationEnd()
	t.stopProgressTicker()
	t.processing = false
	t.statusText = ""

	if err != nil {
		if strings.Contains(err.Error(), "context canceled") {
			t.addLine("[gray::-]  [interrupted][-:-:-]")
		} else {
			t.addLine(fmt.Sprintf("[gray::-]  Error: %v[-:-:-]", err))
		}
	}

	if content != "" {
		t.mgr.Append(api.Message{Role: "assistant", Content: content})

		// Replace raw streaming lines with rendered markdown
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
		rendered := fmt.Sprintf(" [purple::b] * [-:-:-]%s", t.renderMarkdown(content))
		t.lines = append(t.lines[:streamStart], strings.Split(rendered, "\n")...)
	}

	t.streaming.Reset()
	t.addLine("")
	t.refreshChatView()
	t.updateStatusBar()
}

// ── Agent Turn ──────────────────────────────────────────────────────────

func (t *tuiApp) startAgentTurn(input string) {
	t.mgr.Append(api.Message{Role: "user", Content: input})

	if t.memoryEnabled {
		results, err := t.client.MemorySearch(context.Background(), input, 3)
		if err == nil && len(results.Results) > 0 {
			var memMsgs []api.Message
			for _, r := range results.Results {
				userMsg := truncate(r.Entry.UserMsg, 200)
				assistMsg := truncate(r.Entry.AssistMsg, 500)
				memContent := fmt.Sprintf("[Memory from %s] User asked: %s\nAssistant replied: %s",
					r.Entry.Timestamp.Format("2006-01-02"), userMsg, assistMsg)
				memMsgs = append(memMsgs, api.Message{Role: "system", Content: memContent})
			}
			t.mgr.SetMemories(memMsgs)
		} else {
			t.mgr.ClearMemories()
		}
	}

	if t.mgr.NeedsSummary() {
		_ = t.mgr.Summarize(context.Background(), chatctx.CompletionFunc(t.completeFn))
	}

	windowedMsgs := t.mgr.Messages()

	turnCtx, turnCancel := context.WithCancel(context.Background())
	t.mu.Lock()
	t.turnCancel = turnCancel
	t.mu.Unlock()

	toolCount := 0
	var contentBuf strings.Builder

	flushContent := func() {
		if contentBuf.Len() > 0 {
			text := contentBuf.String()
			contentBuf.Reset()
			t.app.QueueUpdateDraw(func() {
				trimmed := strings.TrimSpace(text)
				if trimmed != "" {
					rendered := t.renderMarkdown(trimmed)
					for _, line := range strings.Split(rendered, "\n") {
						t.addLine("    " + line)
					}
					t.refreshChatView()
				}
			})
		}
	}

	cfg := agent.StreamingConfig{
		Config: agent.Config{
			MaxIterations: t.maxIterations,
			Tools:         t.registry,
			Hooks: agent.Hooks{
				OnToolCall: func(call api.ToolCall) {
					flushContent()
					toolCount++
					t.currentIterOutput += len(call.Function.Name) + len(call.Function.Arguments)
					display := fmt.Sprintf("%d tools | %s", toolCount, call.Function.Name)
					name := call.Function.Name
					t.app.QueueUpdateDraw(func() {
						t.statusText = display
						t.updateStatusBar()
						idx := len(t.lines)
						if name == "file_read" || name == "file_write" || name == "patch_file" {
							path := extractFilePath(call)
							label := "[gray::-]    > " + tview.Escape(display) + " [-:-:-][#00afff::u]" + tview.Escape(path) + "[-:-:-]"
							t.addLine(label)
							t.toolCallLines[idx] = call
						} else {
							t.addLine("[gray::-]    > " + tview.Escape(display) + "[-:-:-]")
						}
						t.refreshChatView()
					})
				},
				OnToolResult: func(call api.ToolCall, result string) {
					t.app.QueueUpdateDraw(func() {
						preview := strings.TrimSpace(result)
						preview = strings.Join(strings.Fields(preview), " ")
						if len(preview) > 120 {
							preview = preview[:120] + "..."
						}
						idx := len(t.lines)
						t.addLine("[gray::-]      " + tview.Escape(preview) + "[-:-:-]")
						t.toolResults[idx] = result
						t.refreshChatView()
					})
				},
				OnAssistantMessage: func(content string) {
					// Content already flushed via OnContentDelta; ignore.
				},
			},
		},
		OnIterationStart: func(iteration, maxIter int, messages []api.Message) {
			flushContent()
			t.app.QueueUpdateDraw(func() {
				// Record previous iteration duration
				t.recordIterationEnd()

				// Estimate input tokens for this iteration
				inputTokens := t.mgr.Estimator().EstimateMessages(messages)
				t.currentIterTokens = inputTokens
				t.lastInputTokens = inputTokens
				t.currentIterOutput = 0

				// Start timing this iteration
				t.statusText = fmt.Sprintf("iteration %d, %d tools", iteration, toolCount)
				t.startProgressTicker()
				t.iterStartTime = time.Now()
				t.estimatedDur = t.predictDuration(inputTokens)
				t.updateStatusBar()
				t.addLine(fmt.Sprintf("[gray::-]  -- iteration %d --[-:-:-]", iteration))
				t.refreshChatView()
			})
		},
		OnThinking: func() {
			t.app.QueueUpdateDraw(func() {
				t.statusText = "Thinking..."
				t.updateStatusBar()
			})
		},
		OnThinkingDone: func() {
			t.app.QueueUpdateDraw(func() {
				t.statusText = "Generating..."
				t.updateStatusBar()
			})
		},
		OnContentDelta: func(delta string) {
			t.currentIterOutput += len(delta)
			contentBuf.WriteString(delta)
		},
	}

	result, err := agent.RunStreaming(turnCtx, t.streamFn, windowedMsgs, cfg)
	flushContent()
	turnCancel()
	t.mu.Lock()
	t.turnCancel = nil
	t.mu.Unlock()

	t.app.QueueUpdateDraw(func() {
		t.handleTurnDone(result, windowedMsgs, err)
	})
}

func (t *tuiApp) handleTurnDone(result, windowedMsgs []api.Message, err error) {
	t.recordIterationEnd()
	t.stopProgressTicker()
	t.processing = false
	t.statusText = ""

	if err != nil {
		if strings.Contains(err.Error(), "context canceled") {
			t.addLine("[gray::-]  [interrupted][-:-:-]")
		} else {
			t.addLine(fmt.Sprintf("[gray::-]  Error: %v[-:-:-]", err))
		}
	}

	if len(result) > len(windowedMsgs) {
		newMsgs := result[len(windowedMsgs):]
		t.mgr.AppendMany(newMsgs)

		var finalContent string
		for i := len(newMsgs) - 1; i >= 0; i-- {
			if newMsgs[i].Role == "assistant" && newMsgs[i].Content != "" {
				finalContent = newMsgs[i].Content
				break
			}
		}
		if finalContent != "" {
			formatted := fmt.Sprintf(" [purple::b] * [-:-:-]%s", t.renderMarkdown(finalContent))
			for _, line := range strings.Split(formatted, "\n") {
				t.addLine(line)
			}
		}

		if t.memoryEnabled {
			var assistContent string
			for _, msg := range newMsgs {
				if msg.Role == "assistant" && msg.Content != "" {
					if assistContent != "" {
						assistContent += "\n"
					}
					assistContent += msg.Content
				}
			}
			if len(assistContent) > 2000 {
				assistContent = assistContent[:2000]
			}
			userInput := ""
			for i := len(newMsgs) - 1; i >= 0; i-- {
				if newMsgs[i].Role == "user" {
					userInput = newMsgs[i].Content
					break
				}
			}
			if assistContent != "" && userInput != "" {
				client := t.client
				go func() {
					_, _ = client.MemoryStore(context.Background(), userInput, assistContent)
				}()
			}
		}
	}

	t.addLine("")
	t.refreshChatView()
	t.updateStatusBar()
}

// ── Content Management ──────────────────────────────────────────────────

func (t *tuiApp) addLine(line string) {
	t.lines = append(t.lines, line)
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

func (t *tuiApp) updateStatusBar() {
	if t.processing && t.statusText != "" {
		tokenInfo := ""
		if t.lastInputTokens > 0 {
			tokenInfo = " | ~" + formatTokenCount(t.lastInputTokens) + " in"
		}
		bar := ""
		if !t.iterStartTime.IsZero() {
			elapsed := time.Since(t.iterStartTime)
			bar = " " + renderProgressBar(elapsed, t.estimatedDur)
		}
		t.statusBar.SetText(" [gray::-]" + tview.Escape(t.statusText+tokenInfo) + "[-:-:-] " + bar)
	} else if t.lastInputTokens > 0 || t.lastOutputTokens > 0 {
		parts := []string{}
		if t.lastInputTokens > 0 {
			parts = append(parts, "~"+formatTokenCount(t.lastInputTokens)+" in")
		}
		if t.lastOutputTokens > 0 {
			parts = append(parts, "~"+formatTokenCount(t.lastOutputTokens)+" out")
		}
		t.statusBar.SetText(" [gray::-]" + strings.Join(parts, " / ") + "[-:-:-]")
	} else {
		t.statusBar.SetText("")
	}
}

func (t *tuiApp) startProgressTicker() {
	t.stopProgressTicker()
	t.progressStop = make(chan struct{})
	t.progressTicker = time.NewTicker(500 * time.Millisecond)
	stop := t.progressStop
	ticker := t.progressTicker
	go func() {
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				t.app.QueueUpdateDraw(func() {
					t.updateStatusBar()
				})
			}
		}
	}()
}

func (t *tuiApp) stopProgressTicker() {
	if t.progressTicker != nil {
		t.progressTicker.Stop()
		t.progressTicker = nil
	}
	if t.progressStop != nil {
		close(t.progressStop)
		t.progressStop = nil
	}
	t.iterStartTime = time.Time{}
}

func (t *tuiApp) recordIterationEnd() {
	if !t.iterStartTime.IsZero() && t.currentIterTokens > 0 {
		t.iterHistory = append(t.iterHistory, iterRecord{
			inputTokens: t.currentIterTokens,
			duration:    time.Since(t.iterStartTime),
		})
	}
	if t.currentIterOutput > 0 {
		t.lastOutputTokens = t.currentIterOutput / 4 // rough char→token
	}
}

func (t *tuiApp) predictDuration(inputTokens int) time.Duration {
	n := len(t.iterHistory)
	if n == 0 {
		return 0
	}
	if n == 1 {
		r := t.iterHistory[0]
		if r.inputTokens > 0 {
			return time.Duration(float64(r.duration) * float64(inputTokens) / float64(r.inputTokens))
		}
		return r.duration
	}
	// 2+ data points: least-squares linear regression
	var sumX, sumY, sumXY, sumX2 float64
	for _, r := range t.iterHistory {
		x := float64(r.inputTokens)
		y := float64(r.duration)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}
	fn := float64(n)
	denom := fn*sumX2 - sumX*sumX
	if denom == 0 {
		return time.Duration(sumY / fn)
	}
	slope := (fn*sumXY - sumX*sumY) / denom
	intercept := (sumY - slope*sumX) / fn
	predicted := slope*float64(inputTokens) + intercept
	if predicted < 0 {
		predicted = 0
	}
	return time.Duration(predicted)
}

func formatTokenCount(n int) string {
	if n >= 1000 {
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	}
	return fmt.Sprintf("%d", n)
}

func renderProgressBar(elapsed, estimated time.Duration) string {
	const barWidth = 20
	if estimated <= 0 {
		// No estimate yet — just show elapsed
		return fmt.Sprintf("[gray::-]%ds[-:-:-]", int(elapsed.Seconds()))
	}
	ratio := float64(elapsed) / float64(estimated)
	if ratio > 1.0 {
		ratio = 1.0
	}
	filled := int(ratio * barWidth)
	if filled > barWidth {
		filled = barWidth
	}
	empty := barWidth - filled

	remaining := estimated - elapsed
	countdown := ""
	if remaining > 0 {
		countdown = fmt.Sprintf("%ds", int(remaining.Seconds()))
	} else {
		over := elapsed - estimated
		countdown = fmt.Sprintf("+%ds", int(over.Seconds()))
	}

	return fmt.Sprintf("[gray::-][[-]%s%s[gray::-]][-:-:-] [gray::-]%s[-:-:-]",
		strings.Repeat("█", filled),
		strings.Repeat("░", empty),
		countdown)
}

// ── File Viewer ─────────────────────────────────────────────────────────

func (t *tuiApp) loadFileViewer(path string) {
	const maxSize = 64 * 1024
	data, err := os.ReadFile(path)
	if err != nil {
		t.app.QueueUpdateDraw(func() {
			t.openFileViewerContent(path, "", err)
		})
		return
	}
	content := string(data)
	if len(content) > maxSize {
		content = content[:maxSize] + "\n... (truncated at 64KB)"
	}
	t.app.QueueUpdateDraw(func() {
		t.openFileViewerContent(path, content, nil)
	})
}

func (t *tuiApp) openFileViewerContent(path, content string, err error) {
	t.filePath = path
	t.focus = focusFileViewer

	// Create file header
	t.fileHeader = tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(false)
	t.fileHeader.SetBorder(false)
	t.fileHeader.SetText(fmt.Sprintf("[blue::b]%s[-:-:-] [gray::-]Esc close | Tab focus[-:-:-]",
		tview.Escape(path)))

	// Create file view
	t.fileView = tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(true).
		SetWordWrap(false)
	t.fileView.SetBorder(false)

	if err != nil {
		t.fileView.SetText(fmt.Sprintf("[red::-]Error: %v[-:-:-]", err))
	} else {
		highlighted := highlightContent(path, content)
		numbered := addLineNumbers(highlighted)
		// Convert ANSI to tview color tags
		t.fileView.SetText(tview.TranslateANSI(numbered))
	}

	// Build file panel (header + file content)
	t.filePanel = tview.NewFlex().SetDirection(tview.FlexRow)
	t.filePanel.AddItem(t.fileHeader, 1, 0, false)
	t.filePanel.AddItem(t.fileView, 0, 1, false)

	// Rebuild chatArea with split
	t.chatArea.Clear()
	t.chatArea.AddItem(t.chatView, 0, 1, false)
	t.chatArea.AddItem(newVDivider(t.focus == focusFileViewer), 1, 0, false)
	t.chatArea.AddItem(t.filePanel, 0, 1, false)
}

func (t *tuiApp) rebuildFileViewer() {
	if t.filePath == "" {
		return
	}
	// Rebuild chatArea to update divider focus color
	t.chatArea.Clear()
	t.chatArea.AddItem(t.chatView, 0, 1, false)
	t.chatArea.AddItem(newVDivider(t.focus == focusFileViewer), 1, 0, false)
	t.chatArea.AddItem(t.filePanel, 0, 1, false)
}

func (t *tuiApp) closeFileViewer() {
	t.filePath = ""
	t.focus = focusChat
	t.filePanel = nil
	t.fileHeader = nil
	t.fileView = nil

	t.chatArea.Clear()
	t.chatArea.AddItem(t.chatView, 0, 1, false)
}

// ── Display Line Mapping ────────────────────────────────────────────────

func (t *tuiApp) displayLineToLogicalLine(displayLine int) int {
	_, _, cw, _ := t.chatView.GetRect()
	if cw <= 0 {
		cw = 80
	}

	cur := 0
	for i, line := range t.lines {
		if t.expanded && len(t.toolResults) > 0 {
			if full, ok := t.toolResults[i]; ok {
				for _, fline := range strings.Split(strings.TrimRight(full, "\n"), "\n") {
					escaped := "[gray::-]      " + tview.Escape(fline) + "[-:-:-]"
					rows := wrappedLineRows(escaped, cw)
					if displayLine < cur+rows {
						return i
					}
					cur += rows
				}
				continue
			}
		}
		rows := wrappedLineRows(line, cw)
		if displayLine < cur+rows {
			return i
		}
		cur += rows
	}
	return -1
}

// wrappedLineRows estimates how many display rows a logical line occupies
// when word-wrapped to the given view width.
func wrappedLineRows(taggedLine string, viewWidth int) int {
	w := tview.TaggedStringWidth(taggedLine)
	if w <= viewWidth {
		return 1
	}
	return (w + viewWidth - 1) / viewWidth
}

// ── Syntax Highlighting ─────────────────────────────────────────────────

func highlightContent(path, content string) string {
	lexer := lexers.Match(path)
	if lexer == nil {
		lexer = lexers.Analyse(content)
	}
	if lexer == nil {
		lexer = lexers.Fallback
	}
	lexer = chroma.Coalesce(lexer)

	style := styles.Get("monokai")
	formatter := formatters.Get("terminal256")
	if formatter == nil {
		return content
	}

	iterator, err := lexer.Tokenise(nil, content)
	if err != nil {
		return content
	}

	var buf bytes.Buffer
	if err := formatter.Format(&buf, style, iterator); err != nil {
		return content
	}
	return buf.String()
}

func addLineNumbers(content string) string {
	lines := strings.Split(content, "\n")
	width := len(fmt.Sprintf("%d", len(lines)))

	var b strings.Builder
	for i, line := range lines {
		num := fmt.Sprintf("%*d", width, i+1)
		// ANSI gray for line numbers
		b.WriteString("\033[38;5;240m")
		b.WriteString(num)
		b.WriteString("\033[0m ")
		b.WriteString(line)
		if i < len(lines)-1 {
			b.WriteString("\n")
		}
	}
	return b.String()
}

// ── Markdown Rendering ──────────────────────────────────────────────────

var ansiSGR = regexp.MustCompile("\x1b\\[([0-9;:]*)m")
var tviewTag = regexp.MustCompile(`\[([^\[\]]*):([^\[\]]*):([^\[\]]*)\]`)

// stripTviewUnderline removes 'u' (underline) from the attributes field of
// tview color tags like [fg:bg:attrs]. This is a safety net in case ANSI
// underline sequences slip through stripANSIUnderline.
func stripTviewUnderline(s string) string {
	return tviewTag.ReplaceAllStringFunc(s, func(tag string) string {
		inner := tag[1 : len(tag)-1]
		parts := strings.SplitN(inner, ":", 3)
		if len(parts) < 3 {
			return tag
		}
		attrs := parts[2]
		if !strings.ContainsRune(attrs, 'u') {
			return tag
		}
		newAttrs := strings.ReplaceAll(attrs, "u", "")
		if newAttrs == "" {
			newAttrs = "-"
		}
		return "[" + parts[0] + ":" + parts[1] + ":" + newAttrs + "]"
	})
}

// stripANSIUnderline removes underline (4, 4:N) and no-underline (24) parameters
// from ANSI SGR sequences, preserving all other attributes and colors.
func stripANSIUnderline(s string) string {
	return ansiSGR.ReplaceAllStringFunc(s, func(seq string) string {
		inner := seq[2 : len(seq)-1] // between \x1b[ and m
		if inner == "" {
			return seq
		}
		params := strings.Split(inner, ";")
		var out []string
		for i := 0; i < len(params); i++ {
			p := params[i]
			// Strip underline: 4, 4:N (colon sub-params), 24
			if p == "4" || p == "24" || strings.HasPrefix(p, "4:") {
				continue
			}
			// 38;5;N / 48;5;N: extended color — consume all three parts
			if (p == "38" || p == "48") && i+2 < len(params) && params[i+1] == "5" {
				out = append(out, p, params[i+1], params[i+2])
				i += 2
				continue
			}
			// 38;2;R;G;B / 48;2;R;G;B: true color — consume all five parts
			if (p == "38" || p == "48") && i+4 < len(params) && params[i+1] == "2" {
				out = append(out, p, params[i+1], params[i+2], params[i+3], params[i+4])
				i += 4
				continue
			}
			// 38:5:N / 48:5:N / 38:2:R:G:B / 48:2:R:G:B: colon-style extended color
			if strings.HasPrefix(p, "38:") || strings.HasPrefix(p, "48:") {
				out = append(out, p)
				continue
			}
			out = append(out, p)
		}
		if len(out) == 0 {
			return ""
		}
		return "\x1b[" + strings.Join(out, ";") + "m"
	})
}

func (t *tuiApp) renderMarkdown(content string) string {
	r, err := glamour.NewTermRenderer(
		glamour.WithStandardStyle("dark"),
		glamour.WithWordWrap(0),
	)
	if err != nil {
		return tview.Escape(content)
	}
	out, err := r.Render(content)
	if err != nil {
		return tview.Escape(content)
	}
	out = stripANSIUnderline(out)
	translated := tview.TranslateANSI(strings.TrimRight(out, "\n"))
	return stripTviewUnderline(translated)
}

// ── Helpers ─────────────────────────────────────────────────────────────

func extractFilePath(call api.ToolCall) string {
	var args struct {
		Path string `json:"path"`
	}
	_ = json.Unmarshal([]byte(call.Function.Arguments), &args)
	return args.Path
}
