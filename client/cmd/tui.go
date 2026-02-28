package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/alecthomas/chroma/v2"
	"github.com/alecthomas/chroma/v2/formatters"
	"github.com/alecthomas/chroma/v2/lexers"
	"github.com/alecthomas/chroma/v2/styles"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/x/ansi"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/ThatCatDev/tanrenai/client/internal/agent"
	"github.com/ThatCatDev/tanrenai/client/internal/apiclient"
	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/client/internal/tools"
	"github.com/ThatCatDev/tanrenai/client/pkg/api"
)

// ── Styles ──────────────────────────────────────────────────────────────

var (
	borderStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	userPfx       = lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Bold(true).Render(">>>")
	botPfx        = lipgloss.NewStyle().Foreground(lipgloss.Color("5")).Bold(true).Render(" ● ")
	dimStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("241")).Italic(true)
	fileClickStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("39")).Underline(true)
)

// ── Tea messages ────────────────────────────────────────────────────────

type contentDeltaMsg string
type agentContentMsg string // assistant text between iterations (dimmed)
type thinkingMsg struct{}
type thinkingDoneMsg struct{}
type toolCallMsg struct {
	display string
	call    api.ToolCall
}
type toolResultMsg struct{ name, result string }
type iterationMsg struct{ iteration, tools int }
type turnDoneMsg struct {
	result []api.Message
	err    error
}
type streamDoneMsg struct {
	content string
	err     error
}
type fileContentMsg struct {
	path    string
	content string
	err     error
}

type focusTarget int

const (
	focusChat focusTarget = iota
	focusFileViewer
)

type fileViewerState struct {
	viewport viewport.Model
	filePath string
	err      error
}

// ── Shared mutable state (pointer, avoids value-copy issues) ────────────

type tuiShared struct {
	mu         sync.Mutex
	turnCancel context.CancelFunc
	program    *tea.Program
}

func (s *tuiShared) setCancel(fn context.CancelFunc) {
	s.mu.Lock()
	s.turnCancel = fn
	s.mu.Unlock()
}

func (s *tuiShared) cancel() {
	s.mu.Lock()
	if s.turnCancel != nil {
		s.turnCancel()
	}
	s.mu.Unlock()
}

func (s *tuiShared) send(msg tea.Msg) {
	if s.program != nil {
		s.program.Send(msg)
	}
}

// ── TUI model ───────────────────────────────────────────────────────────

type tuiModel struct {
	viewport  viewport.Model
	input     textinput.Model
	width     int
	height    int
	ready     bool
	quitting  bool
	agentMode bool

	// Chat content
	lines       []string
	toolResults map[int]string // line index -> full tool result
	expanded    bool           // Tab toggles full tool output

	// File viewer
	fileViewer    *fileViewerState
	toolCallLines map[int]api.ToolCall // line index -> original tool call
	focus         focusTarget

	// Processing state
	processing  bool
	statusText  string
	ctrlCPending bool // first Ctrl+C warns, second quits
	streaming  strings.Builder

	// Shared state (pointer — safe across value copies)
	shared *tuiShared

	// Dependencies
	client        *apiclient.Client
	modelName     string
	mgr           *chatctx.Manager
	registry      *tools.Registry
	memoryEnabled bool
	maxIterations int
	completeFn    agent.CompletionFunc
	streamFn      agent.StreamingCompletionFunc
}

func newTUIModel(
	client *apiclient.Client,
	modelName string,
	mgr *chatctx.Manager,
	registry *tools.Registry,
	memoryEnabled bool,
	maxIterations int,
	agentMode bool,
	completeFn agent.CompletionFunc,
	streamFn agent.StreamingCompletionFunc,
) tuiModel {
	ti := textinput.New()
	ti.Prompt = " ❯ "
	ti.PromptStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Bold(true)
	ti.Focus()
	ti.CharLimit = 0

	return tuiModel{
		input:         ti,
		agentMode:     agentMode,
		toolResults:   make(map[int]string),
		toolCallLines: make(map[int]api.ToolCall),
		focus:         focusChat,
		shared:        &tuiShared{},
		client:        client,
		modelName:     modelName,
		mgr:           mgr,
		registry:      registry,
		memoryEnabled: memoryEnabled,
		maxIterations: maxIterations,
		completeFn:    completeFn,
		streamFn:      streamFn,
	}
}

func (m tuiModel) Init() tea.Cmd {
	return textinput.Blink
}

// ── Update ──────────────────────────────────────────────────────────────

func (m tuiModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m.recalcLayout(), nil

	case tea.KeyMsg:
		if msg.Type != tea.KeyCtrlC {
			m.ctrlCPending = false
		}
		switch msg.Type {
		case tea.KeyCtrlC:
			if m.processing {
				m.shared.cancel()
				return m, nil
			}
			if m.ctrlCPending {
				m.quitting = true
				return m, tea.Quit
			}
			m.ctrlCPending = true
			m.addLine(dimStyle.Render("  Press Ctrl+C again to quit."))
			m.refreshViewport()
			return m, nil
		case tea.KeyCtrlD:
			m.quitting = true
			return m, tea.Quit
		case tea.KeyEsc:
			if m.fileViewer != nil {
				m.fileViewer = nil
				m.focus = focusChat
				m = m.recalcLayout()
				m.refreshViewport()
				return m, nil
			}
		case tea.KeyTab:
			if m.fileViewer != nil {
				if m.focus == focusChat {
					m.focus = focusFileViewer
				} else {
					m.focus = focusChat
				}
				return m, nil
			}
			m.expanded = !m.expanded
			m.refreshViewport()
			return m, nil
		case tea.KeyUp, tea.KeyDown, tea.KeyPgUp, tea.KeyPgDown:
			if m.fileViewer != nil && m.focus == focusFileViewer {
				var cmd tea.Cmd
				m.fileViewer.viewport, cmd = m.fileViewer.viewport.Update(msg)
				return m, cmd
			}
		case tea.KeyEnter:
			if m.processing {
				return m, nil
			}
			text := strings.TrimSpace(m.input.Value())
			if text == "" {
				return m, nil
			}
			m.input.SetValue("")

			if text == "/quit" || text == "/exit" {
				m.quitting = true
				return m, tea.Quit
			}

			if m.handleSlashCommand(text) {
				m.refreshViewport()
				return m, nil
			}

			m.addLine(fmt.Sprintf(" %s  %s", userPfx, text))
			m.addLine("")
			m.refreshViewport()

			m.processing = true
			m.statusText = "Thinking..."
			m.streaming.Reset()

			if m.agentMode {
				return m, m.startAgentTurn(text)
			}
			return m, m.startChatTurn(text)
		}

	case contentDeltaMsg:
		m.streaming.WriteString(string(msg))
		m.updateStreamingLine()
		m.refreshViewport()
		return m, nil

	case thinkingMsg:
		m.statusText = "Thinking..."
		return m, nil

	case thinkingDoneMsg:
		m.statusText = "Generating..."
		return m, nil

	case toolCallMsg:
		m.statusText = msg.display
		idx := len(m.lines)
		name := msg.call.Function.Name
		if name == "file_read" || name == "file_write" || name == "patch_file" {
			path := extractFilePath(msg.call)
			label := dimStyle.Render("    ↳ "+msg.display+" ") + fileClickStyle.Render("📄 "+path)
			m.addLine(label)
			m.toolCallLines[idx] = msg.call
		} else {
			m.addLine(dimStyle.Render("    ↳ " + msg.display))
		}
		m.refreshViewport()
		return m, nil

	case toolResultMsg:
		preview := strings.TrimSpace(msg.result)
		preview = strings.Join(strings.Fields(preview), " ")
		if len(preview) > 120 {
			preview = preview[:120] + "…"
		}
		idx := len(m.lines)
		m.addLine(dimStyle.Render("      " + preview))
		m.toolResults[idx] = msg.result
		m.refreshViewport()
		return m, nil

	case agentContentMsg:
		text := strings.TrimSpace(string(msg))
		if text != "" {
			for _, line := range strings.Split(text, "\n") {
				m.addLine(dimStyle.Render("    " + line))
			}
			m.refreshViewport()
		}
		return m, nil

	case iterationMsg:
		m.statusText = fmt.Sprintf("iteration %d, %d tools", msg.iteration, msg.tools)
		m.addLine(dimStyle.Render(fmt.Sprintf("  ── iteration %d ──", msg.iteration)))
		m.refreshViewport()
		return m, nil

	case tea.MouseMsg:
		return m.handleMouse(msg)

	case fileContentMsg:
		return m.handleFileContent(msg), nil

	case turnDoneMsg:
		return m.handleTurnDone(msg), nil

	case streamDoneMsg:
		return m.handleStreamDone(msg), nil
	}

	var cmds []tea.Cmd
	var cmd tea.Cmd
	m.viewport, cmd = m.viewport.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}
	m.input, cmd = m.input.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}
	return m, tea.Batch(cmds...)
}

// ── View ────────────────────────────────────────────────────────────────

func (m tuiModel) View() string {
	if !m.ready {
		return "Initializing..."
	}
	if m.quitting {
		return ""
	}

	divider := borderStyle.Render(strings.Repeat("─", m.width))

	// Build the input line with status right-aligned
	inputView := m.input.View()
	inputLine := inputView
	if m.processing && m.statusText != "" {
		status := dimStyle.Render(m.statusText + " ")
		inputW := lipgloss.Width(inputView)
		statusW := lipgloss.Width(status)
		gap := m.width - inputW - statusW
		if gap > 0 {
			inputLine = inputView + strings.Repeat(" ", gap) + status
		} else {
			inputLine = inputView + " " + status
		}
	}

	var chatArea string
	if m.fileViewer != nil {
		chatArea = m.renderSplitView()
	} else {
		chatArea = m.viewport.View()
	}

	var b strings.Builder
	b.WriteString(chatArea)
	b.WriteString("\n")
	b.WriteString(divider)
	b.WriteString("\n")
	b.WriteString(inputLine)
	b.WriteString("\n")
	b.WriteString(divider)

	return b.String()
}

// ── Layout ──────────────────────────────────────────────────────────────

func (m tuiModel) recalcLayout() tuiModel {
	vpHeight := m.height - 3
	if vpHeight < 1 {
		vpHeight = 1
	}

	chatWidth := m.width
	if m.fileViewer != nil {
		chatWidth, _ = m.splitWidths()
	}

	if !m.ready {
		m.viewport = viewport.New(chatWidth, vpHeight)
		m.viewport.MouseWheelEnabled = true
		m.viewport.SetContent(strings.Join(m.lines, "\n"))
		m.ready = true
	} else {
		m.viewport.Width = chatWidth
		m.viewport.Height = vpHeight
	}

	if m.fileViewer != nil {
		_, fvWidth := m.splitWidths()
		// -1 for header line
		fvHeight := vpHeight - 1
		if fvHeight < 1 {
			fvHeight = 1
		}
		m.fileViewer.viewport.Width = fvWidth
		m.fileViewer.viewport.Height = fvHeight
	}

	m.input.Width = m.width - 4
	return m
}

// ── Content management ──────────────────────────────────────────────────

func (m *tuiModel) addLine(line string) {
	m.lines = append(m.lines, line)
}

func (m *tuiModel) updateStreamingLine() {
	content := m.streaming.String()
	formatted := fmt.Sprintf(" %s %s", botPfx, content)
	contentLines := strings.Split(formatted, "\n")

	// Find where streaming started (after last user prefix + blank)
	streamStart := len(m.lines)
	for i := len(m.lines) - 1; i >= 0; i-- {
		if strings.Contains(m.lines[i], ">>>") {
			streamStart = i + 2
			break
		}
	}
	if streamStart > len(m.lines) {
		streamStart = len(m.lines)
	}

	m.lines = append(m.lines[:streamStart], contentLines...)
}

func (m *tuiModel) refreshViewport() {
	if !m.expanded || len(m.toolResults) == 0 {
		m.viewport.SetContent(strings.Join(m.lines, "\n"))
	} else {
		var built []string
		for i, line := range m.lines {
			if full, ok := m.toolResults[i]; ok {
				// Show full result, indented and dimmed
				for _, fline := range strings.Split(strings.TrimRight(full, "\n"), "\n") {
					built = append(built, dimStyle.Render("      "+fline))
				}
			} else {
				built = append(built, line)
			}
		}
		m.viewport.SetContent(strings.Join(built, "\n"))
	}
	m.viewport.GotoBottom()
}

// ── Slash commands ──────────────────────────────────────────────────────

func (m *tuiModel) handleSlashCommand(input string) bool {
	var buf strings.Builder

	switch {
	case input == "/clear":
		m.mgr.Clear()
		m.lines = nil
		m.toolResults = make(map[int]string)
		m.toolCallLines = make(map[int]api.ToolCall)
		m.fileViewer = nil
		m.focus = focusChat
		m.addLine(dimStyle.Render("  History cleared."))
		m.addLine("")
		return true

	case input == "/compact":
		if !m.agentMode {
			m.addLine(dimStyle.Render("  /compact is only available in agent mode."))
			m.addLine("")
			return true
		}
		if m.mgr.NeedsSummary() {
			m.addLine(dimStyle.Render("  [compacting...]"))
			if err := m.mgr.Summarize(context.Background(), chatctx.CompletionFunc(m.completeFn)); err != nil {
				m.addLine(dimStyle.Render(fmt.Sprintf("  Compact failed: %v", err)))
			} else {
				budget := m.mgr.Budget()
				m.addLine(dimStyle.Render(fmt.Sprintf("  Compacted. %d tokens free (%d%%)",
					budget.Available, budget.Available*100/budget.Total)))
			}
		} else {
			m.addLine(dimStyle.Render("  Nothing to compact."))
		}
		m.addLine("")
		return true

	case input == "/help":
		m.addLine(dimStyle.Render("  Commands:"))
		m.addLine(dimStyle.Render("    /clear              Clear conversation history"))
		m.addLine(dimStyle.Render("    /compact            Summarize to free context"))
		m.addLine(dimStyle.Render("    /tokens             Show token budget"))
		m.addLine(dimStyle.Render("    /context add <path> Load file into context"))
		m.addLine(dimStyle.Render("    /context list       Show loaded files"))
		m.addLine(dimStyle.Render("    /context clear      Remove all context files"))
		m.addLine(dimStyle.Render("    /memory             List recent memories"))
		m.addLine(dimStyle.Render("    /memory search <q>  Search memories"))
		m.addLine(dimStyle.Render("    /memory forget <id> Delete a memory"))
		m.addLine(dimStyle.Render("    /memory clear       Clear all memories"))
		m.addLine(dimStyle.Render("    /quit, /exit        Exit"))
		m.addLine("")
		return true
	}

	if handleREPLCommand(&buf, input, m.mgr, m.client, m.memoryEnabled) {
		for _, line := range strings.Split(buf.String(), "\n") {
			if line != "" {
				m.addLine(dimStyle.Render("  " + line))
			}
		}
		m.addLine("")
		return true
	}

	return false
}

// ── Agent turn ──────────────────────────────────────────────────────────

func (m *tuiModel) startAgentTurn(input string) tea.Cmd {
	shared := m.shared
	mgr := m.mgr
	client := m.client
	registry := m.registry
	streamFn := m.streamFn
	completeFn := m.completeFn
	memoryEnabled := m.memoryEnabled
	maxIterations := m.maxIterations

	return func() tea.Msg {
		mgr.Append(api.Message{Role: "user", Content: input})

		if memoryEnabled {
			results, err := client.MemorySearch(context.Background(), input, 3)
			if err == nil && len(results.Results) > 0 {
				var memMsgs []api.Message
				for _, r := range results.Results {
					userMsg := truncate(r.Entry.UserMsg, 200)
					assistMsg := truncate(r.Entry.AssistMsg, 500)
					memContent := fmt.Sprintf("[Memory from %s] User asked: %s\nAssistant replied: %s",
						r.Entry.Timestamp.Format("2006-01-02"), userMsg, assistMsg)
					memMsgs = append(memMsgs, api.Message{Role: "system", Content: memContent})
				}
				mgr.SetMemories(memMsgs)
			} else {
				mgr.ClearMemories()
			}
		}

		if mgr.NeedsSummary() {
			_ = mgr.Summarize(context.Background(), chatctx.CompletionFunc(completeFn))
		}

		windowedMsgs := mgr.Messages()

		turnCtx, turnCancel := context.WithCancel(context.Background())
		shared.setCancel(turnCancel)

		toolCount := 0
		var contentBuf strings.Builder

		flushContent := func() {
			if contentBuf.Len() > 0 {
				shared.send(agentContentMsg(contentBuf.String()))
				contentBuf.Reset()
			}
		}

		cfg := agent.StreamingConfig{
			Config: agent.Config{
				MaxIterations: maxIterations,
				Tools:         registry,
				Hooks: agent.Hooks{
					OnToolCall: func(call api.ToolCall) {
						flushContent()
						toolCount++
						shared.send(toolCallMsg{
							display: fmt.Sprintf("%d tools | %s", toolCount, call.Function.Name),
							call:    call,
						})
					},
					OnToolResult: func(call api.ToolCall, result string) {
						shared.send(toolResultMsg{name: call.Function.Name, result: result})
					},
					OnAssistantMessage: func(content string) {
						// Content already flushed via OnContentDelta; ignore.
					},
				},
			},
			OnIterationStart: func(iteration, maxIter int) {
				flushContent()
				shared.send(iterationMsg{iteration, toolCount})
			},
			OnThinking: func() {
				shared.send(thinkingMsg{})
			},
			OnThinkingDone: func() {
				shared.send(thinkingDoneMsg{})
			},
			OnContentDelta: func(delta string) {
				contentBuf.WriteString(delta)
			},
		}

		result, err := agent.RunStreaming(turnCtx, streamFn, windowedMsgs, cfg)
		flushContent()
		turnCancel()
		shared.setCancel(nil)

		return turnDoneMsg{result: result, err: err}
	}
}

func (m tuiModel) handleTurnDone(msg turnDoneMsg) tuiModel {
	m.processing = false
	m.statusText = ""

	windowedMsgs := m.mgr.Messages()

	if msg.err != nil {
		if strings.Contains(msg.err.Error(), "context canceled") {
			m.addLine(dimStyle.Render("  [interrupted]"))
		} else {
			m.addLine(dimStyle.Render(fmt.Sprintf("  Error: %v", msg.err)))
		}
	}

	if len(msg.result) > len(windowedMsgs) {
		newMsgs := msg.result[len(windowedMsgs):]
		m.mgr.AppendMany(newMsgs)

		var finalContent string
		for i := len(newMsgs) - 1; i >= 0; i-- {
			if newMsgs[i].Role == "assistant" && newMsgs[i].Content != "" {
				finalContent = newMsgs[i].Content
				break
			}
		}
		if finalContent != "" {
			formatted := fmt.Sprintf(" %s %s", botPfx, finalContent)
			for _, line := range strings.Split(formatted, "\n") {
				m.addLine(line)
			}
		}

		if m.memoryEnabled {
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
				client := m.client
				go func() {
					_, _ = client.MemoryStore(context.Background(), userInput, assistContent)
				}()
			}
		}
	}

	m.addLine("")
	m.refreshViewport()
	return m
}

// ── Chat turn (non-agent, streaming) ────────────────────────────────────

func (m *tuiModel) startChatTurn(input string) tea.Cmd {
	shared := m.shared
	mgr := m.mgr
	client := m.client
	modelName := m.modelName

	return func() tea.Msg {
		mgr.Append(api.Message{Role: "user", Content: input})
		windowedMsgs := mgr.Messages()

		req := &api.ChatCompletionRequest{
			Model:    modelName,
			Messages: windowedMsgs,
			Stream:   true,
		}

		turnCtx, turnCancel := context.WithCancel(context.Background())
		shared.setCancel(turnCancel)

		events, err := client.StreamCompletion(turnCtx, req)
		if err != nil {
			turnCancel()
			shared.setCancel(nil)
			return streamDoneMsg{err: err}
		}

		var full strings.Builder
		for ev := range events {
			if ev.Err != nil {
				turnCancel()
				shared.setCancel(nil)
				return streamDoneMsg{content: full.String(), err: ev.Err}
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
					shared.send(contentDeltaMsg(choice.Delta.Content))
				}
			}
		}
		turnCancel()
		shared.setCancel(nil)

		return streamDoneMsg{content: full.String()}
	}
}

func (m tuiModel) handleStreamDone(msg streamDoneMsg) tuiModel {
	m.processing = false
	m.statusText = ""

	if msg.err != nil {
		if strings.Contains(msg.err.Error(), "context canceled") {
			m.addLine(dimStyle.Render("  [interrupted]"))
		} else {
			m.addLine(dimStyle.Render(fmt.Sprintf("  Error: %v", msg.err)))
		}
	}

	if msg.content != "" {
		m.mgr.Append(api.Message{Role: "assistant", Content: msg.content})
	}

	m.streaming.Reset()
	m.addLine("")
	m.refreshViewport()
	return m
}

// ── Syntax highlighting ─────────────────────────────────────────────────

// highlightContent applies chroma syntax highlighting to file content.
// It detects the language from the filename, falls back to content analysis,
// then plain text. Returns ANSI-escaped string for terminal256.
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

// addLineNumbers prepends dimmed, right-aligned line numbers to each line.
func addLineNumbers(content string) string {
	lines := strings.Split(content, "\n")
	width := len(fmt.Sprintf("%d", len(lines)))
	numStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("240"))

	var b strings.Builder
	for i, line := range lines {
		num := fmt.Sprintf("%*d", width, i+1)
		b.WriteString(numStyle.Render(num))
		b.WriteString(" ")
		b.WriteString(line)
		if i < len(lines)-1 {
			b.WriteString("\n")
		}
	}
	return b.String()
}

// ── File viewer ─────────────────────────────────────────────────────────

func extractFilePath(call api.ToolCall) string {
	var args struct {
		Path string `json:"path"`
	}
	_ = json.Unmarshal([]byte(call.Function.Arguments), &args)
	return args.Path
}

// splitWidths returns (chatWidth, fileViewerWidth) for the split layout.
// 50/50 split with 1 col for the divider.
func (m tuiModel) splitWidths() (int, int) {
	divider := 1
	available := m.width - divider
	chatW := available / 2
	fvW := available - chatW
	return chatW, fvW
}

// displayLineToLogicalLine maps a display line (accounting for expanded tool
// results) back to the logical m.lines index.
func (m tuiModel) displayLineToLogicalLine(displayLine int) int {
	if !m.expanded || len(m.toolResults) == 0 {
		return displayLine
	}
	cur := 0
	for i, line := range m.lines {
		if full, ok := m.toolResults[i]; ok {
			expandedCount := len(strings.Split(strings.TrimRight(full, "\n"), "\n"))
			if displayLine < cur+expandedCount {
				return i
			}
			cur += expandedCount
		} else {
			if cur == displayLine {
				return i
			}
			cur++
			_ = line
		}
	}
	return -1
}

func (m tuiModel) handleMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.MouseLeft:
		chatWidth := m.width
		if m.fileViewer != nil {
			chatWidth, _ = m.splitWidths()
		}

		if m.fileViewer != nil && msg.X > chatWidth {
			m.focus = focusFileViewer
			return m, nil
		}

		if msg.X <= chatWidth {
			// Click in chat area — check if it's a tool call line
			displayLine := m.viewport.YOffset + msg.Y
			logicalLine := m.displayLineToLogicalLine(displayLine)
			if logicalLine >= 0 {
				if call, ok := m.toolCallLines[logicalLine]; ok {
					path := extractFilePath(call)
					if path != "" {
						m.focus = focusFileViewer
						return m, openFileViewer(path)
					}
				}
			}
			m.focus = focusChat
		}

	case tea.MouseWheelUp, tea.MouseWheelDown:
		chatWidth := m.width
		if m.fileViewer != nil {
			chatWidth, _ = m.splitWidths()
		}

		if m.fileViewer != nil && msg.X > chatWidth {
			var cmd tea.Cmd
			m.fileViewer.viewport, cmd = m.fileViewer.viewport.Update(msg)
			return m, cmd
		}
		var cmd tea.Cmd
		m.viewport, cmd = m.viewport.Update(msg)
		return m, cmd
	}

	return m, nil
}

func openFileViewer(path string) tea.Cmd {
	return func() tea.Msg {
		const maxSize = 64 * 1024
		data, err := os.ReadFile(path)
		if err != nil {
			return fileContentMsg{path: path, err: err}
		}
		content := string(data)
		if len(content) > maxSize {
			content = content[:maxSize] + "\n... (truncated at 64KB)"
		}
		return fileContentMsg{path: path, content: content}
	}
}

func (m tuiModel) handleFileContent(msg fileContentMsg) tuiModel {
	vpHeight := m.height - 3
	if vpHeight < 1 {
		vpHeight = 1
	}
	// Header takes 1 line
	fvHeight := vpHeight - 1
	if fvHeight < 1 {
		fvHeight = 1
	}

	_, fvWidth := m.splitWidths()

	fv := &fileViewerState{
		filePath: msg.path,
		viewport: viewport.New(fvWidth, fvHeight),
	}
	fv.viewport.MouseWheelEnabled = true

	if msg.err != nil {
		fv.err = msg.err
		fv.viewport.SetContent(fmt.Sprintf("Error: %v", msg.err))
	} else {
		fv.viewport.SetContent(addLineNumbers(highlightContent(msg.path, msg.content)))
	}

	m.fileViewer = fv
	m.focus = focusFileViewer
	savedOffset := m.viewport.YOffset
	m = m.recalcLayout()
	m.refreshViewport()
	m.viewport.SetYOffset(savedOffset)
	return m
}

func (m tuiModel) renderSplitView() string {
	if m.fileViewer == nil {
		return m.viewport.View()
	}

	chatW, fvW := m.splitWidths()
	vpHeight := m.viewport.Height

	// ── File viewer header ──
	headerStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("12")).Bold(true)
	path := m.fileViewer.filePath
	scrollPct := int(m.fileViewer.viewport.ScrollPercent() * 100)
	hint := fmt.Sprintf(" %d%% [Esc close | Tab focus]", scrollPct)
	maxPathW := fvW - ansi.StringWidth(hint)
	if maxPathW < 10 {
		maxPathW = 10
	}
	if ansi.StringWidth(path) > maxPathW {
		path = "…" + path[len(path)-maxPathW+1:]
	}
	header := headerStyle.Render(path + hint)
	// ── Divider ──
	divColor := "240"
	if m.focus == focusFileViewer {
		divColor = "12"
	}
	div := lipgloss.NewStyle().Foreground(lipgloss.Color(divColor)).Render("│")

	// ── Split lines from each pane ──
	chatLines := strings.Split(m.viewport.View(), "\n")
	fvLines := strings.Split(m.fileViewer.viewport.View(), "\n")

	// Column where divider sits (1-based for ANSI CHA escape)
	divCol := chatW + 1

	// ── Compose line by line ──
	var b strings.Builder
	for i := 0; i < vpHeight; i++ {
		// Chat content (viewport already truncates to chatW)
		if i < len(chatLines) {
			b.WriteString(chatLines[i])
		}
		b.WriteString("\x1b[0m") // reset any ANSI bleed from chat

		// Force cursor to exact column — terminal handles placement,
		// bypasses all width-measurement disagreements.
		fmt.Fprintf(&b, "\x1b[%dG", divCol)
		b.WriteString(div)

		// File viewer column: header on row 0, viewport below
		fv := ""
		if i == 0 {
			fv = header
		} else if i-1 < len(fvLines) {
			fv = fvLines[i-1]
		}
		b.WriteString(ansi.Truncate(fv, fvW, ""))

		if i < vpHeight-1 {
			b.WriteString("\n")
		}
	}
	return b.String()
}
