package chatctx

import (
	"fmt"
	"strings"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

// Config configures the context manager.
type Config struct {
	CtxSize        int // total context window in tokens (default 4096)
	ResponseBudget int // tokens reserved for model output (default 512)
	ToolsBudget    int // tokens reserved for tool definitions in the prompt (0 = none)
}

// BudgetInfo contains token budget breakdown information.
type BudgetInfo struct {
	Total        int // total context window size
	System       int // tokens used by system prompt + context files
	Memory       int // tokens used by injected memories
	History      int // tokens used by history messages in the window
	Summary      int // tokens used by conversation summary
	Available    int // tokens available for new content
	HistoryCount int // number of history messages in the window
	TotalHistory int // total number of history messages (including evicted)
}

// contextFile represents a loaded context file.
type contextFile struct {
	Path    string
	Content string
}

// Manager manages windowed message history with token budget tracking.
// System messages and context files are pinned (never evicted).
// History messages are windowed: oldest messages are dropped when the budget is exceeded.
type Manager struct {
	cfg          Config
	estimator    *TokenEstimator
	systemPrompt string
	contextFiles []contextFile
	history      []api.Message // user/assistant/tool messages
	summary      string        // condensed summary of evicted messages
	memories     []api.Message // injected memory messages from RAG
}

// Estimator returns the token estimator used by this manager.
func (m *Manager) Estimator() *TokenEstimator { return m.estimator }

// NewManager creates a Manager with the given config and estimator.
func NewManager(cfg Config, estimator *TokenEstimator) *Manager {
	if cfg.CtxSize <= 0 {
		cfg.CtxSize = 4096
	}
	if cfg.ResponseBudget <= 0 {
		cfg.ResponseBudget = 512
	}
	return &Manager{
		cfg:       cfg,
		estimator: estimator,
	}
}

// SetSystemPrompt sets the pinned system prompt.
func (m *Manager) SetSystemPrompt(prompt string) {
	m.systemPrompt = prompt
}

// AddContextFile loads a file into the pinned context.
func (m *Manager) AddContextFile(path, content string) {
	m.contextFiles = append(m.contextFiles, contextFile{Path: path, Content: content})
}

// ClearContextFiles removes all context files.
func (m *Manager) ClearContextFiles() {
	m.contextFiles = nil
}

// ContextFiles returns the list of loaded context file paths.
func (m *Manager) ContextFiles() []string {
	paths := make([]string, len(m.contextFiles))
	for i, cf := range m.contextFiles {
		paths[i] = cf.Path
	}
	return paths
}

// SetMemories sets the injected memory messages (from RAG retrieval).
func (m *Manager) SetMemories(msgs []api.Message) {
	m.memories = msgs
}

// ClearMemories removes all injected memories.
func (m *Manager) ClearMemories() {
	m.memories = nil
}

// Append adds a single message to history.
func (m *Manager) Append(msg api.Message) {
	m.history = append(m.history, msg)
}

// AppendMany adds multiple messages to history.
func (m *Manager) AppendMany(msgs []api.Message) {
	m.history = append(m.history, msgs...)
}

// Messages returns the windowed message list suitable for sending to the LLM.
// All system content (prompt, context files, memories, summary) is merged into
// a single system message at position 0 to satisfy models like Qwen 3.5 that
// require exactly one system message at the beginning.
func (m *Manager) Messages() []api.Message {
	systemMsgs := m.buildSystemMessages()
	systemTokens := m.estimator.EstimateMessages(systemMsgs)

	available := m.cfg.CtxSize - systemTokens - m.cfg.ResponseBudget - m.cfg.ToolsBudget
	if available < 0 {
		available = 0
	}

	// Reserve space for memories if present
	if len(m.memories) > 0 {
		memoryTokens := m.estimator.EstimateMessages(m.memories)
		available -= memoryTokens
		if available < 0 {
			available = 0
		}
	}

	// Reserve space for summary if present
	var summaryText string
	if m.summary != "" {
		summaryText = fmt.Sprintf("[Conversation summary] %s", m.summary)
		sm := api.Message{Role: "system", Content: summaryText}
		summaryTokens := m.estimator.EstimateMessages([]api.Message{sm})
		available -= summaryTokens
		if available < 0 {
			available = 0
		}
	}

	// Walk history backwards to find the cutoff
	cutoff := len(m.history)
	used := 0
	for i := len(m.history) - 1; i >= 0; i-- {
		msgTokens := m.estimator.EstimateMessages([]api.Message{m.history[i]})
		if used+msgTokens > available {
			break
		}
		used += msgTokens
		cutoff = i
	}

	// Merge all system content into a single message
	var systemContent string
	if len(systemMsgs) > 0 {
		systemContent = systemMsgs[0].Content
	}

	for _, mem := range m.memories {
		if systemContent != "" {
			systemContent += "\n\n"
		}
		systemContent += mem.Content
	}

	if summaryText != "" {
		if systemContent != "" {
			systemContent += "\n\n"
		}
		systemContent += summaryText
	}

	// Build result: [single system msg] + [windowed history]
	var result []api.Message
	if systemContent != "" {
		result = append(result, api.Message{Role: "system", Content: systemContent})
	}
	result = append(result, m.history[cutoff:]...)

	return result
}

// buildSystemMessages constructs the base system message (prompt + context files
// merged into a single message). Returns a slice of 0 or 1 messages.
func (m *Manager) buildSystemMessages() []api.Message {
	var parts []string

	if m.systemPrompt != "" {
		parts = append(parts, m.systemPrompt)
	}

	for _, cf := range m.contextFiles {
		parts = append(parts, fmt.Sprintf("[File: %s]\n%s", cf.Path, cf.Content))
	}

	if len(parts) == 0 {
		return nil
	}

	return []api.Message{{
		Role:    "system",
		Content: strings.Join(parts, "\n\n"),
	}}
}

// NeedsSummary returns true if the history has messages that won't fit in the window
// and could benefit from summarization.
func (m *Manager) NeedsSummary() bool {
	systemMsgs := m.buildSystemMessages()
	systemTokens := m.estimator.EstimateMessages(systemMsgs)
	available := m.cfg.CtxSize - systemTokens - m.cfg.ResponseBudget - m.cfg.ToolsBudget

	if len(m.memories) > 0 {
		available -= m.estimator.EstimateMessages(m.memories)
	}

	if m.summary != "" {
		sm := api.Message{
			Role:    "system",
			Content: fmt.Sprintf("[Conversation summary] %s", m.summary),
		}
		available -= m.estimator.EstimateMessages([]api.Message{sm})
	}

	totalHistory := m.estimator.EstimateMessages(m.history)
	return totalHistory > available && len(m.history) > 0
}

// Clear resets history and summary, keeping system prompt and context files.
func (m *Manager) Clear() {
	m.history = nil
	m.summary = ""
}

// Budget returns the current token budget breakdown.
func (m *Manager) Budget() BudgetInfo {
	systemMsgs := m.buildSystemMessages()
	systemTokens := m.estimator.EstimateMessages(systemMsgs)

	available := m.cfg.CtxSize - systemTokens - m.cfg.ResponseBudget - m.cfg.ToolsBudget

	memoryTokens := 0
	if len(m.memories) > 0 {
		memoryTokens = m.estimator.EstimateMessages(m.memories)
		available -= memoryTokens
	}

	summaryTokens := 0
	if m.summary != "" {
		sm := api.Message{
			Role:    "system",
			Content: fmt.Sprintf("[Conversation summary] %s", m.summary),
		}
		summaryTokens = m.estimator.EstimateMessages([]api.Message{sm})
		available -= summaryTokens
	}

	// Count tokens in the windowed history
	msgs := m.Messages()
	// After merging, Messages() returns [single system msg?] + history
	historyStart := 0
	if len(msgs) > 0 && msgs[0].Role == "system" {
		historyStart = 1
	}
	historyMsgs := msgs[historyStart:]
	historyTokens := m.estimator.EstimateMessages(historyMsgs)

	if available < 0 {
		available = 0
	}
	available -= historyTokens
	if available < 0 {
		available = 0
	}

	return BudgetInfo{
		Total:        m.cfg.CtxSize,
		System:       systemTokens,
		Memory:       memoryTokens,
		History:      historyTokens,
		Summary:      summaryTokens,
		Available:    available,
		HistoryCount: len(historyMsgs),
		TotalHistory: len(m.history),
	}
}

// SetSummary sets the conversation summary directly (used by Summarize).
func (m *Manager) SetSummary(summary string) {
	m.summary = summary
}

// Summary returns the current summary text.
func (m *Manager) Summary() string {
	return m.summary
}

// History returns a copy of the full history (including evicted messages).
func (m *Manager) History() []api.Message {
	out := make([]api.Message, len(m.history))
	copy(out, m.history)
	return out
}
