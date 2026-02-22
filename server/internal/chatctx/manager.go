package chatctx

import (
	"fmt"

	"github.com/thatcatdev/tanrenai/server/pkg/api"
)

// Config configures the context manager.
type Config struct {
	CtxSize        int // total context window in tokens (default 4096)
	ResponseBudget int // tokens reserved for model output (default 512)
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
// Algorithm:
// 1. Compute system tokens from pinned system messages
// 2. available = CtxSize - systemTokens - ResponseBudget
// 3. If summary exists, subtract its tokens from available
// 4. Walk history backwards, summing tokens
// 5. Stop when next message would exceed available
// 6. Return: systemMsgs + [summary msg if present] + history[cutoff:]
func (m *Manager) Messages() []api.Message {
	systemMsgs := m.buildSystemMessages()
	systemTokens := m.estimator.EstimateMessages(systemMsgs)

	available := m.cfg.CtxSize - systemTokens - m.cfg.ResponseBudget
	if available < 0 {
		available = 0
	}

	// Reserve space for memories if present
	memoryTokens := 0
	if len(m.memories) > 0 {
		memoryTokens = m.estimator.EstimateMessages(m.memories)
		available -= memoryTokens
		if available < 0 {
			available = 0
		}
	}

	// Reserve space for summary if present
	var summaryMsg *api.Message
	if m.summary != "" {
		sm := api.Message{
			Role:    "system",
			Content: fmt.Sprintf("[Conversation summary] %s", m.summary),
		}
		summaryTokens := m.estimator.EstimateMessages([]api.Message{sm})
		available -= summaryTokens
		if available < 0 {
			available = 0
		}
		summaryMsg = &sm
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

	// Build result: [system] + [memories] + [summary?] + [windowed history]
	result := make([]api.Message, 0, len(systemMsgs)+len(m.memories)+1+len(m.history)-cutoff)
	result = append(result, systemMsgs...)
	result = append(result, m.memories...)
	if summaryMsg != nil {
		result = append(result, *summaryMsg)
	}
	result = append(result, m.history[cutoff:]...)

	return result
}

// buildSystemMessages constructs the pinned system messages.
func (m *Manager) buildSystemMessages() []api.Message {
	var msgs []api.Message

	if m.systemPrompt != "" {
		msgs = append(msgs, api.Message{Role: "system", Content: m.systemPrompt})
	}

	for _, cf := range m.contextFiles {
		msgs = append(msgs, api.Message{
			Role:    "system",
			Content: fmt.Sprintf("[File: %s]\n%s", cf.Path, cf.Content),
		})
	}

	return msgs
}

// NeedsSummary returns true if the history has messages that won't fit in the window
// and could benefit from summarization.
func (m *Manager) NeedsSummary() bool {
	systemMsgs := m.buildSystemMessages()
	systemTokens := m.estimator.EstimateMessages(systemMsgs)
	available := m.cfg.CtxSize - systemTokens - m.cfg.ResponseBudget

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

	available := m.cfg.CtxSize - systemTokens - m.cfg.ResponseBudget

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
	// History messages are everything after system + memories + summary messages
	historyStart := len(systemMsgs) + len(m.memories)
	if m.summary != "" {
		historyStart++
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
