package chatctx

import (
	"context"
	"fmt"
	"strings"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

// midTurnHighWater is the fraction of the per-request token budget at which
// CompactInFlight starts folding. Set below 1.0 so we fold before the model
// errors on overflow — the next iteration still needs room to append a tool
// result, and the response itself reserves more on top.
const midTurnHighWater = 0.85

// midTurnTargetRatio is the fraction of the per-request budget we aim to
// shrink down to during a mid-turn fold. Set well below the high-water mark
// — a single large tool result (file_read on a big file, grep_search across
// a wide tree) can spend 10–20% of the budget in one shot, so a shallow
// fold would just re-trigger compaction on the next iteration. Going this
// deep keeps a handful of recent turns in plain view while preserving
// everything older as a summary.
const midTurnTargetRatio = 0.35

// manualKeepHistory is the number of trailing history messages that
// SummarizeNow leaves untouched — the user clicked "Compact now" because
// they want to keep working, so the most recent few exchanges stay in
// plain view and only the older portion folds into the summary.
const manualKeepHistory = 4

// CompletionFunc sends a chat completion request and returns the response.
type CompletionFunc func(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error)

const summarizationPrompt = `Summarize the following conversation concisely. Preserve:
- Key facts and decisions made
- File paths and code references mentioned
- Tool results and their outcomes
- User preferences and requirements stated
- Any errors encountered and how they were resolved

Be concise but thorough. This summary will replace the original messages to save context space.`

// Summarize condenses older messages that won't fit in the context window.
// It calls the LLM to generate a summary, then stores it in the Manager.
// The summary replaces evicted messages when Messages() builds the window.
func (m *Manager) Summarize(ctx context.Context, complete CompletionFunc) error {
	if !m.NeedsSummary() {
		return nil
	}

	// Figure out which messages are being evicted (won't fit in window)
	systemMsgs := m.buildSystemMessages()
	systemTokens := m.estimator.EstimateMessages(systemMsgs)
	available := m.cfg.CtxSize - systemTokens - m.cfg.ResponseBudget

	if m.summary != "" {
		sm := api.Message{
			Role:    "system",
			Content: fmt.Sprintf("[Conversation summary] %s", m.summary),
		}
		available -= m.estimator.EstimateMessages([]api.Message{sm})
	}

	// Find cutoff: walk backwards
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

	if cutoff == 0 {
		return nil // nothing to summarize
	}

	// Collect messages to summarize (the ones that would be evicted)
	toSummarize := m.history[:cutoff]

	// Cap summarization input to 50% of context window to avoid overflow
	maxSummarizeTokens := m.cfg.CtxSize / 2
	summarizeTokens := 0
	startIdx := 0
	for i := len(toSummarize) - 1; i >= 0; i-- {
		t := m.estimator.EstimateMessages([]api.Message{toSummarize[i]})
		if summarizeTokens+t > maxSummarizeTokens {
			startIdx = i + 1

			break
		}
		summarizeTokens += t
	}
	toSummarize = toSummarize[startIdx:]

	if len(toSummarize) == 0 {
		return nil
	}

	// Build the summarization request
	summaryMsgs := []api.Message{
		{Role: "system", Content: summarizationPrompt},
	}

	// Include existing summary if present
	if m.summary != "" {
		summaryMsgs = append(summaryMsgs, api.Message{
			Role:    "user",
			Content: fmt.Sprintf("Previous summary:\n%s", m.summary),
		})
	}

	// Add the messages to summarize as a user message
	var msgText string
	for _, msg := range toSummarize {
		msgText += fmt.Sprintf("[%s", msg.Role)
		if msg.Name != "" {
			msgText += fmt.Sprintf("/%s", msg.Name)
		}
		msgText += fmt.Sprintf("] %s\n", msg.Content)
	}

	summaryMsgs = append(summaryMsgs, api.Message{
		Role:    "user",
		Content: fmt.Sprintf("Conversation to summarize:\n%s", msgText),
	})

	req := &api.ChatCompletionRequest{
		Messages: summaryMsgs,
		Stream:   false,
	}

	resp, err := complete(ctx, req)
	if err != nil {
		return fmt.Errorf("summarization failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return fmt.Errorf("empty summarization response")
	}

	m.summary = resp.Choices[0].Message.Content

	// Remove the summarized messages from history
	m.history = m.history[cutoff:]

	return nil
}

// SummarizeNow is the manual-compaction entry point. Unlike Summarize() it
// does NOT gate on NeedsSummary() — the user clicked "Compact now"
// deliberately and expects a fold to happen even when the context window
// has plenty of headroom. It keeps the last manualKeepHistory messages
// intact (so the immediate exchange the user is still working on stays
// visible to the model) and folds everything older into m.summary. Returns
// (folded, err) where folded is 0 when there isn't enough history to fold
// — the caller can use that to render "Nothing to compact" instead of a
// misleading "Compacted 0 messages".
func (m *Manager) SummarizeNow(ctx context.Context, complete CompletionFunc) (int, error) {
	if len(m.history) <= manualKeepHistory {
		return 0, nil
	}

	cutoff := len(m.history) - manualKeepHistory
	// Don't sever a tool_call/tool-result pair: walk the cutoff forward
	// past any tool messages so the kept history still has its matching
	// assistant tool_calls at the front.
	for cutoff < len(m.history) && m.history[cutoff].Role == "tool" {
		cutoff++
	}
	if cutoff >= len(m.history) {
		return 0, nil
	}

	toSummarize := m.history[:cutoff]

	// Cap summarization input at half the context window, same as
	// Summarize(), so a very long history doesn't blow the per-request
	// budget while building the summary prompt itself.
	maxSummarizeTokens := m.cfg.CtxSize / 2
	summarizeTokens := 0
	startIdx := 0
	for i := len(toSummarize) - 1; i >= 0; i-- {
		t := m.estimator.EstimateMessages([]api.Message{toSummarize[i]})
		if summarizeTokens+t > maxSummarizeTokens {
			startIdx = i + 1

			break
		}
		summarizeTokens += t
	}
	toSummarize = toSummarize[startIdx:]
	if len(toSummarize) == 0 {
		return 0, nil
	}

	summaryMsgs := []api.Message{{Role: "system", Content: summarizationPrompt}}
	if m.summary != "" {
		summaryMsgs = append(summaryMsgs, api.Message{
			Role:    "user",
			Content: fmt.Sprintf("Previous summary:\n%s", m.summary),
		})
	}
	var msgText string
	for _, sm := range toSummarize {
		msgText += fmt.Sprintf("[%s", sm.Role)
		if sm.Name != "" {
			msgText += fmt.Sprintf("/%s", sm.Name)
		}
		msgText += fmt.Sprintf("] %s\n", sm.Content)
	}
	summaryMsgs = append(summaryMsgs, api.Message{
		Role:    "user",
		Content: fmt.Sprintf("Conversation to summarize:\n%s", msgText),
	})

	resp, err := complete(ctx, &api.ChatCompletionRequest{Messages: summaryMsgs, Stream: false})
	if err != nil {
		return 0, fmt.Errorf("manual summarization failed: %w", err)
	}
	if len(resp.Choices) == 0 {
		return 0, fmt.Errorf("empty manual summarization response")
	}

	m.summary = resp.Choices[0].Message.Content
	m.history = m.history[cutoff:]

	return len(toSummarize), nil
}

// NeedsMidTurnCompact returns true when an in-flight agent-loop messages
// slice has crossed the high-water mark for the per-request token budget.
// Cheap pre-check so callers can decide whether to emit "compacting…" UI
// before paying for the (expensive) summarization round-trip.
func (m *Manager) NeedsMidTurnCompact(msgs []api.Message) bool {
	if len(msgs) == 0 {
		return false
	}
	available := m.cfg.CtxSize - m.cfg.ResponseBudget - m.cfg.ToolsBudget
	if available <= 0 {
		return false
	}

	return m.estimator.EstimateMessages(msgs) > int(float64(available)*midTurnHighWater)
}

// summarySectionMarker is the prefix used to mark the conversation-summary
// portion inside the merged system block. CompactInFlight and Messages()
// both write this exact prefix, and CompactInFlight uses it to find and
// replace any prior summary when re-folding mid-turn.
const summarySectionMarker = "[Conversation summary] "

// CompactInFlight examines an in-flight agent-loop messages slice and, if
// its per-request token cost has crossed the high-water mark, folds the
// oldest non-system messages into a summary embedded inside the leading
// system block. Returns the rewritten slice, the new summary text, the
// number of folded messages, and any error.
//
// The function is PURE on the Manager: it reads Config + estimator but
// never mutates m.summary or m.history. Callers that want to persist the
// new summary across turns (the main agent path) hand it to
// AbsorbMidTurnCompaction explicitly. Callers operating on transient
// subagent contexts (swarm workers, verifier) ignore the returned summary
// — when the subagent's msgs are discarded at completion, the summary
// evaporates with them. This makes each subagent's compaction isolated to
// its own context.
//
// The leading system messages of msgs are treated as the pinned block: the
// existing "[Conversation summary] …" section inside them is stripped,
// the model is asked to produce a new summary that incorporates both any
// prior summary and the freshly-folded tail, and the new summary is
// appended back into the pinned block as a single rewritten system
// message. If folding would split a tool_call from its tool result the
// cutoff walks forward until the pair stays intact.
func (m *Manager) CompactInFlight(ctx context.Context, msgs []api.Message, complete CompletionFunc) ([]api.Message, string, int, error) {
	if len(msgs) == 0 {
		return msgs, "", 0, nil
	}
	available := m.cfg.CtxSize - m.cfg.ResponseBudget - m.cfg.ToolsBudget
	if available <= 0 {
		return msgs, "", 0, nil
	}

	total := m.estimator.EstimateMessages(msgs)
	if total <= int(float64(available)*midTurnHighWater) {
		return msgs, "", 0, nil
	}

	historyStart := 0
	for historyStart < len(msgs) && msgs[historyStart].Role == "system" {
		historyStart++
	}
	if historyStart >= len(msgs) {
		return msgs, "", 0, nil
	}

	pinned := msgs[:historyStart]
	history := msgs[historyStart:]

	pinnedBase, prevSummary := splitPinnedAndSummary(pinned)
	pinnedBaseTokens := m.estimator.EstimateMessages([]api.Message{
		{Role: "system", Content: pinnedBase},
	})

	target := int(float64(available)*midTurnTargetRatio) - pinnedBaseTokens
	if target < 0 {
		target = 0
	}

	keep := 0
	keepTokens := 0
	for i := len(history) - 1; i >= 0; i-- {
		t := m.estimator.EstimateMessages([]api.Message{history[i]})
		if keepTokens+t > target {
			break
		}
		keepTokens += t
		keep++
	}
	cutoff := len(history) - keep
	if cutoff <= 0 {
		return msgs, "", 0, nil
	}

	for cutoff < len(history) && history[cutoff].Role == "tool" {
		cutoff++
	}
	if cutoff >= len(history) {
		return msgs, "", 0, nil
	}

	toSummarize := history[:cutoff]
	keepMsgs := history[cutoff:]

	maxSummarizeTokens := m.cfg.CtxSize / 2
	summarizeTokens := 0
	startIdx := 0
	for i := len(toSummarize) - 1; i >= 0; i-- {
		t := m.estimator.EstimateMessages([]api.Message{toSummarize[i]})
		if summarizeTokens+t > maxSummarizeTokens {
			startIdx = i + 1

			break
		}
		summarizeTokens += t
	}
	toSummarize = toSummarize[startIdx:]
	if len(toSummarize) == 0 {
		return msgs, "", 0, nil
	}

	summaryMsgs := []api.Message{{Role: "system", Content: summarizationPrompt}}
	if prevSummary != "" {
		summaryMsgs = append(summaryMsgs, api.Message{
			Role:    "user",
			Content: fmt.Sprintf("Previous summary:\n%s", prevSummary),
		})
	}
	var msgText string
	for _, sm := range toSummarize {
		msgText += fmt.Sprintf("[%s", sm.Role)
		if sm.Name != "" {
			msgText += fmt.Sprintf("/%s", sm.Name)
		}
		msgText += fmt.Sprintf("] %s\n", sm.Content)
	}
	summaryMsgs = append(summaryMsgs, api.Message{
		Role:    "user",
		Content: fmt.Sprintf("Conversation to summarize:\n%s", msgText),
	})

	resp, err := complete(ctx, &api.ChatCompletionRequest{Messages: summaryMsgs, Stream: false})
	if err != nil {
		return msgs, "", 0, fmt.Errorf("mid-turn summarization failed: %w", err)
	}
	if len(resp.Choices) == 0 {
		return msgs, "", 0, fmt.Errorf("empty mid-turn summarization response")
	}
	newSummary := resp.Choices[0].Message.Content

	newPinned := pinnedBase
	if newPinned != "" {
		newPinned += "\n\n"
	}
	newPinned += summarySectionMarker + newSummary

	rebuilt := make([]api.Message, 0, 1+len(keepMsgs))
	if newPinned != "" {
		rebuilt = append(rebuilt, api.Message{Role: "system", Content: newPinned})
	}
	rebuilt = append(rebuilt, keepMsgs...)

	return rebuilt, newSummary, len(toSummarize), nil
}

// AbsorbMidTurnCompaction reconciles the Manager's stored summary and
// history with the result of a successful CompactInFlight call. Only the
// main agent path calls this — subagent paths (swarm workers, verifier)
// skip it so their compactions stay scoped to their transient msgs.
//
// rewritten is the slice CompactInFlight returned; newSummary is the
// freshly-generated summary text. The Manager's stored history is replaced
// with the rewritten slice's tail (everything after its leading system
// block) so a subsequent Manager.Messages() call rebuilds an accurate
// window from the post-fold state.
func (m *Manager) AbsorbMidTurnCompaction(rewritten []api.Message, newSummary string) {
	m.summary = newSummary
	historyStart := 0
	for historyStart < len(rewritten) && rewritten[historyStart].Role == "system" {
		historyStart++
	}
	tail := rewritten[historyStart:]
	newHistory := make([]api.Message, len(tail))
	copy(newHistory, tail)
	m.history = newHistory
}

// splitPinnedAndSummary parses the leading-system block produced by
// Manager.Messages() (or a previous CompactInFlight fold). The base portion
// is everything before the summary marker; prevSummary is whatever followed
// it, with surrounding whitespace trimmed. If no marker is found, the
// entire merged content is treated as base and prevSummary is empty —
// which is the swarm-worker case (its preamble has no summary section).
func splitPinnedAndSummary(systemBlock []api.Message) (base, prevSummary string) {
	var merged string
	for i, m := range systemBlock {
		if i > 0 {
			merged += "\n\n"
		}
		merged += m.Content
	}
	idx := strings.Index(merged, summarySectionMarker)
	if idx < 0 {
		return strings.TrimRight(merged, "\n"), ""
	}
	base = strings.TrimRight(merged[:idx], "\n")
	prevSummary = strings.TrimSpace(merged[idx+len(summarySectionMarker):])

	return base, prevSummary
}
