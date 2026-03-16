package chatctx

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

func newTestManager(ctxSize int) *Manager {
	return NewManager(Config{
		CtxSize:        ctxSize,
		ResponseBudget: 100,
	}, NewTokenEstimator())
}

func TestMessagesUnderBudget(t *testing.T) {
	mgr := newTestManager(10000)
	mgr.SetSystemPrompt("You are helpful.")
	mgr.Append(api.Message{Role: "user", Content: "Hello"})
	mgr.Append(api.Message{Role: "assistant", Content: "Hi there!"})

	msgs := mgr.Messages()
	if len(msgs) != 3 { // system + 2 history
		t.Errorf("expected 3 messages, got %d", len(msgs))
	}
	if msgs[0].Role != "system" {
		t.Errorf("first message should be system, got %s", msgs[0].Role)
	}
	if msgs[1].Content != "Hello" {
		t.Errorf("expected 'Hello', got %q", msgs[1].Content)
	}
}

func TestMessagesOverBudget(t *testing.T) {
	// Small context window to force windowing
	mgr := newTestManager(200)
	mgr.SetSystemPrompt("sys")

	// Add many messages that won't all fit
	for i := 0; i < 20; i++ {
		mgr.Append(api.Message{Role: "user", Content: fmt.Sprintf("Message number %d with some extra text padding", i)})
		mgr.Append(api.Message{Role: "assistant", Content: fmt.Sprintf("Response number %d with some extra text padding", i)})
	}

	msgs := mgr.Messages()

	// Should have fewer than all 40 history messages + system
	if len(msgs) >= 42 {
		t.Errorf("expected windowing to drop messages, got %d messages", len(msgs))
	}

	// First message should still be system
	if msgs[0].Role != "system" {
		t.Errorf("first message should be system, got %s", msgs[0].Role)
	}

	// Last message should be the most recent one
	lastMsg := msgs[len(msgs)-1]
	if lastMsg.Content != "Response number 19 with some extra text padding" {
		t.Errorf("last message should be most recent, got %q", lastMsg.Content)
	}
}

func TestSystemPromptAlwaysFirst(t *testing.T) {
	mgr := newTestManager(200)
	mgr.SetSystemPrompt("Important system prompt")

	for i := 0; i < 50; i++ {
		mgr.Append(api.Message{Role: "user", Content: strings.Repeat("x", 100)})
	}

	msgs := mgr.Messages()
	if len(msgs) == 0 {
		t.Fatal("expected at least system message")
	}
	if msgs[0].Role != "system" || msgs[0].Content != "Important system prompt" {
		t.Errorf("system prompt should always be first, got role=%s content=%q", msgs[0].Role, msgs[0].Content)
	}
}

func TestContextFilesPinned(t *testing.T) {
	mgr := newTestManager(2000)
	mgr.SetSystemPrompt("sys")
	mgr.AddContextFile("go.mod", "module test\ngo 1.21")

	for i := 0; i < 10; i++ {
		mgr.Append(api.Message{Role: "user", Content: "hello"})
		mgr.Append(api.Message{Role: "assistant", Content: "hi"})
	}

	msgs := mgr.Messages()

	// Should have merged system message + some history
	if len(msgs) < 2 {
		t.Fatalf("expected at least 2 messages, got %d", len(msgs))
	}

	// First should be merged system message containing prompt and context file
	if msgs[0].Role != "system" {
		t.Errorf("first should be system, got role %q", msgs[0].Role)
	}
	if !strings.Contains(msgs[0].Content, "sys") {
		t.Errorf("system message should contain prompt, got %q", msgs[0].Content)
	}
	if !strings.Contains(msgs[0].Content, "go.mod") {
		t.Errorf("system message should contain context file, got %q", msgs[0].Content)
	}
}

func TestContextFilesListed(t *testing.T) {
	mgr := newTestManager(4096)
	mgr.AddContextFile("a.txt", "content a")
	mgr.AddContextFile("b.txt", "content b")

	files := mgr.ContextFiles()
	if len(files) != 2 {
		t.Fatalf("expected 2 files, got %d", len(files))
	}
	if files[0] != "a.txt" || files[1] != "b.txt" {
		t.Errorf("unexpected files: %v", files)
	}
}

func TestClearContextFiles(t *testing.T) {
	mgr := newTestManager(4096)
	mgr.AddContextFile("a.txt", "content")
	mgr.ClearContextFiles()

	if len(mgr.ContextFiles()) != 0 {
		t.Error("context files should be empty after clear")
	}
}

func TestSummarization(t *testing.T) {
	mgr := newTestManager(300)
	mgr.SetSystemPrompt("sys")

	// Add enough messages to trigger NeedsSummary
	for i := 0; i < 20; i++ {
		mgr.Append(api.Message{Role: "user", Content: fmt.Sprintf("Message %d with padding text here", i)})
		mgr.Append(api.Message{Role: "assistant", Content: fmt.Sprintf("Response %d with padding text here", i)})
	}

	if !mgr.NeedsSummary() {
		t.Fatal("expected NeedsSummary to be true")
	}

	// Mock completion function for summarization
	mockComplete := func(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		return &api.ChatCompletionResponse{
			Choices: []api.Choice{
				{Message: api.Message{Role: "assistant", Content: "Summary: user sent 20 messages about various topics."}},
			},
		}, nil
	}

	historyBefore := len(mgr.History())

	err := mgr.Summarize(context.Background(), mockComplete)
	if err != nil {
		t.Fatalf("Summarize failed: %v", err)
	}

	// Summary should be set
	if mgr.Summary() == "" {
		t.Error("expected summary to be set after Summarize")
	}

	// History should be shorter (evicted messages removed)
	if len(mgr.History()) >= historyBefore {
		t.Errorf("history should be shorter after summarization: before=%d, after=%d", historyBefore, len(mgr.History()))
	}

	// Messages() should include the summary
	msgs := mgr.Messages()
	hasSummary := false
	for _, msg := range msgs {
		if strings.Contains(msg.Content, "[Conversation summary]") {
			hasSummary = true
			break
		}
	}
	if !hasSummary {
		t.Error("expected summary message in Messages() output")
	}
}

func TestSummarizationNotNeeded(t *testing.T) {
	mgr := newTestManager(10000)
	mgr.Append(api.Message{Role: "user", Content: "hello"})

	if mgr.NeedsSummary() {
		t.Error("NeedsSummary should be false when everything fits")
	}
}

func TestClear(t *testing.T) {
	mgr := newTestManager(4096)
	mgr.SetSystemPrompt("keep me")
	mgr.AddContextFile("keep.txt", "keep this")
	mgr.Append(api.Message{Role: "user", Content: "hello"})
	mgr.Append(api.Message{Role: "assistant", Content: "hi"})
	mgr.SetSummary("old summary")

	mgr.Clear()

	if len(mgr.History()) != 0 {
		t.Error("history should be empty after Clear")
	}
	if mgr.Summary() != "" {
		t.Error("summary should be empty after Clear")
	}

	// System prompt and context files should be preserved in single system message
	msgs := mgr.Messages()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message (merged system), got %d", len(msgs))
	}
	if !strings.Contains(msgs[0].Content, "keep me") {
		t.Errorf("system prompt should be preserved, got %q", msgs[0].Content)
	}
	if !strings.Contains(msgs[0].Content, "keep.txt") {
		t.Errorf("context file should be preserved, got %q", msgs[0].Content)
	}
}

func TestBudgetInfo(t *testing.T) {
	mgr := newTestManager(4096)
	mgr.SetSystemPrompt("system prompt here")
	mgr.Append(api.Message{Role: "user", Content: "hello"})
	mgr.Append(api.Message{Role: "assistant", Content: "hi"})

	budget := mgr.Budget()

	if budget.Total != 4096 {
		t.Errorf("Total = %d, want 4096", budget.Total)
	}
	if budget.System <= 0 {
		t.Errorf("System = %d, want > 0", budget.System)
	}
	if budget.History <= 0 {
		t.Errorf("History = %d, want > 0", budget.History)
	}
	if budget.HistoryCount != 2 {
		t.Errorf("HistoryCount = %d, want 2", budget.HistoryCount)
	}
	if budget.TotalHistory != 2 {
		t.Errorf("TotalHistory = %d, want 2", budget.TotalHistory)
	}
	if budget.Available <= 0 {
		t.Errorf("Available = %d, want > 0", budget.Available)
	}
	if budget.System+budget.History+budget.Summary+budget.Available+100 != budget.Total {
		t.Errorf("budget components don't add up: system=%d + history=%d + summary=%d + available=%d + response_budget(100) != total=%d",
			budget.System, budget.History, budget.Summary, budget.Available, budget.Total)
	}
}

func TestBudgetInfoWithSummary(t *testing.T) {
	mgr := newTestManager(4096)
	mgr.SetSystemPrompt("sys")
	mgr.SetSummary("This is a summary of previous conversation.")
	mgr.Append(api.Message{Role: "user", Content: "hello"})

	budget := mgr.Budget()
	if budget.Summary <= 0 {
		t.Errorf("Summary tokens = %d, want > 0", budget.Summary)
	}
}

func TestAppendMany(t *testing.T) {
	mgr := newTestManager(4096)

	msgs := []api.Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
		{Role: "user", Content: "how are you?"},
	}

	mgr.AppendMany(msgs)

	if len(mgr.History()) != 3 {
		t.Errorf("expected 3 history messages, got %d", len(mgr.History()))
	}
}

func TestDefaultConfig(t *testing.T) {
	mgr := NewManager(Config{}, NewTokenEstimator())
	if mgr.cfg.CtxSize != 4096 {
		t.Errorf("default CtxSize = %d, want 4096", mgr.cfg.CtxSize)
	}
	if mgr.cfg.ResponseBudget != 512 {
		t.Errorf("default ResponseBudget = %d, want 512", mgr.cfg.ResponseBudget)
	}
}

func TestNoSystemPromptMessages(t *testing.T) {
	mgr := newTestManager(4096)
	// No system prompt set
	mgr.Append(api.Message{Role: "user", Content: "hello"})

	msgs := mgr.Messages()
	if len(msgs) != 1 {
		t.Errorf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0].Role != "user" {
		t.Errorf("expected user message, got %s", msgs[0].Role)
	}
}

func TestEmptyHistory(t *testing.T) {
	mgr := newTestManager(4096)
	mgr.SetSystemPrompt("sys")

	msgs := mgr.Messages()
	if len(msgs) != 1 {
		t.Errorf("expected 1 message (system only), got %d", len(msgs))
	}
}

func TestSummarizeError(t *testing.T) {
	mgr := newTestManager(300)
	mgr.SetSystemPrompt("sys")

	for i := 0; i < 20; i++ {
		mgr.Append(api.Message{Role: "user", Content: fmt.Sprintf("Message %d with some padding", i)})
	}

	// Mock completion function that fails
	mockComplete := func(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		return nil, fmt.Errorf("connection refused")
	}

	err := mgr.Summarize(context.Background(), mockComplete)
	if err == nil {
		t.Error("expected error from failed summarization")
	}
}

func TestSummarizeEmptyResponse(t *testing.T) {
	mgr := newTestManager(300)
	mgr.SetSystemPrompt("sys")

	for i := 0; i < 20; i++ {
		mgr.Append(api.Message{Role: "user", Content: fmt.Sprintf("Message %d with some padding", i)})
	}

	mockComplete := func(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error) {
		return &api.ChatCompletionResponse{Choices: []api.Choice{}}, nil
	}

	err := mgr.Summarize(context.Background(), mockComplete)
	if err == nil {
		t.Error("expected error from empty summarization response")
	}
}

func TestScrollsInMessages(t *testing.T) {
	mgr := newTestManager(4096)
	mgr.SetSystemPrompt("sys")
	mgr.SetScrolls([]api.Message{
		{Role: "system", Content: "scroll content alpha"},
	})

	msgs := mgr.Messages()
	if len(msgs) == 0 {
		t.Fatal("expected at least one message")
	}
	if msgs[0].Role != "system" {
		t.Errorf("first message should be system, got %s", msgs[0].Role)
	}
	if !strings.Contains(msgs[0].Content, "scroll content alpha") {
		t.Errorf("system message should contain scroll content, got %q", msgs[0].Content)
	}
}

func TestScrollsBudget(t *testing.T) {
	mgr := newTestManager(4096)
	mgr.SetSystemPrompt("sys")
	mgr.SetScrolls([]api.Message{
		{Role: "system", Content: "scroll budget test content"},
	})

	budget := mgr.Budget()
	if budget.Scrolls <= 0 {
		t.Errorf("Scrolls = %d, want > 0", budget.Scrolls)
	}
	if budget.System+budget.Scrolls+budget.Memory+budget.History+budget.Summary+budget.Available+100 != budget.Total {
		t.Errorf("budget components don't add up: system=%d + scrolls=%d + memory=%d + history=%d + summary=%d + available=%d + response_budget(100) != total=%d",
			budget.System, budget.Scrolls, budget.Memory, budget.History, budget.Summary, budget.Available, budget.Total)
	}
}

func TestScrollsOrderInSystemMessage(t *testing.T) {
	mgr := newTestManager(4096)
	mgr.SetSystemPrompt("PROMPT_CONTENT")
	mgr.AddContextFile("ctx.txt", "CONTEXT_FILE_CONTENT")
	mgr.SetScrolls([]api.Message{
		{Role: "system", Content: "SCROLL_CONTENT"},
	})
	mgr.SetMemories([]api.Message{
		{Role: "system", Content: "MEMORY_CONTENT"},
	})
	mgr.SetSummary("SUMMARY_CONTENT")

	msgs := mgr.Messages()
	if len(msgs) == 0 {
		t.Fatal("expected at least one message")
	}
	sys := msgs[0].Content

	promptIdx := strings.Index(sys, "PROMPT_CONTENT")
	contextIdx := strings.Index(sys, "CONTEXT_FILE_CONTENT")
	scrollIdx := strings.Index(sys, "SCROLL_CONTENT")
	memoryIdx := strings.Index(sys, "MEMORY_CONTENT")
	summaryIdx := strings.Index(sys, "SUMMARY_CONTENT")

	for name, idx := range map[string]int{
		"prompt":       promptIdx,
		"context file": contextIdx,
		"scroll":       scrollIdx,
		"memory":       memoryIdx,
		"summary":      summaryIdx,
	} {
		if idx == -1 {
			t.Errorf("system message missing %s content", name)
		}
	}

	// Verify order: prompt < context file < scroll < memory < summary
	if promptIdx > contextIdx {
		t.Errorf("prompt (%d) should appear before context file (%d)", promptIdx, contextIdx)
	}
	if contextIdx > scrollIdx {
		t.Errorf("context file (%d) should appear before scroll (%d)", contextIdx, scrollIdx)
	}
	if scrollIdx > memoryIdx {
		t.Errorf("scroll (%d) should appear before memory (%d)", scrollIdx, memoryIdx)
	}
	if memoryIdx > summaryIdx {
		t.Errorf("memory (%d) should appear before summary (%d)", memoryIdx, summaryIdx)
	}
}

func TestClearScrolls(t *testing.T) {
	mgr := newTestManager(4096)
	mgr.SetSystemPrompt("sys")
	mgr.SetScrolls([]api.Message{
		{Role: "system", Content: "scroll to be cleared"},
	})

	// Verify scroll is present before clearing
	msgs := mgr.Messages()
	if !strings.Contains(msgs[0].Content, "scroll to be cleared") {
		t.Fatal("scroll should be present before ClearScrolls")
	}

	mgr.ClearScrolls()

	msgs = mgr.Messages()
	if strings.Contains(msgs[0].Content, "scroll to be cleared") {
		t.Errorf("scroll content should be gone after ClearScrolls, got %q", msgs[0].Content)
	}

	budget := mgr.Budget()
	if budget.Scrolls != 0 {
		t.Errorf("Scrolls budget = %d, want 0 after ClearScrolls", budget.Scrolls)
	}
}

func TestScrollsWithMemoriesBudget(t *testing.T) {
	mgr := newTestManager(4096)
	mgr.SetSystemPrompt("sys")
	mgr.SetScrolls([]api.Message{
		{Role: "system", Content: "scroll content for budget test"},
	})
	mgr.SetMemories([]api.Message{
		{Role: "system", Content: "memory content for budget test"},
	})

	budget := mgr.Budget()
	if budget.Scrolls <= 0 {
		t.Errorf("Scrolls = %d, want > 0", budget.Scrolls)
	}
	if budget.Memory <= 0 {
		t.Errorf("Memory = %d, want > 0", budget.Memory)
	}
	if budget.System+budget.Scrolls+budget.Memory+budget.History+budget.Summary+budget.Available+100 != budget.Total {
		t.Errorf("budget components don't add up: system=%d + scrolls=%d + memory=%d + history=%d + summary=%d + available=%d + response_budget(100) != total=%d",
			budget.System, budget.Scrolls, budget.Memory, budget.History, budget.Summary, budget.Available, budget.Total)
	}
}

func TestNeedsSummaryWithScrolls(t *testing.T) {
	// Use a small context window so scrolls noticeably reduce available space
	mgr := newTestManager(300)
	mgr.SetSystemPrompt("sys")

	// Add messages that fill roughly half the available space without scrolls
	for i := 0; i < 5; i++ {
		mgr.Append(api.Message{Role: "user", Content: fmt.Sprintf("Message %d with padding text here", i)})
		mgr.Append(api.Message{Role: "assistant", Content: fmt.Sprintf("Response %d with padding text here", i)})
	}

	// Confirm no summary needed yet (messages fit)
	if mgr.NeedsSummary() {
		t.Skip("history already too large for this context size; adjust test parameters")
	}

	// Now inject large scrolls to consume the remaining available space
	mgr.SetScrolls([]api.Message{
		{Role: "system", Content: strings.Repeat("scroll content ", 30)},
	})

	if !mgr.NeedsSummary() {
		t.Error("NeedsSummary should be true after large scrolls reduce available space")
	}
}
