package e2e

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ThatCatDev/tanrenai/server/internal/agent"
	"github.com/ThatCatDev/tanrenai/server/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/server/internal/memory"
	"github.com/ThatCatDev/tanrenai/server/internal/runner"
	"github.com/ThatCatDev/tanrenai/server/internal/tools"
	"github.com/ThatCatDev/tanrenai/server/pkg/api"
)

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

const embedDim = 64

// mockEmbedFunc produces deterministic 64-dim normalized vectors using FNV hash.
func mockEmbedFunc(_ context.Context, text string) ([]float32, error) {
	h := fnv.New64a()
	h.Write([]byte(text))
	seed := h.Sum64()

	vec := make([]float32, embedDim)
	for i := range vec {
		b := make([]byte, 8)
		binary.LittleEndian.PutUint64(b, seed+uint64(i))
		h2 := fnv.New32a()
		h2.Write(b)
		vec[i] = float32(h2.Sum32())/float32(math.MaxUint32)*2 - 1
	}

	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	norm := float32(math.Sqrt(sum))
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
	return vec, nil
}

// makeStreamEvents simulates a streaming text response from the LLM.
func makeStreamEvents(content string) <-chan runner.StreamEvent {
	ch := make(chan runner.StreamEvent)
	go func() {
		defer close(ch)
		ch <- runner.StreamEvent{
			Chunk: &api.ChatCompletionChunk{
				ID:    "test-id",
				Model: "test-model",
				Choices: []api.ChunkChoice{{
					Index: 0,
					Delta: api.MessageDelta{Role: "assistant"},
				}},
			},
		}
		for i, word := range strings.Fields(content) {
			text := word
			if i < len(strings.Fields(content))-1 {
				text += " "
			}
			ch <- runner.StreamEvent{
				Chunk: &api.ChatCompletionChunk{
					ID:    "test-id",
					Model: "test-model",
					Choices: []api.ChunkChoice{{
						Index: 0,
						Delta: api.MessageDelta{Content: text},
					}},
				},
			}
		}
		ch <- runner.StreamEvent{Done: true}
	}()
	return ch
}

// makeStreamToolCallEvents simulates a streaming tool call response.
func makeStreamToolCallEvents(id, name, args string) <-chan runner.StreamEvent {
	ch := make(chan runner.StreamEvent)
	go func() {
		defer close(ch)
		ch <- runner.StreamEvent{
			Chunk: &api.ChatCompletionChunk{
				ID:    "test-id",
				Model: "test-model",
				Choices: []api.ChunkChoice{{
					Index: 0,
					Delta: api.MessageDelta{
						Role: "assistant",
						ToolCalls: []api.ToolCallDelta{{
							Index: 0,
							ID:    id,
							Type:  "function",
							Function: &api.ToolCallFunction{
								Name:      name,
								Arguments: args,
							},
						}},
					},
				}},
			},
		}
		ch <- runner.StreamEvent{Done: true}
	}()
	return ch
}

// simulateTurn runs one complete agent loop turn, replicating the logic from
// cmd.agentLoop: user message → memory search → memory injection → agent
// streaming completion → append new messages → store memory.
func simulateTurn(
	t *testing.T,
	mgr *chatctx.Manager,
	memStore memory.Store,
	registry *tools.Registry,
	input string,
	streamFn agent.StreamingCompletionFunc,
) ([]api.Message, error) {
	t.Helper()
	ctx := context.Background()

	mgr.Append(api.Message{Role: "user", Content: input})

	if memStore != nil {
		results, err := memStore.Search(ctx, input, 5)
		if err != nil {
			return nil, fmt.Errorf("memory search: %w", err)
		}
		if len(results) > 0 {
			mgr.SetMemories(formatMemories(results))
		} else {
			mgr.ClearMemories()
		}
	}

	windowedMsgs := mgr.Messages()

	cfg := agent.StreamingConfig{
		Config: agent.Config{
			MaxIterations: 20,
			Tools:         registry,
		},
	}

	result, err := agent.RunStreaming(ctx, streamFn, windowedMsgs, cfg)
	if err != nil {
		return nil, err
	}

	newMsgs := result[len(windowedMsgs):]
	mgr.AppendMany(newMsgs)

	if memStore != nil {
		assistContent := extractAssistantContent(newMsgs)
		if assistContent != "" {
			entry := memory.Entry{
				UserMsg:   input,
				AssistMsg: assistContent,
				Timestamp: time.Now(),
			}
			if err := memStore.Add(ctx, entry); err != nil {
				return nil, fmt.Errorf("store memory: %w", err)
			}
		}
	}

	return newMsgs, nil
}

// formatMemories converts search results into system messages, matching the
// format used by cmd.agentLoop.
func formatMemories(results []memory.SearchResult) []api.Message {
	msgs := make([]api.Message, 0, len(results))
	for _, r := range results {
		content := fmt.Sprintf("[Memory from %s] User asked: %s\nAssistant replied: %s",
			r.Entry.Timestamp.Format("2006-01-02"), r.Entry.UserMsg, r.Entry.AssistMsg)
		msgs = append(msgs, api.Message{Role: "system", Content: content})
	}
	return msgs
}

// extractAssistantContent collects assistant text content from new messages.
func extractAssistantContent(msgs []api.Message) string {
	var parts []string
	for _, msg := range msgs {
		if msg.Role == "assistant" && msg.Content != "" {
			parts = append(parts, msg.Content)
		}
	}
	return strings.Join(parts, "\n")
}

// echoTool is a minimal tool for testing that echoes its input.
type echoTool struct{}

func (e *echoTool) Name() string        { return "echo" }
func (e *echoTool) Description() string  { return "Echoes back the input text" }
func (e *echoTool) Parameters() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}`)
}
func (e *echoTool) Execute(_ context.Context, arguments string) (*tools.ToolResult, error) {
	var args struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return tools.ErrorResult(fmt.Sprintf("invalid arguments: %v", err)), nil
	}
	return &tools.ToolResult{Output: args.Text}, nil
}

// newTestSetup creates the common test components: context manager, in-memory
// store, and a tool registry with an echo tool.
func newTestSetup(t *testing.T) (*chatctx.Manager, memory.Store, *tools.Registry) {
	t.Helper()

	memStore, err := memory.NewChromemStoreInMemory(mockEmbedFunc)
	if err != nil {
		t.Fatalf("create memory store: %v", err)
	}

	estimator := chatctx.NewTokenEstimator()
	mgr := chatctx.NewManager(chatctx.Config{
		CtxSize:        4096,
		ResponseBudget: 512,
	}, estimator)
	mgr.SetSystemPrompt("You are a helpful assistant.")

	registry := tools.NewRegistry()
	registry.Register(&echoTool{})

	return mgr, memStore, registry
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

func TestE2EMemoryRetrievalAndInjection(t *testing.T) {
	mgr, memStore, registry := newTestSetup(t)
	defer memStore.Close()

	ctx := context.Background()

	// Pre-seed memory with a known entry.
	if err := memStore.Add(ctx, memory.Entry{
		UserMsg:   "how to compile Go",
		AssistMsg: "use go build",
		Timestamp: time.Now().Add(-1 * time.Hour),
	}); err != nil {
		t.Fatalf("seed memory: %v", err)
	}

	if memStore.Count() != 1 {
		t.Fatalf("expected 1 seeded memory, got %d", memStore.Count())
	}

	// Verify search returns the seeded entry.
	results, err := memStore.Search(ctx, "compile my Go project", 5)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected search to return at least 1 result")
	}
	if results[0].Entry.AssistMsg != "use go build" {
		t.Errorf("expected seeded entry assist msg, got %q", results[0].Entry.AssistMsg)
	}

	// Mock LLM: return a simple text response.
	streamFn := func(_ context.Context, _ *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		return makeStreamEvents("Run go build in your project directory."), nil
	}

	newMsgs, err := simulateTurn(t, mgr, memStore, registry, "compile my Go project", streamFn)
	if err != nil {
		t.Fatalf("simulateTurn: %v", err)
	}

	// Assert memory was injected into context.
	foundMemory := false
	for _, m := range mgr.Messages() {
		if m.Role == "system" && strings.Contains(m.Content, "[Memory from") {
			foundMemory = true
			break
		}
	}
	if !foundMemory {
		t.Error("expected memory system message in context")
	}

	// Assert agent produced a response.
	if len(newMsgs) == 0 {
		t.Fatal("expected at least one new message")
	}
	if extractAssistantContent(newMsgs) == "" {
		t.Error("expected non-empty assistant content")
	}

	// Assert new memory was stored (1 seed + 1 from this turn).
	if memStore.Count() != 2 {
		t.Errorf("expected 2 memories after turn, got %d", memStore.Count())
	}
}

func TestE2EMemoryWithToolCalls(t *testing.T) {
	mgr, memStore, registry := newTestSetup(t)
	defer memStore.Close()

	ctx := context.Background()

	// Pre-seed memory.
	if err := memStore.Add(ctx, memory.Entry{
		UserMsg:   "greeting protocols",
		AssistMsg: "use a friendly greeting",
		Timestamp: time.Now().Add(-1 * time.Hour),
	}); err != nil {
		t.Fatalf("seed memory: %v", err)
	}

	// Stateful mock: call 1 → tool call, call 2 → text response.
	var callCount atomic.Int32
	streamFn := func(_ context.Context, _ *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		n := callCount.Add(1)
		if n == 1 {
			return makeStreamToolCallEvents("call-1", "echo", `{"text":"Hello, world!"}`), nil
		}
		return makeStreamEvents("I echoed the greeting for you."), nil
	}

	newMsgs, err := simulateTurn(t, mgr, memStore, registry, "say hello", streamFn)
	if err != nil {
		t.Fatalf("simulateTurn: %v", err)
	}

	// Expect: assistant(tool_call) → tool(result) → assistant(text)
	if len(newMsgs) < 3 {
		t.Fatalf("expected at least 3 new messages, got %d", len(newMsgs))
	}

	if newMsgs[0].Role != "assistant" || len(newMsgs[0].ToolCalls) == 0 {
		t.Errorf("msg[0]: expected assistant with tool calls, got role=%s toolCalls=%d",
			newMsgs[0].Role, len(newMsgs[0].ToolCalls))
	}
	if newMsgs[1].Role != "tool" || newMsgs[1].Content != "Hello, world!" {
		t.Errorf("msg[1]: expected tool result 'Hello, world!', got role=%s content=%q",
			newMsgs[1].Role, newMsgs[1].Content)
	}
	if newMsgs[2].Role != "assistant" || newMsgs[2].Content == "" {
		t.Errorf("msg[2]: expected assistant with content, got role=%s content=%q",
			newMsgs[2].Role, newMsgs[2].Content)
	}

	// Memory stores combined assistant content.
	if memStore.Count() != 2 {
		t.Errorf("expected 2 memories, got %d", memStore.Count())
	}
}

func TestE2EMultiTurnMemoryAccumulation(t *testing.T) {
	mgr, memStore, registry := newTestSetup(t)
	defer memStore.Close()

	// Turn 1: ask a question, get an answer.
	streamFn1 := func(_ context.Context, _ *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		return makeStreamEvents("Go is compiled with go build."), nil
	}
	if _, err := simulateTurn(t, mgr, memStore, registry, "how do I compile Go?", streamFn1); err != nil {
		t.Fatalf("turn 1: %v", err)
	}
	if memStore.Count() != 1 {
		t.Fatalf("expected 1 memory after turn 1, got %d", memStore.Count())
	}

	// Turn 2: ask a related question — memory from turn 1 should be injected.
	streamFn2 := func(_ context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		foundMemory := false
		for _, m := range req.Messages {
			if m.Role == "system" && strings.Contains(m.Content, "[Memory from") {
				foundMemory = true
				break
			}
		}
		if !foundMemory {
			t.Error("expected memory from turn 1 in turn 2 context")
		}
		return makeStreamEvents("Use go test to run tests."), nil
	}
	if _, err := simulateTurn(t, mgr, memStore, registry, "how do I test Go code?", streamFn2); err != nil {
		t.Fatalf("turn 2: %v", err)
	}
	if memStore.Count() != 2 {
		t.Errorf("expected 2 memories after turn 2, got %d", memStore.Count())
	}
}

func TestE2ENoMemory(t *testing.T) {
	estimator := chatctx.NewTokenEstimator()
	mgr := chatctx.NewManager(chatctx.Config{
		CtxSize:        4096,
		ResponseBudget: 512,
	}, estimator)
	mgr.SetSystemPrompt("You are a helpful assistant.")

	registry := tools.NewRegistry()
	registry.Register(&echoTool{})

	streamFn := func(_ context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		for _, m := range req.Messages {
			if m.Role == "system" && strings.Contains(m.Content, "[Memory from") {
				t.Error("unexpected memory message in no-memory mode")
			}
		}
		return makeStreamEvents("I can help with that."), nil
	}

	newMsgs, err := simulateTurn(t, mgr, nil, registry, "hello there", streamFn)
	if err != nil {
		t.Fatalf("simulateTurn: %v", err)
	}

	if len(newMsgs) == 0 {
		t.Fatal("expected at least one new message")
	}
	if extractAssistantContent(newMsgs) == "" {
		t.Error("expected non-empty assistant content")
	}
}

func TestE2EEmptyMemorySearch(t *testing.T) {
	mgr, memStore, registry := newTestSetup(t)
	defer memStore.Close()

	// Store is empty — search should return nothing, ClearMemories is called.
	streamFn := func(_ context.Context, req *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		for _, m := range req.Messages {
			if m.Role == "system" && strings.Contains(m.Content, "[Memory from") {
				t.Error("unexpected memory message with empty store")
			}
		}
		return makeStreamEvents("Here is my response."), nil
	}

	newMsgs, err := simulateTurn(t, mgr, memStore, registry, "tell me something", streamFn)
	if err != nil {
		t.Fatalf("simulateTurn: %v", err)
	}

	if len(newMsgs) == 0 {
		t.Fatal("expected at least one new message")
	}

	// New memory should be stored from this turn.
	if memStore.Count() != 1 {
		t.Errorf("expected 1 memory after turn, got %d", memStore.Count())
	}
}

func TestE2EContextWindowWithMemory(t *testing.T) {
	memStore, err := memory.NewChromemStoreInMemory(mockEmbedFunc)
	if err != nil {
		t.Fatalf("create memory store: %v", err)
	}
	defer memStore.Close()

	estimator := chatctx.NewTokenEstimator()
	mgr := chatctx.NewManager(chatctx.Config{
		CtxSize:        500,
		ResponseBudget: 100,
	}, estimator)
	mgr.SetSystemPrompt("You are a helpful assistant.")

	registry := tools.NewRegistry()
	registry.Register(&echoTool{})

	ctx := context.Background()

	// Pre-seed multiple memories.
	for i := 0; i < 3; i++ {
		if err := memStore.Add(ctx, memory.Entry{
			UserMsg:   fmt.Sprintf("question %d about coding", i),
			AssistMsg: fmt.Sprintf("answer %d about coding practices", i),
			Timestamp: time.Now().Add(time.Duration(-i) * time.Hour),
		}); err != nil {
			t.Fatalf("seed memory %d: %v", i, err)
		}
	}

	// Fill history so windowing kicks in.
	for i := 0; i < 10; i++ {
		mgr.Append(api.Message{
			Role:    "user",
			Content: fmt.Sprintf("This is message %d with some padding text to use tokens", i),
		})
		mgr.Append(api.Message{
			Role:    "assistant",
			Content: fmt.Sprintf("Response to message %d with additional padding content here", i),
		})
	}

	streamFn := func(_ context.Context, _ *api.ChatCompletionRequest) (<-chan runner.StreamEvent, error) {
		return makeStreamEvents("Windowed response."), nil
	}

	if _, err := simulateTurn(t, mgr, memStore, registry, "coding question", streamFn); err != nil {
		t.Fatalf("simulateTurn: %v", err)
	}

	budget := mgr.Budget()
	if budget.Memory == 0 {
		t.Error("expected non-zero Memory in budget")
	}
	if budget.HistoryCount >= budget.TotalHistory {
		t.Errorf("expected windowed history (%d) < total history (%d)",
			budget.HistoryCount, budget.TotalHistory)
	}

	// Memories should appear in the windowed output.
	foundMemory := false
	for _, m := range mgr.Messages() {
		if m.Role == "system" && strings.Contains(m.Content, "[Memory from") {
			foundMemory = true
			break
		}
	}
	if !foundMemory {
		t.Error("expected memory messages in windowed context")
	}
}

func TestE2EREPLMemoryCommands(t *testing.T) {
	memStore, err := memory.NewChromemStoreInMemory(mockEmbedFunc)
	if err != nil {
		t.Fatalf("create memory store: %v", err)
	}
	defer memStore.Close()

	ctx := context.Background()

	// Seed known entries.
	for i := 0; i < 3; i++ {
		if err := memStore.Add(ctx, memory.Entry{
			ID:        fmt.Sprintf("test-id-%d-suffix", i),
			UserMsg:   fmt.Sprintf("question %d about Go", i),
			AssistMsg: fmt.Sprintf("answer %d about Go", i),
			Timestamp: time.Now().Add(time.Duration(-i) * time.Hour),
		}); err != nil {
			t.Fatalf("seed entry %d: %v", i, err)
		}
	}

	// List: should return all 3 entries.
	entries, err := memStore.List(ctx, 10)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(entries) != 3 {
		t.Errorf("list: expected 3 entries, got %d", len(entries))
	}

	// Search: should return results.
	results, err := memStore.Search(ctx, "Go programming", 5)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) == 0 {
		t.Error("search: expected results")
	}

	// Delete (forget): remove one entry.
	if err := memStore.Delete(ctx, "test-id-0-suffix"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if memStore.Count() != 2 {
		t.Errorf("after delete: expected 2, got %d", memStore.Count())
	}

	// Clear: remove all remaining entries.
	if err := memStore.Clear(ctx); err != nil {
		t.Fatalf("clear: %v", err)
	}
	if memStore.Count() != 0 {
		t.Errorf("after clear: expected 0, got %d", memStore.Count())
	}
}
