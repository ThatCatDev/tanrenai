package apiclient

import (
	"strings"
	"testing"
)

func TestParseSSEStream_BasicContent(t *testing.T) {
	input := strings.NewReader(
		"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n" +
			"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"}}]}\n\n" +
			"data: [DONE]\n\n",
	)

	events := ParseSSEStream(input)

	var chunks []string
	var doneReceived bool
	for ev := range events {
		if ev.Err != nil {
			t.Fatalf("unexpected error: %v", ev.Err)
		}
		if ev.Done {
			doneReceived = true
			continue
		}
		if ev.Chunk != nil {
			for _, choice := range ev.Chunk.Choices {
				if choice.Delta.Content != "" {
					chunks = append(chunks, choice.Delta.Content)
				}
			}
		}
	}

	if !doneReceived {
		t.Error("expected Done event")
	}
	got := strings.Join(chunks, "")
	if got != "Hello world" {
		t.Errorf("content = %q, want %q", got, "Hello world")
	}
}

func TestParseSSEStream_SkipsNonDataLines(t *testing.T) {
	input := strings.NewReader(
		": this is a comment\n" +
			"event: ping\n" +
			"\n" +
			"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n" +
			"data: [DONE]\n\n",
	)

	events := ParseSSEStream(input)

	var contentCount int
	for ev := range events {
		if ev.Err != nil {
			t.Fatalf("unexpected error: %v", ev.Err)
		}
		if ev.Chunk != nil {
			contentCount++
		}
	}
	if contentCount != 1 {
		t.Errorf("expected 1 content chunk, got %d", contentCount)
	}
}

func TestParseSSEStream_InvalidJSON(t *testing.T) {
	input := strings.NewReader(
		"data: {invalid json}\n\n",
	)

	events := ParseSSEStream(input)

	ev := <-events
	if ev.Err == nil {
		t.Fatal("expected error for invalid JSON")
	}

	// Channel should be closed after error
	_, ok := <-events
	if ok {
		t.Error("expected channel to be closed after error")
	}
}

func TestParseSSEStream_EmptyStream(t *testing.T) {
	input := strings.NewReader("")

	events := ParseSSEStream(input)

	var count int
	for range events {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 events from empty stream, got %d", count)
	}
}

func TestParseSSEStream_DoneOnly(t *testing.T) {
	input := strings.NewReader("data: [DONE]\n\n")

	events := ParseSSEStream(input)

	ev := <-events
	if !ev.Done {
		t.Error("expected Done event")
	}

	_, ok := <-events
	if ok {
		t.Error("expected channel to be closed after DONE")
	}
}

func TestAccumulateResponse_BasicContent(t *testing.T) {
	input := strings.NewReader(
		"data: {\"id\":\"resp-1\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n" +
			"data: {\"id\":\"resp-1\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"}}]}\n\n" +
			"data: [DONE]\n\n",
	)

	events := ParseSSEStream(input)
	resp, err := AccumulateResponse(events)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.ID != "resp-1" {
		t.Errorf("ID = %q, want %q", resp.ID, "resp-1")
	}
	if resp.Model != "test-model" {
		t.Errorf("Model = %q, want %q", resp.Model, "test-model")
	}
	if resp.Object != "chat.completion" {
		t.Errorf("Object = %q, want %q", resp.Object, "chat.completion")
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}
	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("Role = %q, want %q", resp.Choices[0].Message.Role, "assistant")
	}
	if resp.Choices[0].Message.Content != "Hello world" {
		t.Errorf("Content = %q, want %q", resp.Choices[0].Message.Content, "Hello world")
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", resp.Choices[0].FinishReason, "stop")
	}
}

func TestAccumulateResponse_WithFinishReason(t *testing.T) {
	stop := "stop"
	input := strings.NewReader(
		"data: {\"id\":\"r1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"done\"}}]}\n\n" +
			"data: {\"id\":\"r1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"" + stop + "\"}]}\n\n" +
			"data: [DONE]\n\n",
	)

	events := ParseSSEStream(input)
	resp, err := AccumulateResponse(events)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", resp.Choices[0].FinishReason, "stop")
	}
}

func TestAccumulateResponse_ToolCalls(t *testing.T) {
	input := strings.NewReader(
		"data: {\"id\":\"r1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"file_read\",\"arguments\":\"\"}}]}}]}\n\n" +
			"data: {\"id\":\"r1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\"\"}}]}}]}\n\n" +
			"data: {\"id\":\"r1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\": \\\"test.txt\\\"}\"}}]}}]}\n\n" +
			"data: [DONE]\n\n",
	)

	events := ParseSSEStream(input)
	resp, err := AccumulateResponse(events)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.Choices[0].Message.ToolCalls))
	}

	tc := resp.Choices[0].Message.ToolCalls[0]
	if tc.ID != "call_1" {
		t.Errorf("ToolCall.ID = %q, want %q", tc.ID, "call_1")
	}
	if tc.Type != "function" {
		t.Errorf("ToolCall.Type = %q, want %q", tc.Type, "function")
	}
	if tc.Function.Name != "file_read" {
		t.Errorf("ToolCall.Function.Name = %q, want %q", tc.Function.Name, "file_read")
	}
	wantArgs := `{"path": "test.txt"}`
	if tc.Function.Arguments != wantArgs {
		t.Errorf("ToolCall.Function.Arguments = %q, want %q", tc.Function.Arguments, wantArgs)
	}

	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, want %q", resp.Choices[0].FinishReason, "tool_calls")
	}
}

func TestAccumulateResponse_DefaultRole(t *testing.T) {
	// No role delta sent -- should default to "assistant"
	input := strings.NewReader(
		"data: {\"id\":\"r1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
			"data: [DONE]\n\n",
	)

	events := ParseSSEStream(input)
	resp, err := AccumulateResponse(events)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("Role = %q, want %q", resp.Choices[0].Message.Role, "assistant")
	}
}

func TestAccumulateResponse_Error(t *testing.T) {
	input := strings.NewReader(
		"data: {invalid json}\n\n",
	)

	events := ParseSSEStream(input)
	_, err := AccumulateResponse(events)
	if err == nil {
		t.Fatal("expected error from invalid JSON in stream")
	}
}

func TestAccumulateResponse_MultipleToolCalls(t *testing.T) {
	input := strings.NewReader(
		"data: {\"id\":\"r1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"tool_a\",\"arguments\":\"{\\\"x\\\":1}\"}}]}}]}\n\n" +
			"data: {\"id\":\"r1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"c2\",\"type\":\"function\",\"function\":{\"name\":\"tool_b\",\"arguments\":\"{\\\"y\\\":2}\"}}]}}]}\n\n" +
			"data: [DONE]\n\n",
	)

	events := ParseSSEStream(input)
	resp, err := AccumulateResponse(events)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Choices[0].Message.ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(resp.Choices[0].Message.ToolCalls))
	}
	if resp.Choices[0].Message.ToolCalls[0].Function.Name != "tool_a" {
		t.Errorf("tool 0 name = %q, want %q", resp.Choices[0].Message.ToolCalls[0].Function.Name, "tool_a")
	}
	if resp.Choices[0].Message.ToolCalls[1].Function.Name != "tool_b" {
		t.Errorf("tool 1 name = %q, want %q", resp.Choices[0].Message.ToolCalls[1].Function.Name, "tool_b")
	}
}
