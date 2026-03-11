package ui

import (
	"context"
	"strings"
	"time"

	"github.com/diamondburned/gotk4/pkg/glib/v2"
	"github.com/diamondburned/gotk4/pkg/gtk/v4"
	"github.com/diamondburned/gotk4/pkg/pango"

	"github.com/ThatCatDev/tanrenai/shared/agent"
	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

const defaultAgentSystemPrompt = `You are a helpful assistant with access to tools for interacting with the filesystem and system.

Important rules:
1. Gather information with tools BEFORE answering. Do not guess or speculate when you can look it up.
2. Call tools directly — do not narrate what you plan to do. Just do it.
3. Use multiple tool calls in one response when possible.
4. Complete multi-step tasks automatically without stopping for confirmation.
5. Only use shell_exec when no other tool fits. Prefer file_read, list_dir, grep_search, find_files.
6. Use "." for the current directory. Never use placeholder names.
7. If a tool call fails, try different arguments. Never repeat an identical failing call.
8. To edit existing files, use patch_file. Only use file_write for creating new files or when you need to rewrite the entire file. Always use file_read first to understand what you're changing.
9. After making changes, verify your work by building or running tests with shell_exec.`

func (a *App) runAgentTurn(ctx context.Context) {
	registry := tools.DefaultRegistry()

	// Build messages with system prompt
	var msgs []api.Message
	msgs = append(msgs, api.Message{Role: "system", Content: defaultAgentSystemPrompt})
	msgs = append(msgs, a.activeSession.Messages...)

	// Streaming state
	var streamingLabel *gtk.Label
	var streamBuf strings.Builder
	var thinkingBox *gtk.Box

	streamFn := func(ctx context.Context, req *api.ChatCompletionRequest) (<-chan apiclient.StreamEvent, error) {
		req.Model = a.selectedModel
		return a.client.StreamCompletion(ctx, req)
	}

	cfg := agent.StreamingConfig{
		Config: agent.Config{
			MaxIterations:     200,
			Tools:             registry,
			MaxResponseTokens: 4096,
			Hooks: agent.Hooks{
				OnToolCall: func(call api.ToolCall) {
					glib.IdleAdd(func() {
						if thinkingBox != nil {
							a.messageList.Remove(thinkingBox)
							thinkingBox = nil
						}
						if streamingLabel != nil {
							a.messageList.Remove(streamingLabel)
							streamingLabel = nil
							streamBuf.Reset()
						}
						w := toolCallWidget(call)
						a.messageList.Append(w)
						a.scrollToBottom()
					})
				},
				OnToolResult: func(call api.ToolCall, result string) {
					glib.IdleAdd(func() {
						w := toolResultWidget(call, result)
						a.messageList.Append(w)
						a.scrollToBottom()
					})
				},
				OnAssistantMessage: func(content string) {
					// Content already handled via OnContentDelta
				},
			},
		},
		OnIterationStart: func(iteration, maxIterations int, messages []api.Message) {
			// No-op for GUI
		},
		OnThinking: func() {
			glib.IdleAdd(func() {
				thinkingBox = thinkingWidget()
				a.messageList.Append(thinkingBox)
				a.scrollToBottom()
			})
		},
		OnThinkingDone: func() {
			glib.IdleAdd(func() {
				if thinkingBox != nil {
					a.messageList.Remove(thinkingBox)
					thinkingBox = nil
				}
			})
		},
		OnContentDelta: func(delta string) {
			glib.IdleAdd(func() {
				streamBuf.WriteString(delta)
				if streamingLabel == nil {
					streamingLabel = gtk.NewLabel("")
					streamingLabel.SetWrap(true)
					streamingLabel.SetWrapMode(pango.WrapWordChar)
					streamingLabel.SetXAlign(0)
					streamingLabel.SetSelectable(true)
					streamingLabel.AddCSSClass("assistant-message")
					streamingLabel.SetHAlign(gtk.AlignStart)
					a.messageList.Append(streamingLabel)
				}
				streamingLabel.SetMarkup(markdownToPango(streamBuf.String()))
				a.scrollToBottom()
			})
		},
	}

	result, err := agent.RunStreaming(ctx, streamFn, msgs, cfg)

	glib.IdleAdd(func() {
		if thinkingBox != nil {
			a.messageList.Remove(thinkingBox)
			thinkingBox = nil
		}

		if err != nil {
			if !strings.Contains(err.Error(), "context canceled") {
				a.showToast("Error: " + err.Error())
			}
		}

		if streamingLabel != nil {
			content := streamBuf.String()
			if content != "" {
				streamingLabel.SetMarkup(markdownToPango(content))
			}
			streamingLabel = nil
			streamBuf.Reset()
		}

		// Update active session messages from agent result (skip system prompt)
		if a.activeSession != nil && len(result) > 1 {
			a.activeSession.Messages = result[1:]
			a.activeSession.UpdatedAt = time.Now()

			// Update sidebar title if this was the first response
			if a.activeSession.Title == "New Chat" {
				a.activeSession.AutoTitle()
				a.updateChatHeader()
				a.updateSidebarTitle()
			}

			go a.saveSessionsAsync()
		}

		a.finishGenerating()
	})
}
