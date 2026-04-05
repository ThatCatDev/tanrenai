package cmd

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"time"

	"github.com/rivo/tview"

	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/shared/agent"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
	"github.com/ThatCatDev/tanrenai/shared/scrolls"
	"github.com/ThatCatDev/tanrenai/shared/tools"
)

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

	content, streamErr := streamSimpleChat(events, chatStreamHooks{
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
			t.app.QueueUpdateDraw(func() {
				t.streaming.WriteString(delta)
				t.updateStreamingLine()
				t.refreshChatView()
			})
		},
	})
	turnCancel()
	t.mu.Lock()
	t.turnCancel = nil
	t.mu.Unlock()

	t.app.QueueUpdateDraw(func() {
		t.handleStreamDone(content, streamErr)
	})
}

func (t *tuiApp) handleStreamDone(content string, err error) {
	t.recordIterationEnd()
	t.stopProgressTicker()
	t.processing = false
	t.statusText = ""

	if err != nil {
		if errors.Is(err, context.Canceled) {
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
		cleaned := stripToolCallXML(content)
		rendered := fmt.Sprintf(" [purple::b] * [-:-:-]%s", t.renderMarkdown(cleaned))
		t.lines = append(t.lines[:streamStart], strings.Split(rendered, "\n")...)
	}

	t.streaming.Reset()
	t.addLine("")
	t.refreshChatView()
	t.updateStatusBar()
}

func (t *tuiApp) handleTurnDone(result, windowedMsgs []api.Message, err error) {
	t.recordIterationEnd()
	t.stopProgressTicker()
	t.processing = false
	t.statusText = ""
	t.liveCtxTokens = 0

	if err != nil {
		if errors.Is(err, context.Canceled) {
			t.addLine("[gray::-]  [interrupted][-:-:-]")
		} else {
			t.addLine(fmt.Sprintf("[gray::-]  Error: %v[-:-:-]", err))
		}
	}

	if len(result) > len(windowedMsgs) {
		newMsgs := result[len(windowedMsgs):]
		t.mgr.AppendMany(newMsgs)
		t.displayFinalContent(newMsgs)
		t.maybePersistMemory(newMsgs)
	}

	t.addLine("")
	t.refreshChatView()
	t.updateStatusBar()
}

// displayFinalContent renders the last assistant message from newMsgs to the chat view.
func (t *tuiApp) displayFinalContent(newMsgs []api.Message) {
	var finalContent string
	for i := len(newMsgs) - 1; i >= 0; i-- {
		if newMsgs[i].Role == "assistant" && newMsgs[i].Content != "" {
			finalContent = newMsgs[i].Content

			break
		}
	}
	finalContent = stripToolCallXML(finalContent)
	if finalContent == "" {
		return
	}
	formatted := fmt.Sprintf(" [purple::b] * [-:-:-]%s", t.renderMarkdown(finalContent))
	for _, line := range strings.Split(formatted, "\n") {
		t.addLine(line)
	}
}

// maybePersistMemory stores a memory entry asynchronously when memory is enabled.
func (t *tuiApp) maybePersistMemory(newMsgs []api.Message) {
	if !t.memoryEnabled {
		return
	}

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

	if assistContent == "" || userInput == "" {
		return
	}

	client := t.client
	t.memoryWg.Add(1)
	go func() {
		defer t.memoryWg.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := client.MemoryStore(ctx, userInput, assistContent); err != nil {
			slog.Error("failed to store memory", "error", err)
		}
	}()
}

// displayToolResult renders a tool result (diff or output preview) into the chat view.
// Must be called from within a QueueUpdateDraw callback.
func (t *tuiApp) displayToolResult(result *tools.ToolResult) {
	if result.Diff != "" {
		t.displayDiff(result.Diff)

		return
	}

	preview := strings.TrimSpace(result.Output)
	preview = strings.Join(strings.Fields(preview), " ")
	if len(preview) > 120 {
		preview = preview[:120] + "..."
	}
	idx := len(t.lines)
	t.addLine("[gray::-]      " + tview.Escape(preview) + "[-:-:-]")
	t.toolResults[idx] = result.Output
}

// displayDiff renders a unified diff into the chat view with syntax colouring.
func (t *tuiApp) displayDiff(diff string) {
	for _, line := range strings.Split(strings.TrimRight(diff, "\n"), "\n") {
		if len(line) == 0 {
			continue
		}
		escaped := tview.Escape(line)
		switch line[0] {
		case '+':
			if strings.HasPrefix(line, "+++") {
				continue
			}
			t.addLine("[green:#1a3a1a:-]      " + escaped + "[-:-:-]")
		case '-':
			if strings.HasPrefix(line, "---") {
				continue
			}
			t.addLine("[red:#3a1a1a:-]      " + escaped + "[-:-:-]")
		case '@':
			t.addLine("[#6688cc::-]      " + escaped + "[-:-:-]")
		default:
			t.addLine("[gray::-]      " + escaped + "[-:-:-]")
		}
	}
}

// ── Planned Agent Turn ──────────────────────────────────────────────────

func (t *tuiApp) startPlannedAgentTurn(input string) {
	t.mgr.Append(api.Message{Role: "user", Content: input})

	if t.scrollsEnabled {
		matched := scrolls.Match(t.allScrolls, input, 3)
		if len(matched) > 0 {
			var scrollMsgs []api.Message
			var names []string
			for _, s := range matched {
				content := fmt.Sprintf("[Scroll: %s]\n%s", s.Name, s.Content)
				scrollMsgs = append(scrollMsgs, api.Message{Role: "system", Content: content})
				names = append(names, s.Name)
			}
			t.mgr.SetScrolls(scrollMsgs)
			t.addLine(fmt.Sprintf("[gray::-]  [scrolls matched: %s][-:-:-]", strings.Join(names, ", ")))
		} else {
			t.mgr.ClearScrolls()
		}
	}

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

	// Create user injection channel for mid-turn input
	userInputCh := make(chan string, 8)
	t.app.QueueUpdateDraw(func() {
		t.userInputCh = userInputCh
		t.plannedMode = true
	})

	toolCount := 0
	var contentFilt xmlFilter
	var reasoningBuf strings.Builder

	flushReasoning := func() {
		if reasoningBuf.Len() > 0 {
			text := reasoningBuf.String()
			reasoningBuf.Reset()
			t.app.QueueUpdateDraw(func() {
				trimmed := strings.TrimSpace(text)
				if trimmed != "" {
					for _, line := range strings.Split(trimmed, "\n") {
						t.addLine("[gray::-]    " + tview.Escape(line) + "[-:-:-]")
					}
					t.refreshChatView()
				}
			})
		}
	}

	flushContent := func() {
		flushReasoning()
		if contentFilt.len() > 0 {
			text := contentFilt.string()
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

	cfg := agent.PlanAgentConfig{
		StreamingConfig: agent.StreamingConfig{
			Config: agent.Config{
				MaxIterations:  t.maxIterations,
				EnableThinking: t.enableThinking,
				Tools:          t.registry,
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
							switch name {
							case "file_read", "file_write", "patch_file":
								path := extractFilePath(call)
								label := "[gray::-]    > " + tview.Escape(display) + " [-:-:-][#00afff::u]" + tview.Escape(path) + "[::U][-:-:-]"
								t.addLine(label)
								t.toolCallLines[idx] = call
							case "shell_exec":
								cmd := extractShellCommand(call)
								label := "[gray::-]    > " + tview.Escape(display) + " [-:-:-][yellow::-]$ " + tview.Escape(cmd) + "[-:-:-]"
								t.addLine(label)
							default:
								t.addLine("[gray::-]    > " + tview.Escape(display) + "[-:-:-]")
							}
							t.refreshChatView()
						})
					},
					OnToolResult: func(call api.ToolCall, result *tools.ToolResult) {
						t.app.QueueUpdateDraw(func() {
							t.displayToolResult(result)
							t.refreshChatView()
						})
					},
					OnAssistantMessage: func(content string) {},
					OnToolApproval: func(call api.ToolCall) agent.ApprovalAction {
						return t.approveToolCall(call)
					},
				},
			},
			OnIterationStart: func(iteration, maxIter int, messages []api.Message) {
				flushContent()
				t.app.QueueUpdateDraw(func() {
					t.recordIterationEnd()
					inputTokens := t.mgr.Estimator().EstimateMessages(messages)
					t.currentIterTokens = inputTokens
					t.lastInputTokens = inputTokens
					t.liveCtxTokens = inputTokens
					t.currentIterOutput = 0
					t.startProgressTicker()
					t.iterStartTime = time.Now()
					t.estimatedDur = t.predictDuration(inputTokens)
					t.updateStatusBar()
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
				contentFilt.write(delta)
			},
			OnReasoningDelta: func(delta string) {
				t.currentIterOutput += len(delta)
				reasoningBuf.WriteString(delta)
			},
			UserInput: userInputCh,
		},
		UserInput: userInputCh,
		OnPlanningStart: func() {
			t.app.QueueUpdateDraw(func() {
				t.statusText = "Planning..."
				t.updateStatusBar()
				t.addLine("[blue::b]  -- planning --[-:-:-]")
				t.refreshChatView()
			})
		},
		OnPlanGenerated: func(plan *agent.Plan) {
			t.app.QueueUpdateDraw(func() {
				t.addLine("[blue::b]  Plan:[-:-:-]")
				for _, step := range plan.Steps {
					t.addLine(fmt.Sprintf("[gray::-]    %d. %s[-:-:-]", step.Index, tview.Escape(step.Description)))
				}
				t.addLine("")
				t.refreshChatView()
			})
		},
		OnStepStart: func(stepIdx int, step *agent.PlanStep) {
			flushContent()
			t.app.QueueUpdateDraw(func() {
				t.addLine(fmt.Sprintf("[yellow::b]  -- step %d: %s --[-:-:-]", step.Index, tview.Escape(step.Description)))
				t.refreshChatView()
				t.statusText = fmt.Sprintf("step %d/%d", step.Index, stepIdx+1)
				t.updateStatusBar()
			})
		},
		OnStepDone: func(stepIdx int, step *agent.PlanStep) {
			flushContent()
			t.app.QueueUpdateDraw(func() {
				status := step.Status.String()
				color := "green"
				switch step.Status {
				case agent.StepFailed:
					color = "red"
				case agent.StepSkipped:
					color = "gray"
				default:
					// other statuses keep default color
				}
				t.addLine(fmt.Sprintf("[%s::-]  -- step %d: %s --[-:-:-]", color, step.Index, status))
				t.addLine("")
				t.refreshChatView()
			})
		},
		OnReplan: func(reason string, newPlan *agent.Plan) {
			t.app.QueueUpdateDraw(func() {
				t.addLine("[yellow::b]  Re-planned:[-:-:-]")
				for _, step := range newPlan.Steps {
					statusMark := " "
					switch step.Status {
					case agent.StepDone:
						statusMark = "[green::-]✓[-:-:-]"
					case agent.StepFailed:
						statusMark = "[red::-]✗[-:-:-]"
					default:
						// other statuses keep default mark
					}
					t.addLine(fmt.Sprintf("    %s %d. %s", statusMark, step.Index, tview.Escape(step.Description)))
				}
				t.addLine("")
				t.refreshChatView()
			})
		},
		OnSynthesize: func() {
			flushContent()
			t.app.QueueUpdateDraw(func() {
				t.addLine("[blue::b]  -- synthesizing --[-:-:-]")
				t.refreshChatView()
				t.statusText = "Synthesizing..."
				t.updateStatusBar()
			})
		},
	}

	result, err := agent.RunPlannedStreaming(turnCtx, t.streamFn, windowedMsgs, cfg)
	flushContent()
	turnCancel()
	t.mu.Lock()
	t.turnCancel = nil
	t.mu.Unlock()

	t.app.QueueUpdateDraw(func() {
		t.userInputCh = nil
		t.plannedMode = false
		t.handleTurnDone(result, windowedMsgs, err)
	})
}

// ── Swarm Agent Turn ──────────────────────────────────────────────────

func (t *tuiApp) startSwarmAgentTurn(input string) {
	t.mgr.Append(api.Message{Role: "user", Content: input})

	if t.scrollsEnabled {
		matched := scrolls.Match(t.allScrolls, input, 3)
		if len(matched) > 0 {
			var scrollMsgs []api.Message
			var names []string
			for _, s := range matched {
				content := fmt.Sprintf("[Scroll: %s]\n%s", s.Name, s.Content)
				scrollMsgs = append(scrollMsgs, api.Message{Role: "system", Content: content})
				names = append(names, s.Name)
			}
			t.mgr.SetScrolls(scrollMsgs)
			t.addLine(fmt.Sprintf("[gray::-]  [scrolls matched: %s][-:-:-]", strings.Join(names, ", ")))
		} else {
			t.mgr.ClearScrolls()
		}
	}

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

	var contentFilt xmlFilter
	var reasoningBuf strings.Builder

	flushReasoning := func() {
		if reasoningBuf.Len() > 0 {
			text := reasoningBuf.String()
			reasoningBuf.Reset()
			t.app.QueueUpdateDraw(func() {
				trimmed := strings.TrimSpace(text)
				if trimmed != "" {
					for _, line := range strings.Split(trimmed, "\n") {
						t.addLine("[gray::-]    " + tview.Escape(line) + "[-:-:-]")
					}
					t.refreshChatView()
				}
			})
		}
	}

	flushContent := func() {
		flushReasoning()
		if contentFilt.len() > 0 {
			text := contentFilt.string()
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

	// Worker tools: filesystem + shell.
	var workerTools *tools.Registry
	if t.registry != nil {
		workerTools = t.registry.Subset(
			"file_read", "file_write", "patch_file",
			"list_dir", "grep_search", "shell_exec",
		)
	}

	cfg := agent.SwarmConfig{
		StreamingConfig: agent.StreamingConfig{
			Config: agent.Config{
				MaxIterations:  t.maxIterations,
				EnableThinking: t.enableThinking,
				Tools:          t.registry,
				Hooks: agent.Hooks{
					OnToolCall: func(call api.ToolCall) {
						flushContent()
						name := call.Function.Name
						t.app.QueueUpdateDraw(func() {
							t.statusText = name
							t.updateStatusBar()
							switch name {
							case "file_read", "file_write", "patch_file":
								path := extractFilePath(call)
								t.addLine("[gray::-]    > " + tview.Escape(name) + " [-:-:-][#00afff::u]" + tview.Escape(path) + "[::U][-:-:-]")
							case "shell_exec":
								cmd := extractShellCommand(call)
								t.addLine("[gray::-]    > " + tview.Escape(name) + " [-:-:-][yellow::-]$ " + tview.Escape(cmd) + "[-:-:-]")
							default:
								t.addLine("[gray::-]    > " + tview.Escape(name) + "[-:-:-]")
							}
							t.refreshChatView()
						})
					},
					OnToolResult: func(call api.ToolCall, result *tools.ToolResult) {
						t.app.QueueUpdateDraw(func() {
							t.displayToolResult(result)
							t.refreshChatView()
						})
					},
					OnAssistantMessage: func(content string) {},
					OnToolApproval: func(call api.ToolCall) agent.ApprovalAction {
						return t.approveToolCall(call)
					},
				},
			},
			OnIterationStart: func(iteration, maxIter int, messages []api.Message) {
				flushContent()
				t.app.QueueUpdateDraw(func() {
					t.startProgressTicker()
					t.iterStartTime = time.Now()
					t.updateStatusBar()
				})
			},
			OnContentDelta: func(delta string) {
				contentFilt.write(delta)
			},
			OnReasoningDelta: func(delta string) {
				reasoningBuf.WriteString(delta)
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
		},
		WorkerTools: workerTools,
		OnPlanGenerated: func(depth int, plan *agent.Plan) {
			t.app.QueueUpdateDraw(func() {
				tree := strings.Repeat("│   ", depth)
				if depth == 0 {
					t.addLine("[blue::b]  Orchestrator[-:-:-]")
				} else {
					t.addLine(fmt.Sprintf("[blue::-]  %s├── Sub-orchestrator[-:-:-]", tree[:len(tree)-4]))
				}
				for i, step := range plan.Steps {
					branch := "├──"
					if i == len(plan.Steps)-1 {
						branch = "└──"
					}
					t.addLine(fmt.Sprintf("[gray::-]  %s%s %d. %s[-:-:-]", tree, branch, step.Index, tview.Escape(step.Description)))
				}
				t.addLine("")
				t.refreshChatView()
			})
		},
		OnWorkerStart: func(depth, stepIdx int, step *agent.PlanStep) {
			flushContent()
			tree := strings.Repeat("│   ", depth)
			t.app.QueueUpdateDraw(func() {
				t.addLine(fmt.Sprintf("[yellow::b]  %s▶ Worker %d: %s[-:-:-]", tree, step.Index, tview.Escape(step.Description)))
				t.refreshChatView()
				t.statusText = fmt.Sprintf("Worker %d", step.Index)
				if depth > 0 {
					t.statusText = fmt.Sprintf("Sub-worker %d", step.Index)
				}
				t.updateStatusBar()
			})
		},
		OnWorkerDone: func(depth, stepIdx int, step *agent.PlanStep) {
			flushContent()
			tree := strings.Repeat("│   ", depth)
			t.app.QueueUpdateDraw(func() {
				icon := "[green::-]✓[-:-:-]"
				if step.Status == agent.StepFailed {
					icon = "[red::-]✗[-:-:-]"
				}
				t.addLine(fmt.Sprintf("  %s%s Worker %d: %s", tree, icon, step.Index, step.Status.String()))
				t.refreshChatView()
			})
		},
		OnVerifyStart: func() {
			flushContent()
			t.app.QueueUpdateDraw(func() {
				t.addLine("[blue::b]  -- verifying --[-:-:-]")
				t.refreshChatView()
				t.statusText = "Verifying..."
				t.updateStatusBar()
			})
		},
	}

	result, err := agent.RunSwarm(turnCtx, t.streamFn, windowedMsgs, cfg)
	flushContent()
	turnCancel()
	t.mu.Lock()
	t.turnCancel = nil
	t.mu.Unlock()

	t.app.QueueUpdateDraw(func() {
		t.handleTurnDone(result, windowedMsgs, err)
	})
}

// ── Progress Tracking ───────────────────────────────────────────────────

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
		t.lastOutputTokens = t.currentIterOutput / 4 // rough char->token
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

// approveToolCall checks permissions and shows a modal if needed.
// Called from the agent goroutine — blocks until user responds.
func (t *tuiApp) approveToolCall(call api.ToolCall) agent.ApprovalAction {
	if t.permissions.IsAllowed(call.Function.Name, call.Function.Arguments) {
		return agent.ApprovalAllow
	}

	responseCh := make(chan agent.ApprovalAction, 1)

	t.app.QueueUpdateDraw(func() {
		toolName := call.Function.Name
		argsJSON := call.Function.Arguments
		risk := tools.ToolRisk(toolName)
		approvalKey := tools.ApprovalKey(toolName)
		keyValue := tools.ExtractArg(argsJSON, approvalKey)

		// Format display text
		displayArgs := argsJSON
		if len(displayArgs) > 300 {
			displayArgs = displayArgs[:300] + "..."
		}

		// Build buttons based on risk level
		var buttons []string
		switch risk {
		case tools.RiskReadOnly:
			buttons = []string{"Allow", "Always Allow", "Block"}
		case tools.RiskWrite:
			if keyValue != "" {
				buttons = []string{"Allow", "Always (path)", "Always (all)", "Block"}
			} else {
				buttons = []string{"Allow", "Always Allow", "Block"}
			}
		case tools.RiskExecute:
			if keyValue != "" {
				prefix := tools.CommandPrefix(keyValue)
				if prefix != "" {
					buttons = []string{"Allow", "Always (" + prefix + " *)", "Block"}
				} else {
					buttons = []string{"Allow", "Block"}
				}
			} else {
				buttons = []string{"Allow", "Block"}
			}
		case tools.RiskNetwork:
			buttons = []string{"Allow", "Always Allow", "Block"}
		default:
			buttons = []string{"Allow", "Block"}
		}

		modal := tview.NewModal().
			SetText(fmt.Sprintf("Tool: [yellow::b]%s[-:-:-]\n\n%s",
				toolName, tview.Escape(displayArgs))).
			AddButtons(buttons).
			SetDoneFunc(func(_ int, label string) {
				t.pages.RemovePage("approval")
				t.app.SetFocus(t.inputArea)
				switch {
				case label == "Always Allow":
					_ = t.permissions.AllowTool(toolName)
					responseCh <- agent.ApprovalAlwaysAllow
				case label == "Always (all)":
					_ = t.permissions.AllowTool(toolName)
					responseCh <- agent.ApprovalAlwaysAllow
				case label == "Always (path)" && keyValue != "":
					_ = t.permissions.AllowToolWithArgs(toolName, map[string][]string{
						approvalKey: {keyValue},
					})
					responseCh <- agent.ApprovalAlwaysAllow
				case strings.HasPrefix(label, "Always (") && strings.HasSuffix(label, " *)"):
					// "Always (git *)" → save prefix rule
					prefix := tools.CommandPrefix(keyValue)
					_ = t.permissions.AllowToolWithArgs(toolName, map[string][]string{
						approvalKey: {prefix + " *"},
					})
					responseCh <- agent.ApprovalAlwaysAllow
				case label == "Block":
					responseCh <- agent.ApprovalBlock
				default:
					responseCh <- agent.ApprovalAllow
				}
			})
		modal.SetBackgroundColor(0)
		t.pages.AddPage("approval", modal, true, true)
		t.app.SetFocus(modal)
	})

	return <-responseCh
}

// toolCallXMLPattern matches <tool_call>...</tool_call> blocks that some models
// emit as text content instead of using native tool calling.
var toolCallXMLPattern = regexp.MustCompile(`(?s)<tool_call>.*?</tool_call>\s*`)

// stripToolCallXML removes XML-style tool call blocks from model output.
// These are already handled by the native tool calling mechanism and
// shouldn't be displayed to the user.
func stripToolCallXML(text string) string {
	cleaned := toolCallXMLPattern.ReplaceAllString(text, "")

	return strings.TrimSpace(cleaned)
}

// xmlFilter tracks state for streaming <tool_call> block removal.
// Call write() instead of buf.WriteString() to filter content in real-time.
type xmlFilter struct {
	out     strings.Builder // clean output (no tool_call blocks)
	inBlock bool
}

// write appends delta content, filtering out <tool_call>...</tool_call> blocks.
func (f *xmlFilter) write(delta string) {
	const openTag = "<tool_call>"
	const closeTag = "</tool_call>"

	if f.inBlock {
		// Inside a block — look for closing tag in the delta
		idx := strings.Index(delta, closeTag)
		if idx < 0 {
			return // discard, still inside block
		}
		f.inBlock = false
		// Process remainder after the closing tag
		rest := strings.TrimLeft(delta[idx+len(closeTag):], "\n\r\t ")
		if rest != "" {
			f.write(rest)
		}

		return
	}

	// Not in block — append delta then check if accumulated output
	// now contains an opening tag (handles tags split across deltas)
	f.out.WriteString(delta)
	cur := f.out.String()

	idx := strings.Index(cur, openTag)
	if idx < 0 {
		return // no opening tag
	}

	// Found opening tag — keep content before it, enter block mode
	f.inBlock = true
	after := cur[idx+len(openTag):]
	f.out.Reset()
	f.out.WriteString(cur[:idx])

	// Check if closing tag is in the same chunk
	closeIdx := strings.Index(after, closeTag)
	if closeIdx < 0 {
		return // waiting for close
	}
	f.inBlock = false
	rest := strings.TrimLeft(after[closeIdx+len(closeTag):], "\n\r\t ")
	if rest != "" {
		f.write(rest) // recurse for more blocks
	}
}

// string returns the accumulated clean content and resets the buffer.
func (f *xmlFilter) string() string {
	s := f.out.String()
	f.out.Reset()

	return s
}

// len returns the length of accumulated clean content.
func (f *xmlFilter) len() int {
	return f.out.Len()
}
