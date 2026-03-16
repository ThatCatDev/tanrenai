package cmd

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/rivo/tview"

	"github.com/ThatCatDev/tanrenai/shared/agent"
	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
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

	if t.scrollsEnabled {
		matched := scrolls.Match(t.allScrolls, input, 3)
		if len(matched) > 0 {
			var scrollMsgs []api.Message
			for _, s := range matched {
				content := fmt.Sprintf("[Scroll: %s]\n%s", s.Name, s.Content)
				scrollMsgs = append(scrollMsgs, api.Message{Role: "system", Content: content})
			}
			t.mgr.SetScrolls(scrollMsgs)
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
			MaxIterations:     t.maxIterations,
			MaxResponseTokens: t.maxResponseTokens,
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
						if result.Diff != "" {
							// Show colored inline diff with background highlights
							for _, line := range strings.Split(strings.TrimRight(result.Diff, "\n"), "\n") {
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
						} else {
							preview := strings.TrimSpace(result.Output)
							preview = strings.Join(strings.Fields(preview), " ")
							if len(preview) > 120 {
								preview = preview[:120] + "..."
							}
							idx := len(t.lines)
							t.addLine("[gray::-]      " + tview.Escape(preview) + "[-:-:-]")
							t.toolResults[idx] = result.Output
						}
						t.refreshChatView()
					})
				},
				OnAssistantMessage: func(content string) {
					// Content already flushed via OnContentDelta; ignore.
				},
				OnToolApproval: func(call api.ToolCall) agent.ApprovalAction {
					return t.approveToolCall(call)
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
				t.liveCtxTokens = inputTokens
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
		OnReasoningDelta: func(delta string) {
			t.currentIterOutput += len(delta)
			t.app.QueueUpdateDraw(func() {
				t.statusText = "Reasoning..."
				t.updateStatusBar()
			})
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
		}
	}

	t.addLine("")
	t.refreshChatView()
	t.updateStatusBar()
}

// ── Planned Agent Turn ──────────────────────────────────────────────────

func (t *tuiApp) startPlannedAgentTurn(input string) {
	t.mgr.Append(api.Message{Role: "user", Content: input})

	if t.scrollsEnabled {
		matched := scrolls.Match(t.allScrolls, input, 3)
		if len(matched) > 0 {
			var scrollMsgs []api.Message
			for _, s := range matched {
				content := fmt.Sprintf("[Scroll: %s]\n%s", s.Name, s.Content)
				scrollMsgs = append(scrollMsgs, api.Message{Role: "system", Content: content})
			}
			t.mgr.SetScrolls(scrollMsgs)
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
	userInputCh := make(chan string, 1)
	t.app.QueueUpdateDraw(func() {
		t.userInputCh = userInputCh
		t.plannedMode = true
	})

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

	cfg := agent.PlanAgentConfig{
		StreamingConfig: agent.StreamingConfig{
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
							if result.Diff != "" {
								for _, line := range strings.Split(strings.TrimRight(result.Diff, "\n"), "\n") {
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
							} else {
								preview := strings.TrimSpace(result.Output)
								preview = strings.Join(strings.Fields(preview), " ")
								if len(preview) > 120 {
									preview = preview[:120] + "..."
								}
								idx := len(t.lines)
								t.addLine("[gray::-]      " + tview.Escape(preview) + "[-:-:-]")
								t.toolResults[idx] = result.Output
							}
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
				contentBuf.WriteString(delta)
			},
			OnReasoningDelta: func(delta string) {
				t.currentIterOutput += len(delta)
				t.app.QueueUpdateDraw(func() {
					t.statusText = "Reasoning..."
					t.updateStatusBar()
				})
			},
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
				if step.Status == agent.StepFailed {
					color = "red"
				} else if step.Status == agent.StepSkipped {
					color = "gray"
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
					if step.Status == agent.StepDone {
						statusMark = "[green::-]✓[-:-:-]"
					} else if step.Status == agent.StepFailed {
						statusMark = "[red::-]✗[-:-:-]"
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
