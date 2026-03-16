package cmd

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"

	"github.com/ThatCatDev/tanrenai/client/internal/chatctx"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

// ── Input Capture ──────────────────────────────────────────────────────

//nolint:gocyclo
func (t *tuiApp) setupInputCapture() {
	t.app.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		// When a modal is showing, let tview handle all input (arrow keys, enter, etc.)
		if name, _ := t.pages.GetFrontPage(); name == "approval" {
			return event
		}

		// Autocomplete popup — only intercept navigation keys, let all typing through
		if t.acActive {
			switch event.Key() { //nolint:exhaustive
			case tcell.KeyTab, tcell.KeyEnter:
				t.acceptAutocomplete()

				return nil
			case tcell.KeyEscape:
				t.dismissAutocomplete()

				return nil
			case tcell.KeyUp:
				cur := t.acList.GetCurrentItem()
				if cur > 0 {
					t.acList.SetCurrentItem(cur - 1)
				}

				return nil
			case tcell.KeyDown:
				cur := t.acList.GetCurrentItem()
				if cur < t.acList.GetItemCount()-1 {
					t.acList.SetCurrentItem(cur + 1)
				}

				return nil
			}
			// Everything else (typing, backspace, enter) flows to TextArea normally
		}

		if event.Key() != tcell.KeyCtrlC {
			t.ctrlCPending = false
		}

		// Process panel has focus — delegate key handling
		if t.focus == focusProcessPanel {
			if t.handleProcessPanelKey(event) {
				return nil
			}
		}

		switch event.Key() { //nolint:exhaustive
		case tcell.KeyCtrlC:
			if t.processing {
				t.mu.Lock()
				if t.turnCancel != nil {
					t.turnCancel()
				}
				t.mu.Unlock()

				return nil
			}
			if t.ctrlCPending {
				t.app.Stop()

				return nil
			}
			t.ctrlCPending = true
			t.addLine("[gray::-]  Press Ctrl+C again to quit.[-:-:-]")
			t.refreshChatView()

			return nil

		case tcell.KeyCtrlD:
			t.app.Stop()

			return nil

		case tcell.KeyEscape:
			if t.focus == focusProcessPanel {
				t.focus = focusChat
				t.app.SetFocus(t.inputArea)

				return nil
			}
			if t.filePath != "" {
				t.closeFileViewer()

				return nil
			}

		case tcell.KeyCtrlP:
			// Toggle process panel
			if t.processCount() > 0 { //nolint:nestif
				if t.processPanel != nil {
					if t.focus == focusProcessPanel {
						t.focus = focusChat
						t.app.SetFocus(t.inputArea)
					} else {
						t.focus = focusProcessPanel
						t.refreshProcessPanel()
					}
				} else {
					t.showProcessPanel()
					t.focus = focusProcessPanel
				}

				return nil
			}

		case tcell.KeyTab:
			if t.filePath != "" {
				if t.focus == focusChat {
					t.focus = focusFileViewer
				} else {
					t.focus = focusChat
				}
				t.rebuildFileViewer()

				return nil
			}
			t.expanded = !t.expanded
			t.refreshChatView()

			return nil

		case tcell.KeyUp:
			t.scrollFocusedPane(-1)

			return nil
		case tcell.KeyDown:
			// Down arrow: if processes exist and panel isn't shown, show it
			if t.focus == focusChat && t.processCount() > 0 && t.processPanel == nil {
				t.showProcessPanel()
				t.focus = focusProcessPanel

				return nil
			}
			t.scrollFocusedPane(1)

			return nil
		case tcell.KeyPgUp:
			t.scrollFocusedPane(-10)

			return nil
		case tcell.KeyPgDn:
			t.scrollFocusedPane(10)

			return nil

		case tcell.KeyEnter:
			if t.loading {
				return nil
			}
			// Mid-turn injection for planned agent mode
			if t.processing && t.userInputCh != nil {
				if event.Modifiers()&tcell.ModShift != 0 {
					return event
				}
				text := strings.TrimSpace(t.inputArea.GetText())
				if text == "" {
					return nil
				}
				t.inputArea.SetText("", false)
				select {
				case t.userInputCh <- text:
					t.addLine(fmt.Sprintf("[yellow::b]>>> %s [gray](injected)[-:-:-]", tview.Escape(text)))
					t.refreshChatView()
				default:
					// channel full, drop
				}

				return nil
			}
			if t.processing {
				return nil
			}
			// Shift+Enter inserts a newline (let TextArea handle it)
			if event.Modifiers()&tcell.ModShift != 0 {
				return event
			}
			text := strings.TrimSpace(t.inputArea.GetText())
			if text == "" {
				return nil
			}
			t.inputArea.SetText("", false)
			t.handleEnter(text)

			return nil
		}

		return event
	})
}

func (t *tuiApp) scrollFocusedPane(delta int) {
	var tv *tview.TextView
	if t.filePath != "" && t.focus == focusFileViewer {
		tv = t.fileView
	} else {
		tv = t.chatView
	}
	if tv == nil {
		return
	}
	row, col := tv.GetScrollOffset()
	newRow := row + delta
	if newRow < 0 {
		newRow = 0
	}
	tv.ScrollTo(newRow, col)
}

// ── Mouse Capture ──────────────────────────────────────────────────────

func (t *tuiApp) setupMouseCapture() {
	t.app.SetMouseCapture(func(event *tcell.EventMouse, action tview.MouseAction) (*tcell.EventMouse, tview.MouseAction) {
		mx, my := event.Position()

		switch action { //nolint:exhaustive
		case tview.MouseScrollUp, tview.MouseScrollDown:
			delta := 3
			if action == tview.MouseScrollUp {
				delta = -3
			}

			// Determine which pane based on X coordinate
			if t.filePath != "" && t.fileView != nil {
				fx, _, fw, _ := t.fileView.GetRect()
				if mx >= fx && mx < fx+fw {
					row, col := t.fileView.GetScrollOffset()
					newRow := row + delta
					if newRow < 0 {
						newRow = 0
					}
					t.fileView.ScrollTo(newRow, col)

					return nil, 0
				}
			}

			cx, _, cw, _ := t.chatView.GetRect()
			if mx >= cx && mx < cx+cw {
				row, col := t.chatView.GetScrollOffset()
				newRow := row + delta
				if newRow < 0 {
					newRow = 0
				}
				t.chatView.ScrollTo(newRow, col)

				return nil, 0
			}

		case tview.MouseLeftClick:
			// Check if click is in chat area for tool call click-to-open
			cx, cy, cw, ch := t.chatView.GetRect()
			if mx >= cx && mx < cx+cw && my >= cy && my < cy+ch { //nolint:nestif
				row, _ := t.chatView.GetScrollOffset()
				displayLine := row + (my - cy)
				logicalLine := t.displayLineToLogicalLine(displayLine)
				if logicalLine >= 0 {
					if call, ok := t.toolCallLines[logicalLine]; ok {
						path := extractFilePath(call)
						if path != "" {
							t.focus = focusFileViewer
							go t.loadFileViewer(path)

							return nil, 0
						}
					}
				}
				t.focus = focusChat
			}

			// Click in file viewer area
			if t.filePath != "" && t.fileView != nil {
				fx, fy, fw, fh := t.fileView.GetRect()
				if mx >= fx && mx < fx+fw && my >= fy && my < fy+fh {
					t.focus = focusFileViewer
				}
			}
		}

		return event, action
	})
}

// ── Enter Handler ──────────────────────────────────────────────────────

func (t *tuiApp) handleEnter(text string) {
	if text == "/quit" || text == "/exit" {
		t.app.Stop()

		return
	}

	if t.handleSlashCommand(text) {
		t.refreshChatView()

		return
	}

	// Catch unknown slash commands — never send them to the LLM
	if strings.HasPrefix(text, "/") {
		t.addLine(fmt.Sprintf(" [blue::b]>>>[white] %s", tview.Escape(text)))
		t.addLine("[gray::-]  Unknown command. Type /help for available commands.[-:-:-]")
		t.addLine("")
		t.refreshChatView()

		return
	}

	t.addLine(fmt.Sprintf(" [blue::b]>>>[white] %s", tview.Escape(text)))
	t.addLine("")
	t.refreshChatView()

	t.processing = true
	t.statusText = "Thinking..."
	t.currentIterTokens = 0
	t.currentIterOutput = 0
	t.lastInputTokens = 0
	t.lastOutputTokens = 0
	t.startProgressTicker()
	t.iterStartTime = time.Now()
	t.estimatedDur = 0
	t.updateStatusBar()
	t.streaming.Reset()

	if t.agentMode {
		go t.startPlannedAgentTurn(text)
	} else {
		go t.startChatTurn(text)
	}
}

// ── Slash Commands ──────────────────────────────────────────────────────

func (t *tuiApp) handleSlashCommand(input string) bool {
	switch {
	case input == "/clear":
		t.mgr.Clear()
		t.lines = nil
		t.toolResults = make(map[int]string)
		t.toolCallLines = make(map[int]api.ToolCall)
		t.closeFileViewer()
		t.addLine("[gray::-]  History cleared.[-:-:-]")
		t.addLine("")

		return true

	case input == "/compact":
		if !t.agentMode {
			t.addLine("[gray::-]  /compact is only available in agent mode.[-:-:-]")
			t.addLine("")

			return true
		}
		if t.mgr.NeedsSummary() {
			t.addLine("[gray::-]  [compacting...][-:-:-]")
			if err := t.mgr.Summarize(context.Background(), chatctx.CompletionFunc(t.completeFn)); err != nil {
				t.addLine(fmt.Sprintf("[gray::-]  Compact failed: %v[-:-:-]", err))
			} else {
				budget := t.mgr.Budget()
				t.addLine(fmt.Sprintf("[gray::-]  Compacted. %d tokens free (%d%%)[-:-:-]",
					budget.Available, budget.Available*100/budget.Total))
			}
		} else {
			t.addLine("[gray::-]  Nothing to compact.[-:-:-]")
		}
		t.addLine("")

		return true

	case input == "/scrolls" || input == "/scrolls list":
		if !t.scrollsEnabled {
			t.addLine("[gray::-]  No scrolls loaded.[-:-:-]")
		} else {
			t.addLine(fmt.Sprintf("[gray::-]  Scrolls (%d):[-:-:-]", len(t.allScrolls)))
			for _, s := range t.allScrolls {
				tags := ""
				if len(s.Tags) > 0 {
					tags = " [" + strings.Join(s.Tags, ", ") + "]"
				}
				t.addLine(fmt.Sprintf("[gray::-]    [yellow::-]%s[-:-:-] (%s)%s — %s[-:-:-]",
					s.Name, s.Source, tags, s.Description))
			}
		}
		t.addLine("")

		return true

	case strings.HasPrefix(input, "/scrolls show "):
		name := strings.TrimSpace(strings.TrimPrefix(input, "/scrolls show "))
		if name == "" {
			t.addLine("[gray::-]  Usage: /scrolls show <name>[-:-:-]")
			t.addLine("")

			return true
		}
		found := false
		for _, s := range t.allScrolls {
			if s.Name == name {
				t.addLine(fmt.Sprintf("[yellow::b]  %s[-:-:-] [gray::-](%s)[-:-:-]", s.Name, s.Source))
				t.addLine(fmt.Sprintf("[gray::-]  %s[-:-:-]", s.Description))
				if len(s.Tags) > 0 {
					t.addLine(fmt.Sprintf("[gray::-]  Tags: %s[-:-:-]", strings.Join(s.Tags, ", ")))
				}
				t.addLine("")
				for _, line := range strings.Split(s.Content, "\n") {
					t.addLine("[gray::-]  " + tview.Escape(line) + "[-:-:-]")
				}
				found = true

				break
			}
		}
		if !found {
			t.addLine(fmt.Sprintf("[gray::-]  No scroll named %q[-:-:-]", name))
		}
		t.addLine("")

		return true

	case input == "/help":
		t.addLine("[gray::-]  Commands:[-:-:-]")
		t.addLine("[gray::-]    /clear              Clear conversation history[-:-:-]")
		t.addLine("[gray::-]    /compact            Summarize to free context[-:-:-]")
		t.addLine("[gray::-]    /tokens             Show token budget[-:-:-]")
		t.addLine("[gray::-]    /context add <path> Load file into context[-:-:-]")
		t.addLine("[gray::-]    /context list       Show loaded files[-:-:-]")
		t.addLine("[gray::-]    /context clear      Remove all context files[-:-:-]")
		t.addLine("[gray::-]    /memory             List recent memories[-:-:-]")
		t.addLine("[gray::-]    /memory search <q>  Search memories[-:-:-]")
		t.addLine("[gray::-]    /memory forget <id> Delete a memory[-:-:-]")
		t.addLine("[gray::-]    /memory clear       Clear all memories[-:-:-]")
		t.addLine("[gray::-]    /scrolls            List loaded scrolls[-:-:-]")
		t.addLine("[gray::-]    /scrolls show <n>   Show a scroll's content[-:-:-]")
		t.addLine("[gray::-]    /quit, /exit        Exit[-:-:-]")
		t.addLine("")

		return true
	}

	var buf strings.Builder
	if handleREPLCommand(&buf, input, t.mgr, t.client, t.memoryEnabled) {
		for _, line := range strings.Split(buf.String(), "\n") {
			if line != "" {
				t.addLine("[gray::-]  " + tview.Escape(line) + "[-:-:-]")
			}
		}
		t.addLine("")

		return true
	}

	return false
}

// ── Autocomplete ────────────────────────────────────────────────────────

// updateAutocomplete is called on every text change. Shows/hides the popup.
func (t *tuiApp) updateAutocomplete() {
	if t.acSuppress {
		return
	}
	text := t.inputArea.GetText()

	// Only autocomplete when text starts with "/" and is a single line
	if !strings.HasPrefix(text, "/") || strings.Contains(text, "\n") {
		if t.acActive {
			t.dismissAutocomplete()
		}

		return
	}

	prefix := strings.ToLower(text)
	var matches []struct{ cmd, desc string }
	for _, sc := range slashCommands {
		if strings.HasPrefix(strings.ToLower(sc.cmd), prefix) {
			matches = append(matches, struct{ cmd, desc string }{sc.cmd, sc.desc})
		}
	}

	if len(matches) == 0 {
		if t.acActive {
			t.dismissAutocomplete()
		}

		return
	}

	// Build or update the list
	if t.acList == nil {
		t.acList = tview.NewList().
			ShowSecondaryText(true).
			SetHighlightFullLine(true).
			SetSelectedBackgroundColor(tcell.ColorDarkBlue).
			SetSelectedTextColor(tcell.ColorWhite).
			SetSecondaryTextColor(tcell.ColorGray)
		t.acList.SetBorder(true).
			SetBorderColor(tcell.ColorDarkGray).
			SetTitle(" commands ").
			SetTitleColor(tcell.ColorGray)
		t.acList.SetBackgroundColor(tcell.ColorBlack)
	}

	t.acList.Clear()
	for _, m := range matches {
		cmd := m.cmd
		t.acList.AddItem(cmd, m.desc, 0, nil)
	}

	// Size: 2 lines per item (main + secondary) + 2 for border, cap at 10
	height := len(matches)*2 + 2
	if height > 12 {
		height = 12
	}

	if !t.acActive {
		t.pages.AddPage("autocomplete", t.acList, false, false)
		t.pages.ShowPage("autocomplete")
		t.app.SetFocus(t.inputArea)
		t.acActive = true
	}

	// Position above the input area
	_, _, screenW, screenH := t.rootFlex.GetRect()
	// Input area is at the bottom: 3 rows input + 1 divider below + 1 divider above
	y := screenH - 3 - 1 - height
	if y < 0 {
		y = 0
	}
	width := 40
	if width > screenW-2 {
		width = screenW - 2
	}
	t.acList.SetRect(1, y, width, height)
}

// acceptAutocomplete inserts the selected command into the input area.
func (t *tuiApp) acceptAutocomplete() {
	if t.acList == nil || t.acList.GetItemCount() == 0 {
		return
	}
	idx := t.acList.GetCurrentItem()
	main, _ := t.acList.GetItemText(idx)
	t.dismissAutocomplete()
	t.acSuppress = true
	t.inputArea.SetText(main, true)
	t.acSuppress = false
}

// dismissAutocomplete hides the autocomplete popup.
func (t *tuiApp) dismissAutocomplete() {
	if !t.acActive {
		return
	}
	t.pages.RemovePage("autocomplete")
	t.acActive = false
	t.app.SetFocus(t.inputArea)
}
