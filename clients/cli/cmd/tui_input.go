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

func (t *tuiApp) setupInputCapture() {
	t.app.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		// When a modal is showing, let tview handle all input (arrow keys, enter, etc.)
		if name, _ := t.pages.GetFrontPage(); name == "approval" {
			return event
		}

		if event.Key() != tcell.KeyCtrlC {
			t.ctrlCPending = false
		}

		switch event.Key() {
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
			if t.filePath != "" {
				t.closeFileViewer()
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
			t.scrollFocusedPane(1)
			return nil
		case tcell.KeyPgUp:
			t.scrollFocusedPane(-10)
			return nil
		case tcell.KeyPgDn:
			t.scrollFocusedPane(10)
			return nil

		case tcell.KeyEnter:
			if t.loading || t.processing {
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

		switch action {
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
			if mx >= cx && mx < cx+cw && my >= cy && my < cy+ch {
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
		go t.startAgentTurn(text)
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
