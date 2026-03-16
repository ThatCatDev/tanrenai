package cmd

import (
	"fmt"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"

	"github.com/ThatCatDev/tanrenai/shared/tools"
)

// showProcessPanel creates and adds the process panel below the input area.
func (t *tuiApp) showProcessPanel() {
	if t.processPanel != nil {
		return // already showing
	}
	t.processPanel = tview.NewTable().
		SetSelectable(true, false).
		SetFixed(1, 0)
	t.processPanel.SetBackgroundColor(tcell.ColorDefault)
	t.processPanel.SetBorder(false)

	t.refreshProcessPanel()

	// Insert panel + divider before the last divider in rootFlex
	t.rootFlex.AddItem(t.processPanel, 5, 0, false)
	t.rootFlex.AddItem(newHDivider(), 1, 0, false)
}

// hideProcessPanel removes the process panel from rootFlex.
func (t *tuiApp) hideProcessPanel() {
	if t.processPanel == nil {
		return
	}
	// Rebuild rootFlex without the process panel
	t.rootFlex.Clear()
	t.rootFlex.AddItem(t.chatArea, 0, 1, false)
	t.rootFlex.AddItem(t.statusBar, 1, 0, false)
	t.rootFlex.AddItem(newHDivider(), 1, 0, false)
	t.rootFlex.AddItem(t.inputArea, 3, 0, true)
	t.rootFlex.AddItem(newHDivider(), 1, 0, false)
	t.processPanel = nil
	t.focus = focusChat
	t.app.SetFocus(t.inputArea)
}

// refreshProcessPanel updates the table rows from the process manager.
// Hides the panel when no processes remain.
func (t *tuiApp) refreshProcessPanel() {
	if t.procMgr == nil {
		return
	}

	procs := t.procMgr.List()

	// Auto-hide panel when no processes remain
	if len(procs) == 0 {
		t.hideProcessPanel()

		return
	}

	if t.processPanel == nil {
		return
	}

	t.processPanel.Clear()

	// Header row
	headers := []string{" #", "PID", "Status", "Command"}
	for col, h := range headers {
		cell := tview.NewTableCell(h).
			SetTextColor(tcell.ColorGray).
			SetSelectable(false).
			SetAttributes(tcell.AttrBold)
		t.processPanel.SetCell(0, col, cell)
	}

	for i, p := range procs {
		row := i + 1

		t.processPanel.SetCell(row, 0,
			tview.NewTableCell(fmt.Sprintf(" %d", p.ID)).
				SetTextColor(tcell.ColorWhite))

		t.processPanel.SetCell(row, 1,
			tview.NewTableCell(fmt.Sprintf("%d", p.PID)).
				SetTextColor(tcell.ColorGray))

		statusText, statusColor := processStatusDisplay(p)
		t.processPanel.SetCell(row, 2,
			tview.NewTableCell(statusText).
				SetTextColor(statusColor))

		cmd := p.Command
		if len(cmd) > 60 {
			cmd = cmd[:57] + "..."
		}
		t.processPanel.SetCell(row, 3,
			tview.NewTableCell(cmd).
				SetTextColor(tcell.ColorWhite).
				SetExpansion(1))
	}

	// Select first data row if it exists
	if len(procs) > 0 {
		t.processPanel.Select(1, 0)
	}
}

func processStatusDisplay(p tools.ProcessSnapshot) (string, tcell.Color) {
	if p.Status == tools.ProcessRunning {
		return "running", tcell.ColorGreen
	}

	return fmt.Sprintf("exited(%d)", p.ExitCode), tcell.ColorYellow
}

// handleProcessPanelKey handles keyboard input when the process panel is focused.
// Returns true if the key was consumed.
func (t *tuiApp) handleProcessPanelKey(event *tcell.EventKey) bool {
	if t.processPanel == nil || t.procMgr == nil {
		return false
	}

	procs := t.procMgr.List()
	if len(procs) == 0 {
		return false
	}

	row, _ := t.processPanel.GetSelection()

	switch event.Key() {
	case tcell.KeyUp:
		if row > 1 {
			t.processPanel.Select(row-1, 0)
		}

		return true

	case tcell.KeyDown:
		if row < len(procs) {
			t.processPanel.Select(row+1, 0)
		}

		return true

	case tcell.KeyEnter:
		// View process output in the file viewer
		idx := row - 1 // header is row 0
		if idx >= 0 && idx < len(procs) {
			p := procs[idx]
			output := t.procMgr.Output(p.ID)
			if output == "" {
				output = "(no output yet)"
			}
			syntheticPath := fmt.Sprintf("[process %d: %s]", p.ID, p.Command)
			t.openFileViewerContent(syntheticPath, output, nil)
		}

		return true

	case tcell.KeyEscape:
		t.focus = focusChat
		t.app.SetFocus(t.inputArea)

		return true

	case tcell.KeyRune:
		switch event.Rune() {
		case 'k', 'K':
			idx := row - 1
			if idx >= 0 && idx < len(procs) {
				_ = t.procMgr.Kill(procs[idx].ID)
				t.refreshProcessPanel()
			}

			return true
		default:
			// other runes not handled
		}
	default:
		// other keys not handled
	}

	return false
}

// processCount returns the number of managed processes, or 0 if no manager.
func (t *tuiApp) processCount() int {
	if t.procMgr == nil {
		return 0
	}

	return t.procMgr.Count()
}
