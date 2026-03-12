package cmd

import (
	"strings"
	"time"

	"github.com/rivo/tview"
)

// Anvil forging animation frames — 鍛錬 (tanren) = forging/tempering.
// Each frame is a slice of lines, all the same height for smooth animation.
var anvilFrames = [][]string{
	// Frame 0: hammer raised high
	{
		`       ╔══╗ ╱`,
		`       ║██║╱`,
		`       ╚══╝`,
		``,
		`   ┏━━━━━━━━━━━┓`,
		`   ┃  ▄██████▄  ┃`,
		`   ┗━━━━━━━━━━━┛`,
	},
	// Frame 1: hammer swinging down
	{
		``,
		`       ╔══╗`,
		`       ║██║`,
		`       ╚══╝`,
		`   ┏━━━━━━━━━━━┓`,
		`   ┃  ▄██████▄  ┃`,
		`   ┗━━━━━━━━━━━┛`,
	},
	// Frame 2: impact!
	{
		``,
		``,
		`       ╔══╗`,
		`       ║██║`,
		`   ┏━━━╚══╝━━━━━┓`,
		`   ┃  ▄██████▄  ┃`,
		`   ┗━━━━━━━━━━━┛`,
	},
	// Frame 3: sparks fly
	{
		``,
		`     *       *`,
		`       ╔══╗`,
		`   ┏━*━║██║━━*━━┓`,
		`   ┃  ▄╚══╝██▄  ┃`,
		`   ┗━━━━━━━━━━━┛`,
		`     *       *`,
	},
	// Frame 4: sparks fading
	{
		``,
		`     ·       ·`,
		`       ╔══╗`,
		`       ║██║`,
		`   ┏━━━╚══╝━━━━━┓`,
		`   ┃  ▄██████▄  ┃`,
		`   ┗━━━━━━━━━━━┛`,
	},
	// Frame 5: hammer lifting
	{
		``,
		`       ╔══╗`,
		`       ║██║`,
		`       ╚══╝`,
		`   ┏━━━━━━━━━━━┓`,
		`   ┃  ▄██████▄  ┃`,
		`   ┗━━━━━━━━━━━┛`,
	},
}

const anvilFrameInterval = 180 * time.Millisecond

// startLoadingAnimation begins the forging animation during startup.
func (t *tuiApp) startLoadingAnimation() {
	t.anvilFrame = 0
	t.anvilStop = make(chan struct{})
	stop := t.anvilStop

	// Render first frame immediately
	t.app.QueueUpdateDraw(func() {
		t.refreshChatView()
	})

	go func() {
		ticker := time.NewTicker(anvilFrameInterval)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				t.app.QueueUpdateDraw(func() {
					t.anvilFrame = (t.anvilFrame + 1) % len(anvilFrames)
					t.refreshChatView()
				})
			}
		}
	}()
}

// stopLoadingAnimation stops the animation and clears it from the view.
func (t *tuiApp) stopLoadingAnimation() {
	if t.anvilStop != nil {
		close(t.anvilStop)
		t.anvilStop = nil
	}
	t.anvilFrame = -1
}

// renderAnvilFrame returns the colored animation frame as a string.
func renderAnvilFrame(frameIdx int) string {
	if frameIdx < 0 || frameIdx >= len(anvilFrames) {
		return ""
	}

	frame := anvilFrames[frameIdx]
	var lines []string
	for _, line := range frame {
		if line == "" {
			lines = append(lines, "")
			continue
		}
		lines = append(lines, colorizeAnvil(tview.Escape(line)))
	}
	return strings.Join(lines, "\n")
}

// colorizeAnvil applies tview color tags to anvil animation characters.
func colorizeAnvil(line string) string {
	var b strings.Builder
	inTag := false
	for _, r := range line {
		if r == '[' {
			inTag = true
			b.WriteRune(r)
			continue
		}
		if inTag {
			if r == ']' {
				inTag = false
			}
			b.WriteRune(r)
			continue
		}
		switch r {
		case '╔', '╗', '╚', '╝', '║', '═', '█':
			// Hammer — orange
			b.WriteString("[#ff8800::-]")
			b.WriteRune(r)
			b.WriteString("[-::-]")
		case '┏', '┓', '┗', '┛', '┃', '━', '▄':
			// Anvil — steel blue-gray
			b.WriteString("[#8899aa::-]")
			b.WriteRune(r)
			b.WriteString("[-::-]")
		case '*':
			// Sparks — bright yellow
			b.WriteString("[#ffcc00::b]")
			b.WriteRune(r)
			b.WriteString("[-::-]")
		case '·':
			// Fading sparks — dim
			b.WriteString("[#666644::-]")
			b.WriteRune(r)
			b.WriteString("[-::-]")
		case '╱':
			// Swing arc
			b.WriteString("[#ff8800::-]")
			b.WriteRune(r)
			b.WriteString("[-::-]")
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}
