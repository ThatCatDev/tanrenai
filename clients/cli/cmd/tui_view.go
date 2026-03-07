package cmd

import (
	"bytes"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/alecthomas/chroma/v2"
	"github.com/alecthomas/chroma/v2/formatters"
	"github.com/alecthomas/chroma/v2/lexers"
	"github.com/alecthomas/chroma/v2/styles"
	"github.com/charmbracelet/glamour"
	"github.com/rivo/tview"
)

// ── Status Bar ──────────────────────────────────────────────────────────

func (t *tuiApp) updateStatusBar() {
	if t.processing && t.statusText != "" {
		tokenInfo := ""
		if t.lastInputTokens > 0 {
			tokenInfo = " | ~" + formatTokenCount(t.lastInputTokens) + " in"
		}
		bar := ""
		if !t.iterStartTime.IsZero() {
			elapsed := time.Since(t.iterStartTime)
			bar = " " + renderProgressBar(elapsed, t.estimatedDur)
		}
		t.statusBar.SetText(" [gray::-]" + tview.Escape(t.statusText+tokenInfo) + "[-:-:-] " + bar)
	} else if t.lastInputTokens > 0 || t.lastOutputTokens > 0 {
		parts := []string{}
		if t.lastInputTokens > 0 {
			parts = append(parts, "~"+formatTokenCount(t.lastInputTokens)+" in")
		}
		if t.lastOutputTokens > 0 {
			parts = append(parts, "~"+formatTokenCount(t.lastOutputTokens)+" out")
		}
		t.statusBar.SetText(" [gray::-]" + strings.Join(parts, " / ") + "[-:-:-]")
	} else {
		t.statusBar.SetText("")
	}
}

func formatTokenCount(n int) string {
	if n >= 1000 {
		return fmt.Sprintf("%.1fk", float64(n)/1000)
	}
	return fmt.Sprintf("%d", n)
}

func renderProgressBar(elapsed, estimated time.Duration) string {
	const barWidth = 20
	if estimated <= 0 {
		// No estimate yet — just show elapsed
		return fmt.Sprintf("[gray::-]%ds[-:-:-]", int(elapsed.Seconds()))
	}
	ratio := float64(elapsed) / float64(estimated)
	if ratio > 1.0 {
		ratio = 1.0
	}
	filled := int(ratio * barWidth)
	if filled > barWidth {
		filled = barWidth
	}
	empty := barWidth - filled

	remaining := estimated - elapsed
	countdown := ""
	if remaining > 0 {
		countdown = fmt.Sprintf("%ds", int(remaining.Seconds()))
	} else {
		over := elapsed - estimated
		countdown = fmt.Sprintf("+%ds", int(over.Seconds()))
	}

	return fmt.Sprintf("[gray::-][[-]%s%s[gray::-]][-:-:-] [gray::-]%s[-:-:-]",
		strings.Repeat("█", filled),
		strings.Repeat("░", empty),
		countdown)
}

// ── Display Line Mapping ────────────────────────────────────────────────

func (t *tuiApp) displayLineToLogicalLine(displayLine int) int {
	_, _, cw, _ := t.chatView.GetRect()
	if cw <= 0 {
		cw = 80
	}

	cur := 0
	for i, line := range t.lines {
		if t.expanded && len(t.toolResults) > 0 {
			if full, ok := t.toolResults[i]; ok {
				for _, fline := range strings.Split(strings.TrimRight(full, "\n"), "\n") {
					escaped := "[gray::-]      " + tview.Escape(fline) + "[-:-:-]"
					rows := wrappedLineRows(escaped, cw)
					if displayLine < cur+rows {
						return i
					}
					cur += rows
				}
				continue
			}
		}
		rows := wrappedLineRows(line, cw)
		if displayLine < cur+rows {
			return i
		}
		cur += rows
	}
	return -1
}

// wrappedLineRows estimates how many display rows a logical line occupies
// when word-wrapped to the given view width.
func wrappedLineRows(taggedLine string, viewWidth int) int {
	w := tview.TaggedStringWidth(taggedLine)
	if w <= viewWidth {
		return 1
	}
	return (w + viewWidth - 1) / viewWidth
}

// ── File Viewer ─────────────────────────────────────────────────────────

func (t *tuiApp) loadFileViewer(path string) {
	const maxSize = 64 * 1024
	data, err := os.ReadFile(path)
	if err != nil {
		t.app.QueueUpdateDraw(func() {
			t.openFileViewerContent(path, "", err)
		})
		return
	}
	content := string(data)
	if len(content) > maxSize {
		content = content[:maxSize] + "\n... (truncated at 64KB)"
	}
	t.app.QueueUpdateDraw(func() {
		t.openFileViewerContent(path, content, nil)
	})
}

func (t *tuiApp) openFileViewerContent(path, content string, err error) {
	t.filePath = path
	t.focus = focusFileViewer

	// Create file header
	t.fileHeader = tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(false)
	t.fileHeader.SetBorder(false)
	t.fileHeader.SetText(fmt.Sprintf("[blue::b]%s[-:-:-] [gray::-]Esc close | Tab focus[-:-:-]",
		tview.Escape(path)))

	// Create file view
	t.fileView = tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(true).
		SetWordWrap(false)
	t.fileView.SetBorder(false)

	if err != nil {
		t.fileView.SetText(fmt.Sprintf("[red::-]Error: %v[-:-:-]", err))
	} else {
		highlighted := highlightContent(path, content)
		numbered := addLineNumbers(highlighted)
		// Convert ANSI to tview color tags
		t.fileView.SetText(tview.TranslateANSI(numbered))
	}

	// Build file panel (header + file content)
	t.filePanel = tview.NewFlex().SetDirection(tview.FlexRow)
	t.filePanel.AddItem(t.fileHeader, 1, 0, false)
	t.filePanel.AddItem(t.fileView, 0, 1, false)

	// Rebuild chatArea with split
	t.chatArea.Clear()
	t.chatArea.AddItem(t.chatView, 0, 1, false)
	t.chatArea.AddItem(newVDivider(t.focus == focusFileViewer), 1, 0, false)
	t.chatArea.AddItem(t.filePanel, 0, 1, false)
}

func (t *tuiApp) rebuildFileViewer() {
	if t.filePath == "" {
		return
	}
	// Rebuild chatArea to update divider focus color
	t.chatArea.Clear()
	t.chatArea.AddItem(t.chatView, 0, 1, false)
	t.chatArea.AddItem(newVDivider(t.focus == focusFileViewer), 1, 0, false)
	t.chatArea.AddItem(t.filePanel, 0, 1, false)
}

func (t *tuiApp) closeFileViewer() {
	t.filePath = ""
	t.focus = focusChat
	t.filePanel = nil
	t.fileHeader = nil
	t.fileView = nil

	t.chatArea.Clear()
	t.chatArea.AddItem(t.chatView, 0, 1, false)
}

// ── Syntax Highlighting ─────────────────────────────────────────────────

func highlightContent(path, content string) string {
	lexer := lexers.Match(path)
	if lexer == nil {
		lexer = lexers.Analyse(content)
	}
	if lexer == nil {
		lexer = lexers.Fallback
	}
	lexer = chroma.Coalesce(lexer)

	style := styles.Get("monokai")
	formatter := formatters.Get("terminal256")
	if formatter == nil {
		return content
	}

	iterator, err := lexer.Tokenise(nil, content)
	if err != nil {
		return content
	}

	var buf bytes.Buffer
	if err := formatter.Format(&buf, style, iterator); err != nil {
		return content
	}
	return buf.String()
}

func addLineNumbers(content string) string {
	lines := strings.Split(content, "\n")
	width := len(fmt.Sprintf("%d", len(lines)))

	var b strings.Builder
	for i, line := range lines {
		num := fmt.Sprintf("%*d", width, i+1)
		// ANSI gray for line numbers
		b.WriteString("\033[38;5;240m")
		b.WriteString(num)
		b.WriteString("\033[0m ")
		b.WriteString(line)
		if i < len(lines)-1 {
			b.WriteString("\n")
		}
	}
	return b.String()
}

// ── Markdown Rendering ──────────────────────────────────────────────────

var ansiSGR = regexp.MustCompile("\x1b\\[([0-9;:]*)m")
var tviewTag = regexp.MustCompile(`\[([^\[\]]*):([^\[\]]*):([^\[\]]*)\]`)

// stripTviewUnderline removes 'u' (underline) from the attributes field of
// tview color tags like [fg:bg:attrs]. This is a safety net in case ANSI
// underline sequences slip through stripANSIUnderline.
func stripTviewUnderline(s string) string {
	return tviewTag.ReplaceAllStringFunc(s, func(tag string) string {
		inner := tag[1 : len(tag)-1]
		parts := strings.SplitN(inner, ":", 3)
		if len(parts) < 3 {
			return tag
		}
		attrs := parts[2]
		if !strings.ContainsRune(attrs, 'u') {
			return tag
		}
		newAttrs := strings.ReplaceAll(attrs, "u", "")
		if newAttrs == "" {
			newAttrs = "-"
		}
		return "[" + parts[0] + ":" + parts[1] + ":" + newAttrs + "]"
	})
}

// stripANSIUnderline removes underline (4, 4:N) and no-underline (24) parameters
// from ANSI SGR sequences, preserving all other attributes and colors.
func stripANSIUnderline(s string) string {
	return ansiSGR.ReplaceAllStringFunc(s, func(seq string) string {
		inner := seq[2 : len(seq)-1] // between \x1b[ and m
		if inner == "" {
			return seq
		}
		params := strings.Split(inner, ";")
		var out []string
		for i := 0; i < len(params); i++ {
			p := params[i]
			// Strip underline: 4, 4:N (colon sub-params), 24
			if p == "4" || p == "24" || strings.HasPrefix(p, "4:") {
				continue
			}
			// 38;5;N / 48;5;N: extended color — consume all three parts
			if (p == "38" || p == "48") && i+2 < len(params) && params[i+1] == "5" {
				out = append(out, p, params[i+1], params[i+2])
				i += 2
				continue
			}
			// 38;2;R;G;B / 48;2;R;G;B: true color — consume all five parts
			if (p == "38" || p == "48") && i+4 < len(params) && params[i+1] == "2" {
				out = append(out, p, params[i+1], params[i+2], params[i+3], params[i+4])
				i += 4
				continue
			}
			// 38:5:N / 48:5:N / 38:2:R:G:B / 48:2:R:G:B: colon-style extended color
			if strings.HasPrefix(p, "38:") || strings.HasPrefix(p, "48:") {
				out = append(out, p)
				continue
			}
			out = append(out, p)
		}
		if len(out) == 0 {
			return ""
		}
		return "\x1b[" + strings.Join(out, ";") + "m"
	})
}

func (t *tuiApp) renderMarkdown(content string) string {
	r, err := glamour.NewTermRenderer(
		glamour.WithStandardStyle("dark"),
		glamour.WithWordWrap(0),
	)
	if err != nil {
		return tview.Escape(content)
	}
	out, err := r.Render(content)
	if err != nil {
		return tview.Escape(content)
	}
	out = stripANSIUnderline(out)
	translated := tview.TranslateANSI(strings.TrimRight(out, "\n"))
	return stripTviewUnderline(translated)
}
