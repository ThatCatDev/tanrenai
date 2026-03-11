package ui

import (
	"encoding/json"
	"fmt"

	"github.com/diamondburned/gotk4/pkg/gtk/v4"
	"github.com/diamondburned/gotk4/pkg/pango"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

// messageWidget creates a styled widget for a chat message.
func messageWidget(role, content string) *gtk.Box {
	box := gtk.NewBox(gtk.OrientationVertical, 2)

	label := gtk.NewLabel("")
	label.SetWrap(true)
	label.SetWrapMode(pango.WrapWordChar)
	label.SetXAlign(0)
	label.SetSelectable(true)

	switch role {
	case "user":
		label.SetMarkup(markdownToPango(content))
		box.AddCSSClass("user-message")
		box.SetHAlign(gtk.AlignEnd)
		box.SetSizeRequest(600, -1) // max-width
	case "assistant":
		label.SetMarkup(markdownToPango(content))
		box.AddCSSClass("assistant-message")
		box.SetHAlign(gtk.AlignStart)
		box.SetSizeRequest(600, -1) // max-width
	default:
		label.SetText(content)
		box.SetHAlign(gtk.AlignStart)
	}

	box.Append(label)
	return box
}

// toolCallWidget creates a pill-shaped display for a tool call.
func toolCallWidget(call api.ToolCall) *gtk.Box {
	box := gtk.NewBox(gtk.OrientationHorizontal, 6)
	box.AddCSSClass("tool-call")
	box.SetHAlign(gtk.AlignStart)

	// Circle icon
	icon := gtk.NewLabel("\u2299") // ⊙ character
	icon.AddCSSClass("tool-call-icon")

	// Tool name (bold)
	nameLabel := gtk.NewLabel("")
	nameLabel.SetMarkup(fmt.Sprintf("<b>%s</b>", pangoEscape(call.Function.Name)))

	// Key argument
	keyArg := extractKeyArg(call.Function.Name, call.Function.Arguments)
	argLabel := gtk.NewLabel(keyArg)
	argLabel.AddCSSClass("tool-call-arg")
	argLabel.SetEllipsize(3) // PANGO_ELLIPSIZE_END

	box.Append(icon)
	box.Append(nameLabel)
	if keyArg != "" {
		box.Append(argLabel)
	}
	return box
}

// extractKeyArg extracts the most relevant argument for display.
func extractKeyArg(toolName, argsJSON string) string {
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return ""
	}

	// Try common key argument names in order of specificity
	keyFields := map[string][]string{
		"file_read":   {"path"},
		"file_write":  {"path"},
		"patch_file":  {"path"},
		"list_dir":    {"path"},
		"find_files":  {"pattern"},
		"grep_search": {"pattern"},
		"git_info":    {"command"},
		"shell_exec":  {"command"},
		"web_search":  {"query"},
	}

	if fields, ok := keyFields[toolName]; ok {
		for _, f := range fields {
			if v, ok := args[f]; ok {
				return fmt.Sprintf("%v", v)
			}
		}
	}

	// Fallback: try path, then command, then first string value
	for _, key := range []string{"path", "command", "pattern", "query"} {
		if v, ok := args[key]; ok {
			return fmt.Sprintf("%v", v)
		}
	}

	return ""
}

// toolResultWidget creates a compact display for a tool result.
func toolResultWidget(call api.ToolCall, result string) *gtk.Box {
	box := gtk.NewBox(gtk.OrientationVertical, 2)
	box.AddCSSClass("tool-result")
	box.SetHAlign(gtk.AlignStart)

	summary := plainTextSummary(result, 120)

	expander := gtk.NewExpander(pangoEscape(summary))
	expander.SetUseMarkup(false)

	fullLabel := gtk.NewLabel(result)
	fullLabel.SetWrap(true)
	fullLabel.SetWrapMode(pango.WrapWordChar)
	fullLabel.SetXAlign(0)
	fullLabel.SetSelectable(true)

	scroll := gtk.NewScrolledWindow()
	scroll.SetChild(fullLabel)
	scroll.SetMaxContentHeight(200)
	scroll.SetPropagateNaturalHeight(true)

	expander.SetChild(scroll)
	box.Append(expander)
	return box
}

// thinkingWidget creates a "thinking..." indicator.
func thinkingWidget() *gtk.Box {
	box := gtk.NewBox(gtk.OrientationHorizontal, 4)
	box.AddCSSClass("thinking-indicator")
	box.SetHAlign(gtk.AlignStart)

	spinner := gtk.NewSpinner()
	spinner.Start()
	box.Append(spinner)

	label := gtk.NewLabel("Thinking...")
	box.Append(label)

	return box
}

// streamingWidget creates a label for incremental streaming content.
func streamingWidget() *gtk.Label {
	label := gtk.NewLabel("")
	label.SetWrap(true)
	label.SetWrapMode(pango.WrapWordChar)
	label.SetXAlign(0)
	label.SetSelectable(true)
	label.AddCSSClass("assistant-message")
	return label
}
