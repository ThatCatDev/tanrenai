package ui

import (
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
	case "assistant":
		label.SetMarkup(markdownToPango(content))
		box.AddCSSClass("assistant-message")
		box.SetHAlign(gtk.AlignStart)
	default:
		label.SetText(content)
		box.SetHAlign(gtk.AlignStart)
	}

	box.Append(label)
	return box
}

// toolCallWidget creates a compact display for a tool call.
func toolCallWidget(call api.ToolCall) *gtk.Box {
	box := gtk.NewBox(gtk.OrientationVertical, 2)
	box.AddCSSClass("tool-call")
	box.SetHAlign(gtk.AlignStart)

	argsSummary := plainTextSummary(call.Function.Arguments, 80)
	text := fmt.Sprintf("<b>%s</b> %s", pangoEscape(call.Function.Name), pangoEscape(argsSummary))

	label := gtk.NewLabel("")
	label.SetMarkup(text)
	label.SetWrap(true)
	label.SetWrapMode(pango.WrapWordChar)
	label.SetXAlign(0)
	label.SetSelectable(true)

	box.Append(label)
	return box
}

// toolResultWidget creates a compact display for a tool result.
func toolResultWidget(call api.ToolCall, result string) *gtk.Box {
	box := gtk.NewBox(gtk.OrientationVertical, 2)
	box.AddCSSClass("tool-result")
	box.SetHAlign(gtk.AlignStart)

	summary := plainTextSummary(result, 120)

	expander := gtk.NewExpander(pangoEscape(summary))
	expander.SetUseMarkup(false)

	// Full result in a scrollable label
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
