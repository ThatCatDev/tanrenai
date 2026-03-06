package ui

import (
	"context"
	"strings"

	"github.com/diamondburned/gotk4-adwaita/pkg/adw"
	"github.com/diamondburned/gotk4/pkg/gdk/v4"
	"github.com/diamondburned/gotk4/pkg/gtk/v4"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

// buildChat builds the chat navigation page.
func (a *App) buildChat() *adw.NavigationPage {
	// Message list
	a.messageList = gtk.NewBox(gtk.OrientationVertical, 4)
	a.messageList.AddCSSClass("message-list")
	a.messageList.SetVAlign(gtk.AlignEnd)
	a.messageList.SetVExpand(true)

	scrolled := gtk.NewScrolledWindow()
	scrolled.SetChild(a.messageList)
	scrolled.SetVExpand(true)
	scrolled.SetPolicy(gtk.PolicyNever, gtk.PolicyAutomatic)
	a.chatScroll = scrolled

	// Input area
	a.inputView = gtk.NewTextView()
	a.inputView.SetWrapMode(gtk.WrapWordChar)
	a.inputView.AddCSSClass("chat-input")
	a.inputView.SetAcceptsTab(false)

	// Enter sends, Shift+Enter adds newline
	keyController := gtk.NewEventControllerKey()
	keyController.ConnectKeyPressed(func(keyval uint, keycode uint, state gdk.ModifierType) bool {
		if keyval == gdk.KEY_Return && state&gdk.ShiftMask == 0 {
			a.onSend()
			return true
		}
		return false
	})
	a.inputView.AddController(keyController)

	inputScroll := gtk.NewScrolledWindow()
	inputScroll.SetChild(a.inputView)
	inputScroll.SetMaxContentHeight(120)
	inputScroll.SetPropagateNaturalHeight(true)

	// Send/Stop button
	a.sendButton = gtk.NewButtonWithLabel("Send")
	a.sendButton.AddCSSClass("suggested-action")
	a.sendButton.ConnectClicked(func() {
		if a.generating {
			a.onStop()
		} else {
			a.onSend()
		}
	})

	inputBox := gtk.NewBox(gtk.OrientationHorizontal, 8)
	inputBox.SetMarginStart(8)
	inputBox.SetMarginEnd(8)
	inputBox.SetMarginBottom(8)
	inputBox.Append(inputScroll)
	inputScroll.SetHExpand(true)
	inputBox.Append(a.sendButton)

	sep := gtk.NewSeparator(gtk.OrientationHorizontal)

	contentBox := gtk.NewBox(gtk.OrientationVertical, 0)
	contentBox.Append(scrolled)
	contentBox.Append(sep)
	contentBox.Append(inputBox)

	// Header bar
	header := adw.NewHeaderBar()
	a.chatTitle = gtk.NewLabel("Tanrenai")
	header.SetTitleWidget(a.chatTitle)

	// Clear chat menu
	clearBtn := gtk.NewButtonFromIconName("edit-clear-all-symbolic")
	clearBtn.SetTooltipText("Clear chat")
	clearBtn.ConnectClicked(func() {
		a.clearChat()
	})
	header.PackEnd(clearBtn)

	toolbarView := adw.NewToolbarView()
	toolbarView.AddTopBar(header)
	toolbarView.SetContent(contentBox)

	page := adw.NewNavigationPage(toolbarView, "Chat")
	return page
}

func (a *App) onSend() {
	if a.generating {
		return
	}

	buf := a.inputView.Buffer()
	start := buf.StartIter()
	end := buf.EndIter()
	text := strings.TrimSpace(buf.Text(start, end, false))
	if text == "" {
		return
	}
	buf.SetText("")

	// Add user message to UI
	a.addUserMessage(text)

	// Add to conversation history
	a.messages = append(a.messages, api.Message{Role: "user", Content: text})

	// Start generating
	a.generating = true
	a.sendButton.SetLabel("Stop")
	a.sendButton.RemoveCSSClass("suggested-action")
	a.sendButton.AddCSSClass("destructive-action")

	var ctx context.Context
	ctx, a.cancelGenerate = context.WithCancel(context.Background())
	go a.runAgentTurn(ctx)
}

func (a *App) onStop() {
	if a.cancelGenerate != nil {
		a.cancelGenerate()
	}
}

func (a *App) finishGenerating() {
	a.generating = false
	a.sendButton.SetLabel("Send")
	a.sendButton.RemoveCSSClass("destructive-action")
	a.sendButton.AddCSSClass("suggested-action")
	a.cancelGenerate = nil
}

func (a *App) addUserMessage(text string) {
	w := messageWidget("user", text)
	a.messageList.Append(w)
	a.scrollToBottom()
}

func (a *App) addAssistantMessage(text string) {
	w := messageWidget("assistant", text)
	a.messageList.Append(w)
	a.scrollToBottom()
}

func (a *App) clearChat() {
	// Remove all children
	for {
		child := a.messageList.FirstChild()
		if child == nil {
			break
		}
		a.messageList.Remove(child)
	}
	a.messages = nil
}

func (a *App) scrollToBottom() {
	adj := a.chatScroll.VAdjustment()
	adj.SetValue(adj.Upper())
}
