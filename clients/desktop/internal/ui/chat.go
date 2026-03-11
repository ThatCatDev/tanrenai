package ui

import (
	"context"
	"strings"
	"time"

	"github.com/diamondburned/gotk4-adwaita/pkg/adw"
	"github.com/diamondburned/gotk4/pkg/gdk/v4"
	"github.com/diamondburned/gotk4/pkg/glib/v2"
	"github.com/diamondburned/gotk4/pkg/gtk/v4"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

// buildChat builds the chat view (shown in the content stack).
func (a *App) buildChat() *adw.ToolbarView {
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

	// Input area with rounded container
	a.inputView = gtk.NewTextView()
	a.inputView.SetWrapMode(gtk.WrapWordChar)
	a.inputView.AddCSSClass("chat-input")
	a.inputView.SetAcceptsTab(false)

	// Placeholder
	buf := a.inputView.Buffer()
	buf.Connect("changed", func() {
		// Placeholder logic handled by CSS
	})

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
	inputScroll.SetHExpand(true)

	// Circular send button
	a.sendButton = gtk.NewButtonFromIconName("go-up-symbolic")
	a.sendButton.AddCSSClass("circular")
	a.sendButton.AddCSSClass("send-button")
	a.sendButton.SetTooltipText("Send message")
	a.sendButton.SetVAlign(gtk.AlignEnd)
	a.sendButton.ConnectClicked(func() {
		if a.generating {
			a.onStop()
		} else {
			a.onSend()
		}
	})

	inputContainer := gtk.NewBox(gtk.OrientationHorizontal, 8)
	inputContainer.AddCSSClass("input-container")
	inputContainer.SetMarginStart(16)
	inputContainer.SetMarginEnd(16)
	inputContainer.SetMarginBottom(16)
	inputContainer.SetMarginTop(8)
	inputContainer.Append(inputScroll)
	inputContainer.Append(a.sendButton)

	contentBox := gtk.NewBox(gtk.OrientationVertical, 0)
	contentBox.AddCSSClass("chat-area")
	contentBox.Append(scrolled)
	contentBox.Append(inputContainer)

	// Header bar with title and model badge
	header := adw.NewHeaderBar()
	header.AddCSSClass("chat-header")

	headerBox := gtk.NewBox(gtk.OrientationHorizontal, 8)
	headerBox.SetHAlign(gtk.AlignCenter)

	a.chatTitle = gtk.NewLabel("New Chat")
	a.chatTitle.AddCSSClass("chat-title-label")

	a.modelBadge = gtk.NewLabel("")
	a.modelBadge.AddCSSClass("model-badge")

	headerBox.Append(a.chatTitle)
	headerBox.Append(a.modelBadge)
	header.SetTitleWidget(headerBox)

	// Clear chat button
	clearBtn := gtk.NewButtonFromIconName("edit-clear-all-symbolic")
	clearBtn.SetTooltipText("Clear chat")
	clearBtn.ConnectClicked(func() {
		a.clearChat()
	})
	header.PackEnd(clearBtn)

	toolbarView := adw.NewToolbarView()
	toolbarView.AddTopBar(header)
	toolbarView.SetContent(contentBox)

	return toolbarView
}

// updateChatHeader updates the header title and model badge.
func (a *App) updateChatHeader() {
	if a.activeSession != nil {
		a.chatTitle.SetLabel(a.activeSession.Title)
	}
	a.updateModelBadge()
}

// updateModelBadge updates the model badge pill.
func (a *App) updateModelBadge() {
	model := a.selectedModel
	if a.activeSession != nil && a.activeSession.Model != "" {
		model = a.activeSession.Model
	}
	if model != "" {
		a.modelBadge.SetLabel(model)
		a.modelBadge.SetVisible(true)
	} else {
		a.modelBadge.SetVisible(false)
	}
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

	// Add to active session
	a.activeSession.Messages = append(a.activeSession.Messages, api.Message{Role: "user", Content: text})
	a.activeSession.UpdatedAt = time.Now()

	// Auto-title after first user message
	if a.activeSession.Title == "New Chat" {
		a.activeSession.AutoTitle()
		a.updateChatHeader()
		a.updateSidebarTitle()
	}

	// Save session
	go a.saveSessionsAsync()

	// Start generating
	a.generating = true
	a.sendButton.SetIconName("media-playback-stop-symbolic")
	a.sendButton.RemoveCSSClass("send-button")
	a.sendButton.AddCSSClass("stop-button")

	// Use the active session's model
	if a.activeSession.Model != "" {
		a.selectedModel = a.activeSession.Model
	}

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
	a.sendButton.SetIconName("go-up-symbolic")
	a.sendButton.RemoveCSSClass("stop-button")
	a.sendButton.AddCSSClass("send-button")
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
	for {
		child := a.messageList.FirstChild()
		if child == nil {
			break
		}
		a.messageList.Remove(child)
	}
	if a.activeSession != nil {
		a.activeSession.Messages = nil
		a.activeSession.Title = "New Chat"
		a.activeSession.UpdatedAt = time.Now()
		a.updateChatHeader()
		a.updateSidebarTitle()
		go a.saveSessionsAsync()
	}
}

// loadSessionMessages clears the message list and replays all messages from a session.
func (a *App) loadSessionMessages(session *ChatSession) {
	// Clear current widgets
	for {
		child := a.messageList.FirstChild()
		if child == nil {
			break
		}
		a.messageList.Remove(child)
	}

	for _, msg := range session.Messages {
		switch msg.Role {
		case "user":
			w := messageWidget("user", msg.Content)
			a.messageList.Append(w)
		case "assistant":
			if msg.Content != "" {
				w := messageWidget("assistant", msg.Content)
				a.messageList.Append(w)
			}
			for _, tc := range msg.ToolCalls {
				w := toolCallWidget(tc)
				a.messageList.Append(w)
			}
		case "tool":
			// Tool results are shown inline with tool calls; skip standalone display
		}
	}

	// Scroll to bottom after loading
	glib.IdleAdd(func() {
		a.scrollToBottom()
	})
}

func (a *App) scrollToBottom() {
	adj := a.chatScroll.VAdjustment()
	adj.SetValue(adj.Upper())
}
