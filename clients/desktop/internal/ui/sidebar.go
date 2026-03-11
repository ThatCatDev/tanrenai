package ui

import (
	"github.com/diamondburned/gotk4-adwaita/pkg/adw"
	"github.com/diamondburned/gotk4/pkg/gtk/v4"
)

// buildSidebar builds the chat history sidebar.
func (a *App) buildSidebar() *adw.NavigationPage {
	// New Chat button
	newChatBtn := gtk.NewButton()
	newChatBtn.SetLabel("+ New Chat")
	newChatBtn.AddCSSClass("new-chat-button")
	newChatBtn.SetMarginStart(12)
	newChatBtn.SetMarginEnd(12)
	newChatBtn.SetMarginTop(12)
	newChatBtn.ConnectClicked(func() {
		a.newChat()
	})

	// Section label
	chatsLabel := gtk.NewLabel("CHATS")
	chatsLabel.AddCSSClass("chats-section-label")
	chatsLabel.SetXAlign(0)
	chatsLabel.SetMarginStart(16)
	chatsLabel.SetMarginTop(12)
	chatsLabel.SetMarginBottom(4)

	// Chat list
	a.sessionListBox = gtk.NewListBox()
	a.sessionListBox.AddCSSClass("navigation-sidebar")
	a.sessionListBox.AddCSSClass("chat-list")
	a.sessionListBox.SetSelectionMode(gtk.SelectionSingle)
	a.sessionListBox.ConnectRowActivated(func(row *gtk.ListBoxRow) {
		idx := row.Index()
		if idx >= 0 && idx < len(a.sessions) {
			a.selectSession(a.sessions[idx])
		}
	})

	listScroll := gtk.NewScrolledWindow()
	listScroll.SetChild(a.sessionListBox)
	listScroll.SetVExpand(true)
	listScroll.SetPolicy(gtk.PolicyNever, gtk.PolicyAutomatic)

	// Settings button at bottom
	settingsBtn := gtk.NewButton()
	settingsBtn.SetIconName("emblem-system-symbolic")
	settingsBtn.SetLabel("Settings")
	settingsBtn.AddCSSClass("flat")
	settingsBtn.AddCSSClass("settings-button")
	settingsBtn.SetMarginStart(12)
	settingsBtn.SetMarginEnd(12)
	settingsBtn.SetMarginBottom(12)
	settingsBtn.ConnectClicked(func() {
		a.showSettings()
	})

	sep := gtk.NewSeparator(gtk.OrientationHorizontal)

	// Layout
	content := gtk.NewBox(gtk.OrientationVertical, 0)
	content.AddCSSClass("chat-sidebar")
	content.Append(newChatBtn)
	content.Append(chatsLabel)
	content.Append(listScroll)
	content.Append(sep)
	content.Append(settingsBtn)

	toolbarView := adw.NewToolbarView()
	header := adw.NewHeaderBar()
	header.AddCSSClass("chat-sidebar")
	header.SetTitleWidget(gtk.NewLabel("Tanrenai"))
	toolbarView.AddTopBar(header)
	toolbarView.SetContent(content)

	page := adw.NewNavigationPage(toolbarView, "Chats")
	return page
}

// populateSidebarList rebuilds the chat list from sessions.
func (a *App) populateSidebarList() {
	// Remove all rows
	for {
		row := a.sessionListBox.RowAtIndex(0)
		if row == nil {
			break
		}
		a.sessionListBox.Remove(row)
	}

	for _, s := range a.sessions {
		label := gtk.NewLabel(s.Title)
		label.SetXAlign(0)
		label.SetEllipsize(3) // PANGO_ELLIPSIZE_END
		label.SetMarginStart(8)
		label.SetMarginEnd(8)
		label.SetMarginTop(6)
		label.SetMarginBottom(6)

		row := gtk.NewListBoxRow()
		row.SetChild(label)
		row.AddCSSClass("chat-list-row")
		a.sessionListBox.Append(row)
	}

	// Select active session
	a.highlightActiveSession()
}

// highlightActiveSession selects the row corresponding to the active session.
func (a *App) highlightActiveSession() {
	if a.activeSession == nil {
		return
	}
	for i, s := range a.sessions {
		if s.ID == a.activeSession.ID {
			row := a.sessionListBox.RowAtIndex(i)
			if row != nil {
				a.sessionListBox.SelectRow(row)
			}
			break
		}
	}
}

// selectSession switches to a different chat session.
func (a *App) selectSession(session *ChatSession) {
	if a.activeSession != nil && a.activeSession.ID == session.ID {
		return
	}

	// Stop generation if active
	if a.generating {
		a.onStop()
	}

	a.activeSession = session
	a.selectedModel = session.Model
	a.updateChatHeader()
	a.loadSessionMessages(session)
	a.highlightActiveSession()
	a.showChat()
}

// newChat creates a new session and switches to it.
func (a *App) newChat() {
	session := NewSession(a.selectedModel)
	a.sessions = append([]*ChatSession{session}, a.sessions...)
	a.populateSidebarList()
	a.selectSession(session) // also switches to chat view
	go a.saveSessionsAsync()
}

// updateSidebarTitle updates the title of the active session in the sidebar list.
func (a *App) updateSidebarTitle() {
	if a.activeSession == nil {
		return
	}
	for i, s := range a.sessions {
		if s.ID == a.activeSession.ID {
			row := a.sessionListBox.RowAtIndex(i)
			if row != nil {
				label := row.Child().(*gtk.Label)
				label.SetLabel(s.Title)
			}
			break
		}
	}
}
