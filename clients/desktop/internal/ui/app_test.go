package ui

import "testing"

func TestActivateBuildsUI(t *testing.T) {
	a := newTestApp()

	if a.messageList == nil {
		t.Fatal("buildChat did not set messageList")
	}
	if a.inputView == nil {
		t.Fatal("buildChat did not set inputView")
	}
	if a.sendButton == nil {
		t.Fatal("buildChat did not set sendButton")
	}
	if a.chatTitle == nil {
		t.Fatal("buildChat did not set chatTitle")
	}
	if a.modelBadge == nil {
		t.Fatal("buildChat did not set modelBadge")
	}
	if a.statusRow == nil {
		t.Fatal("buildSettings did not set statusRow")
	}
	if a.serverButton == nil {
		t.Fatal("buildSettings did not set serverButton")
	}
	if a.modelDropdown == nil {
		t.Fatal("buildSettings did not set modelDropdown")
	}
	if a.settingsContent == nil {
		t.Fatal("buildSettings did not set settingsContent")
	}
}

func TestShowSettingsShowChat(t *testing.T) {
	a := newTestApp()

	a.showSettings()
	if name := a.contentStack.VisibleChildName(); name != "settings" {
		t.Fatalf("expected visible child 'settings', got %q", name)
	}

	a.showChat()
	if name := a.contentStack.VisibleChildName(); name != "chat" {
		t.Fatalf("expected visible child 'chat', got %q", name)
	}
}

func TestShowToast(t *testing.T) {
	a := newTestApp()
	// Should not panic
	a.showToast("test message")
}
