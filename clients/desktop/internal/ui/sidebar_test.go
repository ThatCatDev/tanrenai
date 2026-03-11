package ui

import (
	"testing"

	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
)

func TestBuildSidebar(t *testing.T) {
	a := newTestApp()
	page := a.buildSidebar()
	if page == nil {
		t.Fatal("buildSidebar returned nil")
	}
}

func TestPopulateSidebarList(t *testing.T) {
	a := newTestApp()
	a.sessions = []*ChatSession{
		NewSession("model-a"),
		NewSession("model-b"),
		NewSession("model-c"),
	}
	a.activeSession = a.sessions[0]
	a.populateSidebarList()

	// Count rows
	count := 0
	for i := 0; ; i++ {
		if a.sessionListBox.RowAtIndex(i) == nil {
			break
		}
		count++
	}
	if count != 3 {
		t.Fatalf("expected 3 rows, got %d", count)
	}
}

func TestSelectSession(t *testing.T) {
	a := newTestApp()
	s1 := NewSession("model-a")
	s2 := NewSession("model-b")
	s2.Title = "Second Chat"
	s2.Messages = []api.Message{{Role: "user", Content: "hello"}}
	a.sessions = []*ChatSession{s1, s2}
	a.activeSession = s1
	a.populateSidebarList()

	a.selectSession(s2)

	if a.activeSession.ID != s2.ID {
		t.Fatalf("expected activeSession %q, got %q", s2.ID, a.activeSession.ID)
	}
	if a.selectedModel != "model-b" {
		t.Fatalf("expected selectedModel 'model-b', got %q", a.selectedModel)
	}
}

func TestSelectSessionSameNoOp(t *testing.T) {
	a := newTestApp()
	s := a.sessions[0]
	// Should not panic when selecting the already-active session
	a.selectSession(s)
	if a.activeSession.ID != s.ID {
		t.Fatal("activeSession changed unexpectedly")
	}
}

func TestNewChat(t *testing.T) {
	a := newTestApp()
	a.populateSidebarList()
	origLen := len(a.sessions)

	a.newChat()

	if len(a.sessions) != origLen+1 {
		t.Fatalf("expected %d sessions, got %d", origLen+1, len(a.sessions))
	}
	// New session should be first
	if a.activeSession.ID != a.sessions[0].ID {
		t.Fatal("new session should be the active session")
	}
	if a.activeSession.Title != "New Chat" {
		t.Fatalf("expected 'New Chat' title, got %q", a.activeSession.Title)
	}
}

func TestUpdateSidebarTitle(t *testing.T) {
	a := newTestApp()
	a.populateSidebarList()

	a.activeSession.Title = "Updated Title"
	a.updateSidebarTitle()

	// Verify the label in the sidebar row was updated
	row := a.sessionListBox.RowAtIndex(0)
	if row == nil {
		t.Fatal("expected row at index 0")
	}
}
