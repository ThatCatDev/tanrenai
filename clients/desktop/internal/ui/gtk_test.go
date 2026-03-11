package ui

import (
	"fmt"
	"os"
	"testing"

	"github.com/diamondburned/gotk4-adwaita/pkg/adw"
	"github.com/diamondburned/gotk4/pkg/gtk/v4"

	"github.com/ThatCatDev/tanrenai/desktop/internal/server"
	"github.com/ThatCatDev/tanrenai/shared/apiclient"
)

func TestMain(m *testing.M) {
	if !gtk.InitCheck() {
		fmt.Println("SKIP: no display available for GTK tests")
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// countChildren counts the number of direct children of a widget.
func countChildren(w *gtk.Box) int {
	count := 0
	for child := w.FirstChild(); child != nil; child = gtk.BaseWidget(child).NextSibling() {
		count++
	}
	return count
}

// newTestApp creates a minimal App with real GTK widgets for testing.
func newTestApp() *App {
	a := &App{
		serverMgr: server.NewManager(),
		client:    apiclient.New("http://127.0.0.1:8080"),
	}

	a.contentStack = gtk.NewStack()
	a.contentStack.AddNamed(a.buildChat(), "chat")
	a.contentStack.AddNamed(a.buildSettings(), "settings")
	a.contentStack.SetVisibleChildName("chat")

	// Build sidebar so sessionListBox is available
	a.buildSidebar()

	a.toast = adw.NewToastOverlay()

	a.sessions = []*ChatSession{NewSession("test-model")}
	a.activeSession = a.sessions[0]
	a.selectedModel = "test-model"

	return a
}
