package ui

import (
	"context"
	_ "embed"
	"log"

	"github.com/diamondburned/gotk4-adwaita/pkg/adw"
	"github.com/diamondburned/gotk4/pkg/gdk/v4"
	"github.com/diamondburned/gotk4/pkg/gio/v2"
	"github.com/diamondburned/gotk4/pkg/gtk/v4"

	"github.com/ThatCatDev/tanrenai/desktop/internal/server"
	"github.com/ThatCatDev/tanrenai/shared/apiclient"
)

//go:embed styles.css
var cssData string

// App holds all application state and GTK widgets.
type App struct {
	app    *adw.Application
	window *adw.ApplicationWindow
	toast  *adw.ToastOverlay

	// Server management
	serverMgr *server.Manager
	client    *apiclient.Client

	// Settings widgets
	statusRow       *adw.ActionRow
	serverButton    *gtk.Button
	modelDropdown   *gtk.DropDown
	serverURLEntry  *adw.EntryRow
	settingsContent *gtk.Box

	// Content stack (chat vs settings)
	contentStack *gtk.Stack

	// Download state
	downloadProgress *gtk.ProgressBar
	downloadCancel   *gtk.Button
	downloadBox      *gtk.Box
	cancelDownload   context.CancelFunc

	// Sidebar
	sessionListBox *gtk.ListBox

	// Chat widgets
	messageList *gtk.Box
	chatScroll  *gtk.ScrolledWindow
	inputView   *gtk.TextView
	sendButton  *gtk.Button
	chatTitle   *gtk.Label
	modelBadge  *gtk.Label

	// Sessions
	sessions      []*ChatSession
	activeSession *ChatSession

	// State
	modelList      []string
	selectedModel  string
	generating     bool
	cancelGenerate context.CancelFunc
}

// Run creates and runs the application.
func Run() int {
	a := &App{
		serverMgr: server.NewManager(),
		client:    apiclient.New("http://127.0.0.1:8080"),
	}

	a.app = adw.NewApplication("dev.tanrenai.desktop", gio.ApplicationFlagsNone)
	a.app.ConnectActivate(func() { a.activate() })
	a.app.ConnectShutdown(func() { a.shutdown() })

	return a.app.Run(nil)
}

func (a *App) activate() {
	// Load CSS
	provider := gtk.NewCSSProvider()
	provider.LoadFromString(cssData)
	gtk.StyleContextAddProviderForDisplay(
		gdk.DisplayGetDefault(),
		provider,
		gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
	)

	// Load sessions from disk
	sessions, err := LoadSessions()
	if err != nil {
		log.Printf("Failed to load sessions: %v", err)
	}
	a.sessions = sessions

	// Ensure at least one session exists
	if len(a.sessions) == 0 {
		a.sessions = []*ChatSession{NewSession("")}
	}
	a.activeSession = a.sessions[0]

	// Build window
	a.window = adw.NewApplicationWindow(&a.app.Application)
	a.window.SetTitle("Tanrenai")
	a.window.SetDefaultSize(1000, 700)

	// Toast overlay
	a.toast = adw.NewToastOverlay()

	// Content stack: chat and settings pages
	a.contentStack = gtk.NewStack()
	a.contentStack.SetTransitionType(gtk.StackTransitionTypeCrossfade)
	a.contentStack.AddNamed(a.buildChat(), "chat")
	a.contentStack.AddNamed(a.buildSettings(), "settings")
	a.contentStack.SetVisibleChildName("chat")

	// Wrap stack in a NavigationPage for the split view
	contentPage := adw.NewNavigationPage(a.contentStack, "Content")

	// Navigation split view
	splitView := adw.NewNavigationSplitView()
	splitView.SetSidebar(a.buildSidebar())
	splitView.SetContent(contentPage)
	splitView.SetMinSidebarWidth(260)
	splitView.SetMaxSidebarWidth(320)

	a.toast.SetChild(splitView)
	a.window.SetContent(a.toast)

	// Populate sidebar and load active session messages
	a.populateSidebarList()
	a.loadSessionMessages(a.activeSession)
	a.updateChatHeader()

	a.window.Present()

	go a.refreshModels()
}

func (a *App) shutdown() {
	_ = SaveSessions(a.sessions)
	_ = a.serverMgr.Stop()
}

func (a *App) saveSessionsAsync() {
	if err := SaveSessions(a.sessions); err != nil {
		log.Printf("Failed to save sessions: %v", err)
	}
}

func (a *App) showSettings() {
	a.contentStack.SetVisibleChildName("settings")
}

func (a *App) showChat() {
	a.contentStack.SetVisibleChildName("chat")
}

func (a *App) showToast(msg string) {
	toast := adw.NewToast(msg)
	toast.SetTimeout(3)
	a.toast.AddToast(toast)
}
