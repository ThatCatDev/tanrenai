package ui

import (
	"context"
	_ "embed"

	"github.com/diamondburned/gotk4-adwaita/pkg/adw"
	"github.com/diamondburned/gotk4/pkg/gdk/v4"
	"github.com/diamondburned/gotk4/pkg/gio/v2"
	"github.com/diamondburned/gotk4/pkg/gtk/v4"

	"github.com/ThatCatDev/tanrenai/desktop/internal/server"
	"github.com/ThatCatDev/tanrenai/shared/apiclient"
	"github.com/ThatCatDev/tanrenai/shared/pkg/api"
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

	// Sidebar widgets
	statusRow      *adw.ActionRow
	serverButton   *gtk.Button
	modelDropdown  *gtk.DropDown
	serverURLEntry *adw.EntryRow
	sidebarContent *gtk.Box // sidebar content box for adding download progress

	// Download state
	downloadProgress *gtk.ProgressBar
	downloadCancel   *gtk.Button
	downloadBox      *gtk.Box
	cancelDownload   context.CancelFunc

	// Chat widgets
	messageList *gtk.Box
	chatScroll  *gtk.ScrolledWindow
	inputView   *gtk.TextView
	sendButton  *gtk.Button
	chatTitle   *gtk.Label

	// State
	messages       []api.Message
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

	// Build window
	a.window = adw.NewApplicationWindow(&a.app.Application)
	a.window.SetTitle("Tanrenai")
	a.window.SetDefaultSize(1000, 700)

	// Toast overlay
	a.toast = adw.NewToastOverlay()

	// Navigation split view
	splitView := adw.NewNavigationSplitView()
	splitView.SetSidebar(a.buildSidebar())
	splitView.SetContent(a.buildChat())
	splitView.SetMinSidebarWidth(280)
	splitView.SetMaxSidebarWidth(380)

	a.toast.SetChild(splitView)
	a.window.SetContent(a.toast)
	a.window.Present()

	// Populate model dropdown from local files on startup
	go a.refreshModels()
}

func (a *App) shutdown() {
	// Stop servers on exit
	_ = a.serverMgr.Stop()
}

func (a *App) showToast(msg string) {
	toast := adw.NewToast(msg)
	toast.SetTimeout(3)
	a.toast.AddToast(toast)
}
