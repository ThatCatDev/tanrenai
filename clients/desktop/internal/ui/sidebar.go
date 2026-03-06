package ui

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/diamondburned/gotk4-adwaita/pkg/adw"
	coreglib "github.com/diamondburned/gotk4/pkg/core/glib"
	"github.com/diamondburned/gotk4/pkg/glib/v2"
	"github.com/diamondburned/gotk4/pkg/gtk/v4"

	"github.com/ThatCatDev/tanrenai/desktop/internal/server"
	"github.com/ThatCatDev/tanrenai/shared/apiclient"
)

// sidebar builds the settings navigation page.
func (a *App) buildSidebar() *adw.NavigationPage {
	// Server status row
	a.statusRow = adw.NewActionRow()
	a.statusRow.SetTitle("Status")
	a.statusRow.SetSubtitle("Stopped")

	// Start/Stop button
	a.serverButton = gtk.NewButtonWithLabel("Start Server")
	a.serverButton.AddCSSClass("suggested-action")
	a.serverButton.ConnectClicked(func() {
		if a.serverMgr.Status() == server.StatusRunning {
			go a.stopServer()
		} else {
			go a.startServer()
		}
	})

	serverGroup := adw.NewPreferencesGroup()
	serverGroup.SetTitle("Server")
	serverGroup.Add(a.statusRow)

	buttonBox := gtk.NewBox(gtk.OrientationHorizontal, 0)
	buttonBox.SetHAlign(gtk.AlignCenter)
	buttonBox.SetMarginTop(8)
	buttonBox.SetMarginBottom(8)
	buttonBox.Append(a.serverButton)

	// Model selector
	a.modelDropdown = gtk.NewDropDownFromStrings([]string{"(no models)"})
	a.modelDropdown.SetSensitive(false)
	a.modelDropdown.Connect("notify::selected", func(_ coreglib.Objector) {
		idx := a.modelDropdown.Selected()
		if idx < uint(len(a.modelList)) {
			a.selectedModel = a.modelList[idx]
		}
	})

	modelRow := adw.NewActionRow()
	modelRow.SetTitle("Model")
	modelRow.AddSuffix(a.modelDropdown)
	modelRow.SetActivatableWidget(a.modelDropdown)

	modelGroup := adw.NewPreferencesGroup()
	modelGroup.SetTitle("Model")
	modelGroup.Add(modelRow)

	// Download and Import buttons
	downloadBtn := gtk.NewButtonWithLabel("Download")
	downloadBtn.AddCSSClass("flat")
	downloadBtn.ConnectClicked(func() { a.showDownloadDialog() })

	importBtn := gtk.NewButtonWithLabel("Import")
	importBtn.AddCSSClass("flat")
	importBtn.ConnectClicked(func() { a.showImportDialog() })

	modelButtonBox := gtk.NewBox(gtk.OrientationHorizontal, 8)
	modelButtonBox.SetHAlign(gtk.AlignCenter)
	modelButtonBox.SetMarginTop(4)
	modelButtonBox.SetMarginBottom(8)
	modelButtonBox.Append(downloadBtn)
	modelButtonBox.Append(importBtn)

	// Connection settings
	a.serverURLEntry = adw.NewEntryRow()
	a.serverURLEntry.SetTitle("Server URL")
	a.serverURLEntry.SetText("http://127.0.0.1:8080")

	connGroup := adw.NewPreferencesGroup()
	connGroup.SetTitle("Connection")
	connGroup.Add(a.serverURLEntry)

	// Layout
	content := gtk.NewBox(gtk.OrientationVertical, 12)
	content.SetMarginTop(12)
	content.SetMarginBottom(12)
	content.SetMarginStart(12)
	content.SetMarginEnd(12)
	content.Append(serverGroup)
	content.Append(buttonBox)
	content.Append(modelGroup)
	content.Append(modelButtonBox)
	content.Append(connGroup)
	a.sidebarContent = content

	scrolled := gtk.NewScrolledWindow()
	scrolled.SetChild(content)
	scrolled.SetVExpand(true)

	toolbarView := adw.NewToolbarView()
	header := adw.NewHeaderBar()
	header.SetTitleWidget(gtk.NewLabel("Settings"))
	toolbarView.AddTopBar(header)
	toolbarView.SetContent(scrolled)

	page := adw.NewNavigationPage(toolbarView, "Settings")
	return page
}

func (a *App) startServer() {
	glib.IdleAdd(func() {
		a.statusRow.SetSubtitle("Starting...")
		a.serverButton.SetSensitive(false)
	})

	err := a.serverMgr.Start()

	glib.IdleAdd(func() {
		a.serverButton.SetSensitive(true)
		if err != nil {
			a.statusRow.SetSubtitle(fmt.Sprintf("Error: %v", err))
			a.serverButton.SetLabel("Start Server")
			a.serverButton.RemoveCSSClass("destructive-action")
			a.serverButton.AddCSSClass("suggested-action")
			a.showToast(fmt.Sprintf("Server failed: %v", err))
		} else {
			a.statusRow.SetSubtitle("Running")
			a.serverButton.SetLabel("Stop Server")
			a.serverButton.RemoveCSSClass("suggested-action")
			a.serverButton.AddCSSClass("destructive-action")

			// Update client URL
			url := a.serverMgr.ServerURL()
			a.client.SetBaseURL(url)
			a.serverURLEntry.SetText(url)

			a.showToast("Server started")
			go a.refreshModels()
		}
	})
}

func (a *App) stopServer() {
	glib.IdleAdd(func() {
		a.serverButton.SetSensitive(false)
		a.statusRow.SetSubtitle("Stopping...")
	})

	_ = a.serverMgr.Stop()

	glib.IdleAdd(func() {
		a.serverButton.SetSensitive(true)
		a.statusRow.SetSubtitle("Stopped")
		a.serverButton.SetLabel("Start Server")
		a.serverButton.RemoveCSSClass("destructive-action")
		a.serverButton.AddCSSClass("suggested-action")
		a.showToast("Server stopped")
	})
}

func (a *App) refreshModels() {
	// Try API first (returns models from running server)
	var names []string
	url := a.serverURLEntry.Text()
	client := apiclient.New(url)
	if resp, err := client.ListModels(context.Background()); err == nil {
		for _, m := range resp.Data {
			names = append(names, m.ID)
		}
	}

	// Fall back to scanning the models directory
	if len(names) == 0 {
		names = scanLocalModels()
	}

	glib.IdleAdd(func() {
		a.modelList = names

		display := names
		if len(display) == 0 {
			display = []string{"(no models)"}
			a.modelDropdown.SetSensitive(false)
		} else {
			a.modelDropdown.SetSensitive(true)
		}

		a.modelDropdown.SetModel(gtk.NewStringList(display))

		if len(a.modelList) > 0 && a.selectedModel == "" {
			a.selectedModel = a.modelList[0]
		}
	})
}

// scanLocalModels reads .gguf filenames from the models directory.
func scanLocalModels() []string {
	var names []string
	filepath.Walk(modelsDir(), func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(info.Name()), ".gguf") {
			name := strings.TrimSuffix(info.Name(), filepath.Ext(info.Name()))
			names = append(names, name)
		}
		return nil
	})
	return names
}
