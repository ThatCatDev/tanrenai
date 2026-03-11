package ui

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/diamondburned/gotk4-adwaita/pkg/adw"
	coreglib "github.com/diamondburned/gotk4/pkg/core/glib"
	"github.com/diamondburned/gotk4/pkg/glib/v2"
	"github.com/diamondburned/gotk4/pkg/gtk/v4"

	"github.com/ThatCatDev/tanrenai/desktop/internal/server"
)

// buildSettings builds the settings view as a toolbar view (shown in the content stack).
func (a *App) buildSettings() *adw.ToolbarView {
	// Server status row
	a.statusRow = adw.NewActionRow()
	a.statusRow.SetTitle("Status")
	if a.serverMgr.Status() == server.StatusRunning {
		a.statusRow.SetSubtitle("Running")
	} else {
		a.statusRow.SetSubtitle("Stopped")
	}

	// Start/Stop button
	a.serverButton = gtk.NewButtonWithLabel("Start Server")
	if a.serverMgr.Status() == server.StatusRunning {
		a.serverButton.SetLabel("Stop Server")
		a.serverButton.AddCSSClass("destructive-action")
	} else {
		a.serverButton.AddCSSClass("suggested-action")
	}
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
			if a.activeSession != nil {
				a.activeSession.Model = a.selectedModel
				a.updateModelBadge()
			}
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
	a.serverURLEntry.SetText(a.client.BaseURL())

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
	a.settingsContent = content

	scrolled := gtk.NewScrolledWindow()
	scrolled.SetChild(content)
	scrolled.SetVExpand(true)

	// Back button in header
	header := adw.NewHeaderBar()
	header.SetTitleWidget(gtk.NewLabel("Settings"))

	backBtn := gtk.NewButtonFromIconName("go-previous-symbolic")
	backBtn.SetTooltipText("Back to chat")
	backBtn.ConnectClicked(func() {
		a.showChat()
	})
	header.PackStart(backBtn)

	toolbarView := adw.NewToolbarView()
	toolbarView.AddTopBar(header)
	toolbarView.SetContent(scrolled)

	return toolbarView
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

			url := a.serverMgr.ServerURL()
			a.client.SetBaseURL(url)
			if a.serverURLEntry != nil {
				a.serverURLEntry.SetText(url)
			}

			a.showToast("Server started")
			go a.refreshAndLoadModel()
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
	// Start with local models (instant, no network)
	names := scanLocalModels()

	// Try server API with a short timeout — don't block on a dead server
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if resp, err := a.client.ListModels(ctx); err == nil && len(resp.Data) > 0 {
		names = nil
		for _, m := range resp.Data {
			names = append(names, m.ID)
		}
	}

	glib.IdleAdd(func() {
		a.modelList = names

		display := names
		if len(display) == 0 {
			display = []string{"(no models)"}
		}

		if a.modelDropdown != nil {
			a.modelDropdown.SetSensitive(len(names) > 0)
			a.modelDropdown.SetModel(gtk.NewStringList(display))

			// Restore selection to match selectedModel
			for i, n := range a.modelList {
				if n == a.selectedModel {
					a.modelDropdown.SetSelected(uint(i))
					break
				}
			}
		}

		if len(a.modelList) > 0 && a.selectedModel == "" {
			a.selectedModel = a.modelList[0]
		}
		a.updateModelBadge()
	})
}

// refreshAndLoadModel refreshes the model list and auto-loads the selected model.
func (a *App) refreshAndLoadModel() {
	a.refreshModels()

	model := a.selectedModel
	if model == "" {
		return
	}

	glib.IdleAdd(func() {
		a.statusRow.SetSubtitle("Loading model...")
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if _, err := a.client.LoadModel(ctx, model); err != nil {
		log.Printf("Auto-load model %q: %v", model, err)
		glib.IdleAdd(func() {
			a.statusRow.SetSubtitle("Running")
			a.showToast(fmt.Sprintf("Failed to load %s: %v", model, err))
		})
		return
	}

	glib.IdleAdd(func() {
		a.statusRow.SetSubtitle(fmt.Sprintf("Running (%s)", model))
		a.showToast(fmt.Sprintf("Loaded %s", model))
	})
}

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
