package ui

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/diamondburned/gotk4-adwaita/pkg/adw"
	"github.com/diamondburned/gotk4/pkg/gio/v2"
	"github.com/diamondburned/gotk4/pkg/glib/v2"
	"github.com/diamondburned/gotk4/pkg/gtk/v4"
)

// showDownloadDialog opens a dialog with a URL entry for downloading a model.
func (a *App) showDownloadDialog() {
	entry := gtk.NewEntry()
	entry.SetPlaceholderText("https://huggingface.co/...")

	d := adw.NewAlertDialog("Download Model", "Enter a HuggingFace GGUF URL:")
	d.AddResponse("cancel", "Cancel")
	d.AddResponse("download", "Download")
	d.SetResponseAppearance("download", adw.ResponseSuggested)
	d.SetDefaultResponse("download")
	d.SetCloseResponse("cancel")
	d.SetExtraChild(entry)

	d.ConnectResponse(func(response string) {
		if response == "download" {
			url := entry.Buffer().Text()
			if url != "" {
				go a.startDownload(url)
			}
		}
	})

	d.Present(a.window)
}

// startDownload begins downloading a model by URL and updates the progress bar.
func (a *App) startDownload(url string) {
	ctx, cancel := context.WithCancel(context.Background())

	glib.IdleAdd(func() {
		a.cancelDownload = cancel
		a.showDownloadProgress()
	})

	events, err := a.client.PullModel(ctx, url, "")
	if err != nil {
		cancel()
		glib.IdleAdd(func() {
			a.hideDownloadProgress()
			a.showToast(fmt.Sprintf("Download failed: %v", err))
		})
		return
	}

	for ev := range events {
		if ev.Err != nil {
			glib.IdleAdd(func() {
				a.hideDownloadProgress()
				a.showToast(fmt.Sprintf("Download error: %v", ev.Err))
			})
			cancel()
			return
		}

		switch ev.Event.Status {
		case "downloading":
			pct := ev.Event.Percent
			glib.IdleAdd(func() {
				a.downloadProgress.SetFraction(float64(pct) / 100.0)
				a.downloadProgress.SetText(fmt.Sprintf("%d%%", pct))
			})
		case "downloaded":
			glib.IdleAdd(func() {
				a.hideDownloadProgress()
				a.showToast("Model downloaded successfully")
				go a.refreshModels()
			})
			return
		case "error":
			errMsg := ev.Event.Error
			glib.IdleAdd(func() {
				a.hideDownloadProgress()
				a.showToast(fmt.Sprintf("Download error: %s", errMsg))
			})
			return
		}
	}
}

// showDownloadProgress shows the download progress bar in the sidebar.
func (a *App) showDownloadProgress() {
	if a.downloadBox == nil {
		a.downloadProgress = gtk.NewProgressBar()
		a.downloadProgress.SetShowText(true)

		a.downloadCancel = gtk.NewButtonFromIconName("process-stop-symbolic")
		a.downloadCancel.SetTooltipText("Cancel download")
		a.downloadCancel.AddCSSClass("flat")
		a.downloadCancel.ConnectClicked(func() {
			if a.cancelDownload != nil {
				a.cancelDownload()
				a.cancelDownload = nil
				a.hideDownloadProgress()
				a.showToast("Download cancelled")
			}
		})

		a.downloadBox = gtk.NewBox(gtk.OrientationHorizontal, 6)
		a.downloadBox.SetMarginStart(12)
		a.downloadBox.SetMarginEnd(12)
		a.downloadProgress.SetHExpand(true)
		a.downloadBox.Append(a.downloadProgress)
		a.downloadBox.Append(a.downloadCancel)
	}

	a.downloadProgress.SetFraction(0)
	a.downloadProgress.SetText("0%")

	if a.settingsContent != nil {
		a.settingsContent.Append(a.downloadBox)
	}
	a.downloadBox.SetVisible(true)
}

// hideDownloadProgress hides the download progress bar.
func (a *App) hideDownloadProgress() {
	if a.downloadBox != nil {
		a.downloadBox.SetVisible(false)
		if a.settingsContent != nil {
			a.settingsContent.Remove(a.downloadBox)
		}
	}
	a.cancelDownload = nil
}

// showImportDialog opens a file chooser for importing a local .gguf model.
func (a *App) showImportDialog() {
	fd := gtk.NewFileDialog()
	fd.SetTitle("Import Model File")
	fd.SetModal(true)

	filter := gtk.NewFileFilter()
	filter.SetName("GGUF Models")
	filter.AddSuffix("gguf")
	fd.SetDefaultFilter(filter)

	fd.Open(context.Background(), &a.window.Window, func(res gio.AsyncResulter) {
		// OpenFinish can panic when the dialog is dismissed
		defer func() { recover() }()

		file, err := fd.OpenFinish(res)
		if err != nil || file == nil {
			return
		}

		srcPath := file.Path()
		if srcPath == "" {
			return
		}
		go a.importModel(srcPath)
	})
}

// modelsDir returns the path to the tanrenai models directory.
func modelsDir() string {
	if dir := os.Getenv("TANRENAI_MODELS_DIR"); dir != "" {
		return dir
	}
	dataDir := os.Getenv("TANRENAI_DATA_DIR")
	if dataDir == "" {
		home, _ := os.UserHomeDir()
		dataDir = filepath.Join(home, ".local", "share", "tanrenai")
	}
	return filepath.Join(dataDir, "models")
}

// importModel creates a symlink from the source .gguf file into the models directory.
// If the file is already in the models directory, it does nothing.
func (a *App) importModel(srcPath string) {
	filename := filepath.Base(srcPath)
	if !strings.HasSuffix(strings.ToLower(filename), ".gguf") {
		glib.IdleAdd(func() {
			a.showToast("Only .gguf files can be imported")
		})
		return
	}

	mDir := modelsDir()

	// If the source is already inside the models directory, nothing to do
	absSrc, _ := filepath.Abs(srcPath)
	absModels, _ := filepath.Abs(mDir)
	if strings.HasPrefix(absSrc, absModels+string(filepath.Separator)) {
		glib.IdleAdd(func() {
			a.showToast(fmt.Sprintf("%s is already available", strings.TrimSuffix(filename, filepath.Ext(filename))))
			go a.refreshModels()
		})
		return
	}

	destPath := filepath.Join(mDir, filename)

	// Check if a file with the same name already exists
	if _, err := os.Lstat(destPath); err == nil {
		glib.IdleAdd(func() {
			a.showToast(fmt.Sprintf("%s is already available", strings.TrimSuffix(filename, filepath.Ext(filename))))
		})
		return
	}

	if err := os.MkdirAll(mDir, 0755); err != nil {
		glib.IdleAdd(func() {
			a.showToast(fmt.Sprintf("Import error: %v", err))
		})
		return
	}

	if err := os.Symlink(srcPath, destPath); err != nil {
		glib.IdleAdd(func() {
			a.showToast(fmt.Sprintf("Import error: %v", err))
		})
		return
	}

	glib.IdleAdd(func() {
		a.showToast(fmt.Sprintf("Imported %s", strings.TrimSuffix(filename, filepath.Ext(filename))))
		go a.refreshModels()
	})
}
