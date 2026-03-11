package ui

import "testing"

func TestBuildSettings(t *testing.T) {
	a := newTestApp()
	tv := a.buildSettings()
	if tv == nil {
		t.Fatal("buildSettings returned nil")
	}
}

func TestSettingsWidgetsInitialized(t *testing.T) {
	a := newTestApp()

	if a.statusRow == nil {
		t.Fatal("statusRow not set")
	}
	if a.serverButton == nil {
		t.Fatal("serverButton not set")
	}
	if a.modelDropdown == nil {
		t.Fatal("modelDropdown not set")
	}
	if a.serverURLEntry == nil {
		t.Fatal("serverURLEntry not set")
	}
	if a.settingsContent == nil {
		t.Fatal("settingsContent not set")
	}
}

func TestSettingsStatusRowDefault(t *testing.T) {
	a := newTestApp()
	subtitle := a.statusRow.Subtitle()
	if subtitle != "Stopped" {
		t.Fatalf("expected status 'Stopped', got %q", subtitle)
	}
}

func TestSettingsServerButtonDefault(t *testing.T) {
	a := newTestApp()
	label := a.serverButton.Label()
	if label != "Start Server" {
		t.Fatalf("expected button 'Start Server', got %q", label)
	}
}
