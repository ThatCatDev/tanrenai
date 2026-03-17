package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// --- pickerModel tests ---

func TestPickerModelInit(t *testing.T) {
	instances := []vastai.Instance{
		{ID: 1, Status: "running", GPUName: "RTX 4090"},
	}
	m := newPickerModel(instances)
	cmd := m.Init()
	if cmd != nil {
		t.Error("Init() should return nil cmd")
	}
}

func TestPickerModelView(t *testing.T) {
	instances := []vastai.Instance{
		{ID: 42, Status: "running", GPUName: "RTX 4090", NumGPUs: 1, CostPerHr: 0.5, SSHHost: "1.2.3.4", SSHPort: 22},
	}
	m := newPickerModel(instances)
	view := m.View()

	if !strings.Contains(view, "42") {
		t.Errorf("View() should contain instance ID 42, got: %q", view)
	}
	if !strings.Contains(view, "RTX 4090") {
		t.Errorf("View() should contain GPU name, got: %q", view)
	}
	if !strings.Contains(view, "running") {
		t.Errorf("View() should contain status, got: %q", view)
	}
	if !strings.Contains(view, "1.2.3.4") {
		t.Errorf("View() should contain SSH host, got: %q", view)
	}
	if !strings.Contains(view, "Create new instance") {
		t.Errorf("View() should contain 'Create new instance' option")
	}
}

func TestPickerModelViewNoSSHHost(t *testing.T) {
	instances := []vastai.Instance{
		{ID: 1, Status: "loading", GPUName: "A100"},
	}
	m := newPickerModel(instances)
	view := m.View()
	// Should not contain empty SSH host
	if strings.Contains(view, ":0") {
		t.Errorf("View() should not show :0 for empty SSH info, got: %q", view)
	}
}

func TestPickerModelNavigateDown(t *testing.T) {
	instances := []vastai.Instance{
		{ID: 1}, {ID: 2}, {ID: 3},
	}
	m := newPickerModel(instances)
	if m.cursor != 0 {
		t.Errorf("initial cursor = %d, want 0", m.cursor)
	}

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("j")})
	updated := result.(pickerModel)
	if updated.cursor != 1 {
		t.Errorf("cursor after down = %d, want 1", updated.cursor)
	}
}

func TestPickerModelNavigateUp(t *testing.T) {
	instances := []vastai.Instance{{ID: 1}, {ID: 2}}
	m := newPickerModel(instances)
	m.cursor = 1

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("k")})
	updated := result.(pickerModel)
	if updated.cursor != 0 {
		t.Errorf("cursor after up = %d, want 0", updated.cursor)
	}
}

func TestPickerModelNavigateUpAtTop(t *testing.T) {
	instances := []vastai.Instance{{ID: 1}}
	m := newPickerModel(instances)
	// cursor is already 0, pressing up should keep it at 0

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("k")})
	updated := result.(pickerModel)
	if updated.cursor != 0 {
		t.Errorf("cursor should stay at 0 when already at top, got %d", updated.cursor)
	}
}

func TestPickerModelQuit(t *testing.T) {
	instances := []vastai.Instance{{ID: 1}}
	m := newPickerModel(instances)

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("q")})
	updated := result.(pickerModel)
	if !updated.quit {
		t.Error("pressing q should set quit=true")
	}
}

func TestPickerModelSelectInstance(t *testing.T) {
	instances := []vastai.Instance{
		{ID: 10, Status: "running", GPUName: "A100"},
		{ID: 20, Status: "exited", GPUName: "RTX 4090"},
	}
	m := newPickerModel(instances)
	m.cursor = 1 // select second instance

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	updated := result.(pickerModel)
	if updated.choice.Instance == nil {
		t.Fatal("expected instance to be selected")
	}
	if updated.choice.Instance.ID != 20 {
		t.Errorf("selected instance ID = %d, want 20", updated.choice.Instance.ID)
	}
}

func TestPickerModelSelectCreateNew(t *testing.T) {
	instances := []vastai.Instance{{ID: 1}}
	m := newPickerModel(instances)
	m.cursor = len(instances) // cursor on "Create new"

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	updated := result.(pickerModel)
	if updated.choice.Instance != nil {
		t.Error("selecting 'Create new' should set Instance=nil")
	}
}

func TestPickerModelDownArrow(t *testing.T) {
	instances := []vastai.Instance{{ID: 1}, {ID: 2}}
	m := newPickerModel(instances)

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyDown})
	updated := result.(pickerModel)
	if updated.cursor != 1 {
		t.Errorf("cursor after KeyDown = %d, want 1", updated.cursor)
	}
}

func TestPickerModelUpArrow(t *testing.T) {
	instances := []vastai.Instance{{ID: 1}, {ID: 2}}
	m := newPickerModel(instances)
	m.cursor = 1

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyUp})
	updated := result.(pickerModel)
	if updated.cursor != 0 {
		t.Errorf("cursor after KeyUp = %d, want 0", updated.cursor)
	}
}

// --- offerPickerModel tests ---

func TestOfferPickerModelInit(t *testing.T) {
	offers := []vastai.Offer{{ID: 1, GPUName: "RTX 4090"}}
	m := newOfferPickerModel(offers)
	cmd := m.Init()
	if cmd != nil {
		t.Error("Init() should return nil cmd")
	}
}

func TestOfferPickerModelView(t *testing.T) {
	offers := []vastai.Offer{
		{ID: 5, GPUName: "RTX 4090", NumGPUs: 2, GPURAMTotal: 24576, CostPerHr: 0.45, CUDAVersion: 12.4, Reliability: 0.99},
	}
	m := newOfferPickerModel(offers)
	view := m.View()

	if !strings.Contains(view, "RTX 4090") {
		t.Errorf("View() should contain GPU name, got: %q", view)
	}
	if !strings.Contains(view, "1 available") {
		t.Errorf("View() should show count, got: %q", view)
	}
}

func TestOfferPickerModelNavigateDown(t *testing.T) {
	offers := []vastai.Offer{{ID: 1}, {ID: 2}, {ID: 3}}
	m := newOfferPickerModel(offers)

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("j")})
	updated := result.(offerPickerModel)
	if updated.cursor != 1 {
		t.Errorf("cursor after down = %d, want 1", updated.cursor)
	}
}

func TestOfferPickerModelNavigateUpAtTop(t *testing.T) {
	offers := []vastai.Offer{{ID: 1}, {ID: 2}}
	m := newOfferPickerModel(offers)
	// cursor=0, pressing up should stay at 0

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("k")})
	updated := result.(offerPickerModel)
	if updated.cursor != 0 {
		t.Errorf("cursor should stay at 0 when at top, got %d", updated.cursor)
	}
}

func TestOfferPickerModelNavigateDownAtBottom(t *testing.T) {
	offers := []vastai.Offer{{ID: 1}, {ID: 2}}
	m := newOfferPickerModel(offers)
	m.cursor = 1 // last item

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("j")})
	updated := result.(offerPickerModel)
	if updated.cursor != 1 {
		t.Errorf("cursor should stay at last item, got %d", updated.cursor)
	}
}

func TestOfferPickerModelQuit(t *testing.T) {
	offers := []vastai.Offer{{ID: 1}}
	m := newOfferPickerModel(offers)

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("q")})
	updated := result.(offerPickerModel)
	if !updated.quit {
		t.Error("pressing q should set quit=true")
	}
}

func TestOfferPickerModelSelectOffer(t *testing.T) {
	offers := []vastai.Offer{
		{ID: 100, GPUName: "RTX 4090"},
		{ID: 200, GPUName: "A100"},
	}
	m := newOfferPickerModel(offers)
	m.cursor = 1

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	updated := result.(offerPickerModel)
	if updated.choice.Offer == nil {
		t.Fatal("expected offer to be selected")
	}
	if updated.choice.Offer.ID != 200 {
		t.Errorf("selected offer ID = %d, want 200", updated.choice.Offer.ID)
	}
}

func TestOfferPickerModelScrolling(t *testing.T) {
	// Create more offers than maxVisible (10)
	offers := make([]vastai.Offer, 15)
	for i := range offers {
		offers[i] = vastai.Offer{ID: i + 1, GPUName: "RTX 4090"}
	}
	m := newOfferPickerModel(offers)

	// Navigate down past maxVisible
	for i := 0; i < 11; i++ {
		result, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("j")})
		m = result.(offerPickerModel)
	}

	// Should have scrolled
	if m.offset == 0 {
		t.Error("offset should be > 0 after scrolling past maxVisible items")
	}
}

func TestOfferPickerModelScrollUp(t *testing.T) {
	offers := make([]vastai.Offer, 15)
	for i := range offers {
		offers[i] = vastai.Offer{ID: i + 1}
	}
	m := newOfferPickerModel(offers)
	m.cursor = 11
	m.offset = 2

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("k")})
	updated := result.(offerPickerModel)
	if updated.cursor != 10 {
		t.Errorf("cursor after up = %d, want 10", updated.cursor)
	}
}
