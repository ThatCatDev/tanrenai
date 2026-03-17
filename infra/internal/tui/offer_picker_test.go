package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// TestPickOfferNoOffers verifies PickOffer returns an error when no offers are provided.
func TestPickOfferNoOffers(t *testing.T) {
	_, err := PickOffer([]vastai.Offer{})
	if err == nil {
		t.Error("PickOffer() should return error when no offers are available")
	}
	if !strings.Contains(err.Error(), "no offers") {
		t.Errorf("error should mention no offers, got: %v", err)
	}
}

// TestOfferPickerModelCtrlC verifies ctrl+c sets quit=true.
func TestOfferPickerModelCtrlC(t *testing.T) {
	offers := []vastai.Offer{{ID: 1, GPUName: "RTX 4090"}}
	m := newOfferPickerModel(offers)

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	updated := result.(offerPickerModel)
	if !updated.quit {
		t.Error("ctrl+c should set quit=true")
	}
}

// TestOfferPickerModelDownArrow verifies the down arrow key moves the cursor.
func TestOfferPickerModelDownArrow(t *testing.T) {
	offers := []vastai.Offer{{ID: 1}, {ID: 2}}
	m := newOfferPickerModel(offers)

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyDown})
	updated := result.(offerPickerModel)
	if updated.cursor != 1 {
		t.Errorf("cursor after KeyDown = %d, want 1", updated.cursor)
	}
}

// TestOfferPickerModelUpArrow verifies the up arrow key moves the cursor.
func TestOfferPickerModelUpArrow(t *testing.T) {
	offers := []vastai.Offer{{ID: 1}, {ID: 2}}
	m := newOfferPickerModel(offers)
	m.cursor = 1

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyUp})
	updated := result.(offerPickerModel)
	if updated.cursor != 0 {
		t.Errorf("cursor after KeyUp = %d, want 0", updated.cursor)
	}
}

// TestOfferPickerModelNavigateUpAtTopWithArrow verifies up arrow at top stays at 0.
func TestOfferPickerModelNavigateUpAtTopWithArrow(t *testing.T) {
	offers := []vastai.Offer{{ID: 1}, {ID: 2}}
	m := newOfferPickerModel(offers)

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyUp})
	updated := result.(offerPickerModel)
	if updated.cursor != 0 {
		t.Errorf("cursor should stay at 0 when at top, got %d", updated.cursor)
	}
}

// TestOfferPickerModelNavigateDownAtBottomWithArrow verifies down arrow at bottom stays.
func TestOfferPickerModelNavigateDownAtBottomWithArrow(t *testing.T) {
	offers := []vastai.Offer{{ID: 1}, {ID: 2}}
	m := newOfferPickerModel(offers)
	m.cursor = 1 // at last item

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyDown})
	updated := result.(offerPickerModel)
	if updated.cursor != 1 {
		t.Errorf("cursor should stay at 1 when at bottom, got %d", updated.cursor)
	}
}

// TestOfferPickerModelSelectFirstOffer verifies enter selects the first offer.
func TestOfferPickerModelSelectFirstOffer(t *testing.T) {
	offers := []vastai.Offer{
		{ID: 10, GPUName: "RTX 4090"},
		{ID: 20, GPUName: "A100"},
	}
	m := newOfferPickerModel(offers)
	// cursor=0 by default

	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
	updated := result.(offerPickerModel)
	if updated.choice.Offer == nil {
		t.Fatal("expected offer to be selected")
	}
	if updated.choice.Offer.ID != 10 {
		t.Errorf("selected offer ID = %d, want 10", updated.choice.Offer.ID)
	}
}

// TestOfferPickerModelViewScrollIndicators verifies scroll indicators appear when offset > 0.
func TestOfferPickerModelViewScrollIndicators(t *testing.T) {
	// Create enough offers to require scrolling
	offers := make([]vastai.Offer, 20)
	for i := range offers {
		offers[i] = vastai.Offer{ID: i + 1, GPUName: "RTX 4090"}
	}
	m := newOfferPickerModel(offers)
	m.offset = 2 // scrolled down
	m.cursor = 5

	view := m.View()
	if !strings.Contains(view, "↑ more") {
		t.Error("View() should show '↑ more' when offset > 0")
	}
	if !strings.Contains(view, "↓") {
		t.Error("View() should show '↓ N more' when there are items below")
	}
}

// TestOfferPickerModelViewRAMConversion verifies GPU RAM > 500 is converted from MB to GB.
func TestOfferPickerModelViewRAMConversion(t *testing.T) {
	offers := []vastai.Offer{
		{ID: 1, GPUName: "A100", GPURAMTotal: 81920, CostPerHr: 1.5, Reliability: 0.99}, // 80 GB in MB
	}
	m := newOfferPickerModel(offers)
	view := m.View()

	// 81920 MB / 1024 = 80 GB
	if !strings.Contains(view, "80") {
		t.Errorf("View() should show 80 GB for 81920 MB RAM, got: %q", view)
	}
}

// TestOfferPickerModelViewRAMSmall verifies GPU RAM <= 500 is shown as-is (already in GB).
func TestOfferPickerModelViewRAMSmall(t *testing.T) {
	offers := []vastai.Offer{
		{ID: 1, GPUName: "RTX 4090", GPURAMTotal: 24, CostPerHr: 0.5, Reliability: 0.99},
	}
	m := newOfferPickerModel(offers)
	view := m.View()

	if !strings.Contains(view, "24") {
		t.Errorf("View() should show 24 GB for small GPURAMTotal, got: %q", view)
	}
}

// TestOfferPickerModelViewCursorIndicator verifies cursor indicator is shown for selected item.
func TestOfferPickerModelViewCursorIndicator(t *testing.T) {
	offers := []vastai.Offer{
		{ID: 1, GPUName: "RTX 4090", CostPerHr: 0.5},
		{ID: 2, GPUName: "A100", CostPerHr: 1.0},
	}
	m := newOfferPickerModel(offers)
	m.cursor = 0

	view := m.View()
	if !strings.Contains(view, "> ") {
		t.Error("View() should show '> ' cursor indicator for selected item")
	}
}

// TestOfferPickerModelViewNavigateHelp verifies the navigation help text is shown.
func TestOfferPickerModelViewNavigateHelp(t *testing.T) {
	offers := []vastai.Offer{{ID: 1, GPUName: "RTX 4090"}}
	m := newOfferPickerModel(offers)
	view := m.View()

	if !strings.Contains(view, "navigate") {
		t.Errorf("View() should contain navigation help, got: %q", view)
	}
	if !strings.Contains(view, "select") {
		t.Errorf("View() should contain select help, got: %q", view)
	}
}

// TestOfferPickerModelUnknownKeyIgnored verifies unknown keys don't change state.
func TestOfferPickerModelUnknownKeyIgnored(t *testing.T) {
	offers := []vastai.Offer{{ID: 1}, {ID: 2}}
	m := newOfferPickerModel(offers)
	m.cursor = 0

	result, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("x")})
	updated := result.(offerPickerModel)
	if updated.cursor != 0 {
		t.Errorf("unknown key should not change cursor, got %d", updated.cursor)
	}
	if cmd != nil {
		t.Error("unknown key should return nil cmd")
	}
}

// TestOfferPickerModelScrollUpOffset verifies scrolling up adjusts offset when cursor < offset.
func TestOfferPickerModelScrollUpOffset(t *testing.T) {
	offers := make([]vastai.Offer, 15)
	for i := range offers {
		offers[i] = vastai.Offer{ID: i + 1}
	}
	m := newOfferPickerModel(offers)
	m.cursor = 5
	m.offset = 5 // cursor at top of viewport

	// Press up — cursor goes to 4, which is < offset (5), so offset should adjust
	result, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("k")})
	updated := result.(offerPickerModel)
	if updated.cursor != 4 {
		t.Errorf("cursor after up = %d, want 4", updated.cursor)
	}
	if updated.offset != 4 {
		t.Errorf("offset should adjust to %d, got %d", 4, updated.offset)
	}
}

// TestNewOfferPickerModelInitialState verifies initial state.
func TestNewOfferPickerModelInitialState(t *testing.T) {
	offers := []vastai.Offer{
		{ID: 1, GPUName: "RTX 4090"},
		{ID: 2, GPUName: "A100"},
	}
	m := newOfferPickerModel(offers)

	if m.cursor != 0 {
		t.Errorf("initial cursor = %d, want 0", m.cursor)
	}
	if m.offset != 0 {
		t.Errorf("initial offset = %d, want 0", m.offset)
	}
	if m.quit {
		t.Error("initial quit should be false")
	}
	if m.choice.Offer != nil {
		t.Error("initial choice.Offer should be nil")
	}
	if len(m.offers) != 2 {
		t.Errorf("offers length = %d, want 2", len(m.offers))
	}
}
