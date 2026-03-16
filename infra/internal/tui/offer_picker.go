package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

const maxVisible = 10

// OfferChoice represents what the user selected from offers.
type OfferChoice struct {
	Offer *vastai.Offer // nil means cancelled
}

// PickOffer shows an interactive picker for GPU offers.
func PickOffer(offers []vastai.Offer) (*OfferChoice, error) {
	if len(offers) == 0 {
		return nil, fmt.Errorf("no offers available")
	}
	m := newOfferPickerModel(offers)
	p := tea.NewProgram(m)
	result, err := p.Run()
	if err != nil {
		return nil, err
	}
	final := result.(offerPickerModel)
	if final.quit {
		return nil, fmt.Errorf("cancelled")
	}

	return &final.choice, nil
}

type offerPickerModel struct {
	offers []vastai.Offer
	cursor int
	offset int // scroll offset
	choice OfferChoice
	quit   bool
}

func newOfferPickerModel(offers []vastai.Offer) offerPickerModel {
	return offerPickerModel{offers: offers}
}

func (m offerPickerModel) Init() tea.Cmd {
	return nil
}

func (m offerPickerModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			m.quit = true

			return m, tea.Quit
		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
				if m.cursor < m.offset {
					m.offset = m.cursor
				}
			}
		case "down", "j":
			if m.cursor < len(m.offers)-1 {
				m.cursor++
				if m.cursor >= m.offset+maxVisible {
					m.offset = m.cursor - maxVisible + 1
				}
			}
		case "enter":
			offer := m.offers[m.cursor]
			m.choice = OfferChoice{Offer: &offer}

			return m, tea.Quit
		}
	}

	return m, nil
}

func (m offerPickerModel) View() string {
	var b strings.Builder

	total := len(m.offers)
	fmt.Fprintf(&b, "Select a GPU offer (%d available):\n\n", total)
	fmt.Fprintf(&b, "  %-8s  %-22s  %-5s  %-8s  %-10s  %-6s  %s\n",
		"ID", "GPU", "GPUs", "RAM", "$/hr", "CUDA", "Reliability")
	fmt.Fprintf(&b, "  %-8s  %-22s  %-5s  %-8s  %-10s  %-6s  %s\n",
		"──", "───", "────", "───", "────", "────", "───────────")

	end := m.offset + maxVisible
	if end > total {
		end = total
	}

	if m.offset > 0 {
		b.WriteString("  ↑ more\n")
	}

	for i := m.offset; i < end; i++ {
		offer := m.offers[i]
		cursor := "  "
		if m.cursor == i {
			cursor = "> "
		}
		ram := offer.GPURAMTotal
		if ram > 500 {
			ram = ram / 1024 // API returns MB, convert to GB
		}
		fmt.Fprintf(&b, "%s%-8d  %-22s  x%-4d  %-4.0f GB  $%-9.3f  %-6.1f  %.0f%%\n",
			cursor, offer.ID, offer.GPUName, offer.NumGPUs,
			ram, offer.CostPerHr,
			offer.CUDAVersion, offer.Reliability*100)
	}

	if end < total {
		fmt.Fprintf(&b, "  ↓ %d more\n", total-end)
	}

	b.WriteString("\n↑/↓ navigate • enter select • q cancel\n")

	return b.String()
}
