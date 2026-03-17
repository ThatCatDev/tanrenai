package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// Choice represents what the user selected.
type Choice struct {
	Instance *vastai.Instance // nil means "create new"
}

// PickInstance shows an interactive picker for existing instances or creating a new one.
// Returns the user's choice.
func PickInstance(instances []vastai.Instance) (*Choice, error) {
	m := newPickerModel(instances)
	p := tea.NewProgram(m)
	result, err := p.Run()
	if err != nil {
		return nil, err
	}
	final := result.(pickerModel)
	if final.quit {
		return nil, fmt.Errorf("cancelled")
	}

	return &final.choice, nil
}

type pickerModel struct {
	instances []vastai.Instance
	cursor    int
	choice    Choice
	quit      bool
}

func newPickerModel(instances []vastai.Instance) pickerModel {
	return pickerModel{instances: instances}
}

func (m pickerModel) Init() tea.Cmd {
	return nil
}

func (m pickerModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			m.quit = true

			return m, tea.Quit
		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			}
		case "down", "j":
			if m.cursor < len(m.instances) {
				m.cursor++
			}
		case "enter":
			if m.cursor < len(m.instances) {
				inst := m.instances[m.cursor]
				m.choice = Choice{Instance: &inst}
			} else {
				m.choice = Choice{Instance: nil}
			}

			return m, tea.Quit
		}
	}

	return m, nil
}

func (m pickerModel) View() string {
	var b strings.Builder
	b.WriteString("Select an instance to deploy to:\n\n")

	for i, inst := range m.instances {
		cursor := "  "
		if m.cursor == i {
			cursor = "> "
		}
		fmt.Fprintf(&b, "%s%d  %-10s  %-20s  x%d  $%.3f/hr",
			cursor, inst.ID, inst.Status, inst.GPUName, inst.NumGPUs, inst.CostPerHr)
		if inst.SSHHost != "" {
			fmt.Fprintf(&b, "  %s:%d", inst.SSHHost, inst.SSHPort)
		}
		b.WriteString("\n")
	}

	// "Create new" option
	cursor := "  "
	if m.cursor == len(m.instances) {
		cursor = "> "
	}
	fmt.Fprintf(&b, "\n%s[Create new instance]\n", cursor)

	b.WriteString("\n↑/↓ navigate • enter select • q quit\n")

	return b.String()
}
