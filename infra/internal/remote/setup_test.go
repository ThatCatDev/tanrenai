package remote

import "testing"

func TestGPUServerSetupStages(t *testing.T) {
	stages := GPUServerSetupStages(nil, "", 11435)

	// storage-setup, install-deps, install-gpu-binaries, start-gpu-server
	if len(stages) != 4 {
		t.Fatalf("got %d stages, want 4", len(stages))
	}

	names := []string{"storage-setup", "install-deps", "install-gpu-binaries", "start-gpu-server"}
	for i, want := range names {
		if stages[i].Name != want {
			t.Errorf("stage[%d].Name = %q, want %q", i, stages[i].Name, want)
		}
	}
}

func TestGPUServerSetupStagesWithNetwork(t *testing.T) {
	networkCmds := []string{"curl -fsSL https://tailscale.com/install.sh | sh", "tailscale up --authkey key"}
	stages := GPUServerSetupStages(networkCmds, "qwen2.5:7b", 11435)

	// storage-setup, install-deps, install-gpu-binaries, pull-model, setup-network, start-gpu-server
	if len(stages) != 6 {
		t.Fatalf("got %d stages, want 6", len(stages))
	}

	if stages[3].Name != "pull-model" {
		t.Errorf("stage[3].Name = %q, want \"pull-model\"", stages[3].Name)
	}
	if stages[4].Name != "setup-network" {
		t.Errorf("stage[4].Name = %q, want \"setup-network\"", stages[4].Name)
	}
	if len(stages[4].Commands) != 2 {
		t.Errorf("setup-network has %d commands, want 2", len(stages[4].Commands))
	}
}
