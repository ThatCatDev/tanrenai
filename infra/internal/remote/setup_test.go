package remote

import "testing"

func TestGPUServerSetupStages(t *testing.T) {
	stages := GPUServerSetupStages(nil, "", 11435)

	// Without network commands and model, should have: install-deps, cuda-check, build-llama-cpp, build-tanrenai-gpu, start-gpu-server
	if len(stages) != 5 {
		t.Fatalf("got %d stages, want 5", len(stages))
	}

	names := []string{"install-deps", "cuda-check", "build-llama-cpp", "build-tanrenai-gpu", "start-gpu-server"}
	for i, want := range names {
		if stages[i].Name != want {
			t.Errorf("stage[%d].Name = %q, want %q", i, stages[i].Name, want)
		}
	}
}

func TestGPUServerSetupStagesWithNetwork(t *testing.T) {
	networkCmds := []string{"curl -fsSL https://tailscale.com/install.sh | sh", "tailscale up --authkey key"}
	stages := GPUServerSetupStages(networkCmds, "qwen2.5:7b", 11435)

	// Should have: install-deps, cuda-check, build-llama-cpp, build-tanrenai-gpu, pull-model, setup-network, start-gpu-server
	if len(stages) != 7 {
		t.Fatalf("got %d stages, want 7", len(stages))
	}

	if stages[4].Name != "pull-model" {
		t.Errorf("stage[4].Name = %q, want \"pull-model\"", stages[4].Name)
	}
	if stages[5].Name != "setup-network" {
		t.Errorf("stage[5].Name = %q, want \"setup-network\"", stages[5].Name)
	}
	if len(stages[5].Commands) != 2 {
		t.Errorf("setup-network has %d commands, want 2", len(stages[5].Commands))
	}
}
