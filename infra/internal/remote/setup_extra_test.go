package remote

import (
	"strings"
	"testing"
)

func TestGPUServerSetupStagesCommandContents(t *testing.T) {
	stages := GPUServerSetupStages(nil, "", 11435)

	stageMap := make(map[string]SetupStage)
	for _, s := range stages {
		stageMap[s.Name] = s
	}

	// storage-setup: should reference tanrenai directory
	storage, ok := stageMap["storage-setup"]
	if !ok {
		t.Fatal("missing storage-setup stage")
	}
	allCmds := strings.Join(storage.Commands, "\n")
	if !strings.Contains(allCmds, "tanrenai") {
		t.Error("storage-setup should reference tanrenai directory")
	}
	if !strings.Contains(allCmds, "mkdir") {
		t.Error("storage-setup should use mkdir")
	}

	// verify-binaries: should check llama-server and tanrenai-gpu
	verify, ok := stageMap["verify-binaries"]
	if !ok {
		t.Fatal("missing verify-binaries stage")
	}
	verifyCmds := strings.Join(verify.Commands, "\n")
	if !strings.Contains(verifyCmds, "llama-server") {
		t.Error("verify-binaries should check llama-server")
	}
	if !strings.Contains(verifyCmds, "tanrenai-gpu") {
		t.Error("verify-binaries should check tanrenai-gpu")
	}

	// start-gpu-server: should reference the configured port
	start, ok := stageMap["start-gpu-server"]
	if !ok {
		t.Fatal("missing start-gpu-server stage")
	}
	startCmds := strings.Join(start.Commands, "\n")
	if !strings.Contains(startCmds, "11435") {
		t.Errorf("start-gpu-server should reference port 11435, got: %q", startCmds[:100])
	}
	if !strings.Contains(startCmds, "tanrenai-gpu serve") {
		t.Error("start-gpu-server should invoke tanrenai-gpu serve")
	}
}

func TestGPUServerSetupStagesCustomPort(t *testing.T) {
	stages := GPUServerSetupStages(nil, "", 9999)

	var startStage *SetupStage
	for i := range stages {
		if stages[i].Name == "start-gpu-server" {
			startStage = &stages[i]
			break
		}
	}
	if startStage == nil {
		t.Fatal("missing start-gpu-server stage")
	}

	allCmds := strings.Join(startStage.Commands, "\n")
	if !strings.Contains(allCmds, "9999") {
		t.Error("start-gpu-server should use custom port 9999")
	}
	if strings.Contains(allCmds, "11435") {
		t.Error("start-gpu-server should not contain default port 11435 when custom port is set")
	}
}

func TestGPUServerSetupStagesWithModel(t *testing.T) {
	stages := GPUServerSetupStages(nil, "qwen2.5:72b", 11435)

	// With a model: storage, verify-binaries, pull-model, start-gpu
	if len(stages) != 4 {
		t.Fatalf("got %d stages, want 4 (with model, no network)", len(stages))
	}

	var pullStage *SetupStage
	for i := range stages {
		if stages[i].Name == "pull-model" {
			pullStage = &stages[i]
			break
		}
	}
	if pullStage == nil {
		t.Fatal("expected pull-model stage when model is specified")
	}

	allCmds := strings.Join(pullStage.Commands, "\n")
	if !strings.Contains(allCmds, "qwen2.5:72b") {
		t.Errorf("pull-model should reference model name, got: %q", allCmds)
	}
	if !strings.Contains(allCmds, "tanrenai-gpu pull") {
		t.Error("pull-model should invoke tanrenai-gpu pull")
	}
}

func TestGPUServerSetupStagesModelPullIsNonFatal(t *testing.T) {
	stages := GPUServerSetupStages(nil, "some-model", 11435)

	for _, s := range stages {
		if s.Name == "pull-model" {
			allCmds := strings.Join(s.Commands, "\n")
			if !strings.Contains(allCmds, "|| true") {
				t.Error("pull-model command should use '|| true' to be non-fatal")
			}
			return
		}
	}
	t.Fatal("pull-model stage not found")
}

func TestGPUServerSetupStagesNetworkCommandsIncluded(t *testing.T) {
	networkCmds := []string{
		"curl -fsSL https://tailscale.com/install.sh | sh",
		"tailscale up --authkey tskey-123 --hostname gpu-node",
	}
	stages := GPUServerSetupStages(networkCmds, "", 11435)

	var networkStage *SetupStage
	for i := range stages {
		if stages[i].Name == "setup-network" {
			networkStage = &stages[i]
			break
		}
	}
	if networkStage == nil {
		t.Fatal("expected setup-network stage when network commands are provided")
	}
	if len(networkStage.Commands) != len(networkCmds) {
		t.Errorf("setup-network has %d commands, want %d", len(networkStage.Commands), len(networkCmds))
	}
	for i, cmd := range networkCmds {
		if networkStage.Commands[i] != cmd {
			t.Errorf("setup-network.Commands[%d] = %q, want %q", i, networkStage.Commands[i], cmd)
		}
	}
}

func TestGPUServerSetupStagesOrderWithModelAndNetwork(t *testing.T) {
	networkCmds := []string{"tailscale up"}
	stages := GPUServerSetupStages(networkCmds, "mymodel", 11435)

	// Expected: storage, verify-binaries, pull-model, setup-network, start-gpu
	if len(stages) != 5 {
		t.Fatalf("got %d stages, want 5", len(stages))
	}

	expectedOrder := []string{
		"storage-setup",
		"verify-binaries",
		"pull-model",
		"setup-network",
		"start-gpu-server",
	}
	for i, want := range expectedOrder {
		if stages[i].Name != want {
			t.Errorf("stages[%d].Name = %q, want %q", i, stages[i].Name, want)
		}
	}
}

func TestGPUServerSetupStagesStartServerAlwaysLast(t *testing.T) {
	stages := GPUServerSetupStages(nil, "", 11435)
	last := stages[len(stages)-1]
	if last.Name != "start-gpu-server" {
		t.Errorf("last stage = %q, want \"start-gpu-server\"", last.Name)
	}

	stages = GPUServerSetupStages(nil, "somemodel", 11435)
	last = stages[len(stages)-1]
	if last.Name != "start-gpu-server" {
		t.Errorf("last stage with model = %q, want \"start-gpu-server\"", last.Name)
	}

	stages = GPUServerSetupStages([]string{"cmd"}, "", 11435)
	last = stages[len(stages)-1]
	if last.Name != "start-gpu-server" {
		t.Errorf("last stage with network = %q, want \"start-gpu-server\"", last.Name)
	}

	stages = GPUServerSetupStages([]string{"cmd"}, "somemodel", 11435)
	last = stages[len(stages)-1]
	if last.Name != "start-gpu-server" {
		t.Errorf("last stage with model+network = %q, want \"start-gpu-server\"", last.Name)
	}
}

func TestGPUServerSetupStagesNoModel(t *testing.T) {
	stages := GPUServerSetupStages(nil, "", 11435)
	for _, s := range stages {
		if s.Name == "pull-model" {
			t.Error("should not have pull-model stage when model is empty")
		}
	}
}

func TestGPUServerSetupStagesNoNetwork(t *testing.T) {
	stages := GPUServerSetupStages(nil, "", 11435)
	for _, s := range stages {
		if s.Name == "setup-network" {
			t.Error("should not have setup-network stage when network commands are empty")
		}
	}
}

func TestDefaultGPUImage(t *testing.T) {
	if DefaultGPUImage == "" {
		t.Fatal("DefaultGPUImage should not be empty")
	}
	if !strings.Contains(DefaultGPUImage, "tanrenai-gpu") {
		t.Errorf("DefaultGPUImage = %q, should contain tanrenai-gpu", DefaultGPUImage)
	}
}
