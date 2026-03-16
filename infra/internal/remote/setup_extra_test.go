package remote

import (
	"strings"
	"testing"
)

func TestGPUServerSetupStagesCommandContents(t *testing.T) {
	stages := GPUServerSetupStages(nil, "", 11435)

	// Find a stage by name and verify it exists
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

	// install-deps: should install git and build-essential
	deps, ok := stageMap["install-deps"]
	if !ok {
		t.Fatal("missing install-deps stage")
	}
	depsCmds := strings.Join(deps.Commands, "\n")
	if !strings.Contains(depsCmds, "apt-get") {
		t.Error("install-deps should use apt-get")
	}
	if !strings.Contains(depsCmds, "git") {
		t.Error("install-deps should install git")
	}
	if !strings.Contains(depsCmds, "build-essential") {
		t.Error("install-deps should install build-essential")
	}
	if !strings.Contains(depsCmds, "cmake") {
		t.Error("install-deps should install cmake")
	}

	// cuda-check: should reference nvidia-smi
	cuda, ok := stageMap["cuda-check"]
	if !ok {
		t.Fatal("missing cuda-check stage")
	}
	cudaCmds := strings.Join(cuda.Commands, "\n")
	if !strings.Contains(cudaCmds, "nvidia-smi") {
		t.Error("cuda-check should use nvidia-smi")
	}
	if !strings.Contains(cudaCmds, "nvcc") {
		t.Error("cuda-check should check nvcc version")
	}

	// build-llama-cpp: should clone llama.cpp and build with CUDA
	llama, ok := stageMap["build-llama-cpp"]
	if !ok {
		t.Fatal("missing build-llama-cpp stage")
	}
	llamaCmds := strings.Join(llama.Commands, "\n")
	if !strings.Contains(llamaCmds, "llama.cpp") {
		t.Error("build-llama-cpp should reference llama.cpp repo")
	}
	if !strings.Contains(llamaCmds, "DGGML_CUDA=ON") {
		t.Error("build-llama-cpp should build with CUDA")
	}
	if !strings.Contains(llamaCmds, "llama-server") {
		t.Error("build-llama-cpp should build llama-server target")
	}

	// build-tanrenai-gpu: should clone tanrenai and build with go
	tanrenai, ok := stageMap["build-tanrenai-gpu"]
	if !ok {
		t.Fatal("missing build-tanrenai-gpu stage")
	}
	tanrenaiCmds := strings.Join(tanrenai.Commands, "\n")
	if !strings.Contains(tanrenaiCmds, "ThatCatDev/tanrenai") {
		t.Error("build-tanrenai-gpu should clone tanrenai repo")
	}
	if !strings.Contains(tanrenaiCmds, "go build") {
		t.Error("build-tanrenai-gpu should use go build")
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
	// Should NOT contain the default port (it was overridden)
	if strings.Contains(allCmds, "11435") {
		t.Error("start-gpu-server should not contain default port 11435 when custom port is set")
	}
}

func TestGPUServerSetupStagesWithModel(t *testing.T) {
	stages := GPUServerSetupStages(nil, "qwen2.5:72b", 11435)

	// With a model, expect: storage, install-deps, cuda-check, build-llama, build-tanrenai, pull-model, start-gpu
	if len(stages) != 7 {
		t.Fatalf("got %d stages, want 7 (with model, no network)", len(stages))
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

	// Expected order: storage, install-deps, cuda-check, build-llama, build-tanrenai, pull-model, setup-network, start-gpu
	if len(stages) != 8 {
		t.Fatalf("got %d stages, want 8", len(stages))
	}

	expectedOrder := []string{
		"storage-setup",
		"install-deps",
		"cuda-check",
		"build-llama-cpp",
		"build-tanrenai-gpu",
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
	// Without model or network
	stages := GPUServerSetupStages(nil, "", 11435)
	last := stages[len(stages)-1]
	if last.Name != "start-gpu-server" {
		t.Errorf("last stage = %q, want \"start-gpu-server\"", last.Name)
	}

	// With model only
	stages = GPUServerSetupStages(nil, "somemodel", 11435)
	last = stages[len(stages)-1]
	if last.Name != "start-gpu-server" {
		t.Errorf("last stage with model = %q, want \"start-gpu-server\"", last.Name)
	}

	// With network only
	stages = GPUServerSetupStages([]string{"cmd"}, "", 11435)
	last = stages[len(stages)-1]
	if last.Name != "start-gpu-server" {
		t.Errorf("last stage with network = %q, want \"start-gpu-server\"", last.Name)
	}

	// With both model and network
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

func TestGPUServerSetupStagesInstallDepsHasCurlAndWget(t *testing.T) {
	stages := GPUServerSetupStages(nil, "", 11435)

	for _, s := range stages {
		if s.Name == "install-deps" {
			allCmds := strings.Join(s.Commands, "\n")
			if !strings.Contains(allCmds, "curl") {
				t.Error("install-deps should install curl")
			}
			if !strings.Contains(allCmds, "wget") {
				t.Error("install-deps should install wget")
			}

			return
		}
	}
	t.Fatal("install-deps stage not found")
}
