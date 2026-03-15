package remote

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"strings"
)

// SetupStage represents a named setup step to execute over SSH.
type SetupStage struct {
	Name     string
	Commands []string
}

// GPUServerSetupStages returns the stages to set up a GPU server on a remote host.
func GPUServerSetupStages(networkCmds []string, model string, gpuPort int) []SetupStage {
	stages := []SetupStage{
		{
			Name: "install-deps",
			Commands: []string{
				"apt-get update -qq",
				"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git build-essential cmake curl wget",
			},
		},
		{
			Name: "build-llama-cpp",
			Commands: []string{
				"[ -d /opt/llama.cpp ] && cd /opt/llama.cpp && git pull || git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
				"cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native",
				"cd /opt/llama.cpp && cmake --build build --config Release -j$(nproc) -t llama-server",
				"cp /opt/llama.cpp/build/bin/llama-server /usr/local/bin/",
			},
		},
		{
			Name: "build-tanrenai-gpu",
			Commands: []string{
				"[ -f /usr/local/go/bin/go ] || curl -fsSL https://go.dev/dl/go1.25.0.linux-amd64.tar.gz | tar -C /usr/local -xzf -",
				"export PATH=$PATH:/usr/local/go/bin",
				"[ -d /opt/tanrenai ] && cd /opt/tanrenai && git pull || git clone --depth 1 https://github.com/ThatCatDev/tanrenai.git /opt/tanrenai",
				"cd /opt/tanrenai/gpu && /usr/local/go/bin/go build -o /usr/local/bin/tanrenai-gpu .",
			},
		},
	}

	if model != "" {
		stages = append(stages, SetupStage{
			Name: "pull-model",
			Commands: []string{
				fmt.Sprintf("mkdir -p /root/.local/share/tanrenai/models"),
				fmt.Sprintf("/usr/local/bin/tanrenai-gpu pull %s || true", model),
			},
		})
	}

	if len(networkCmds) > 0 {
		stages = append(stages, SetupStage{
			Name:     "setup-network",
			Commands: networkCmds,
		})
	}

	stages = append(stages, SetupStage{
		Name: "start-gpu-server",
		Commands: []string{
			fmt.Sprintf("nohup /usr/local/bin/tanrenai-gpu serve --host 0.0.0.0 --port %d > /var/log/tanrenai-gpu.log 2>&1 &", gpuPort),
			"sleep 2",
			fmt.Sprintf("curl -sf http://localhost:%d/health || echo 'WARNING: health check failed'", gpuPort),
		},
	})

	return stages
}

// RunStages executes setup stages sequentially over SSH.
func RunStages(ctx context.Context, ssh *SSHClient, stages []SetupStage, output io.Writer) error {
	for _, stage := range stages {
		slog.Info("running setup stage", "stage", stage.Name)
		fmt.Fprintf(output, "\n=== %s ===\n", stage.Name)

		cmd := strings.Join(stage.Commands, " && ")
		if err := ssh.Run(ctx, cmd, output); err != nil {
			return fmt.Errorf("stage %q failed: %w", stage.Name, err)
		}
	}
	return nil
}
