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
				"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git build-essential curl wget",
				// Install recent cmake (distro version is often too old)
				"[ -f /usr/local/bin/cmake ] || (curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz | tar -xzf - -C /usr/local --strip-components=1)",
			},
		},
		{
			Name: "cuda-check",
			Commands: []string{
				`CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')`,
				`NVCC_VER=$(/usr/local/cuda/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')`,
				`echo "GPU arch: compute_$CUDA_ARCH, NVCC: $NVCC_VER"`,
				// Upgrade CUDA toolkit if nvcc is too old for the GPU
				// compute_120 (Blackwell) needs CUDA 12.8+, compute_100 (Hopper) needs 12.0+
				`CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.') && ` +
					`NVCC_MAJOR=$(/usr/local/cuda/bin/nvcc --version | grep -oP 'release \K[0-9]+') && ` +
					`NVCC_MINOR=$(/usr/local/cuda/bin/nvcc --version | grep -oP 'release [0-9]+\.\K[0-9]+') && ` +
					`NEED_UPGRADE=0 && ` +
					`if [ "$CUDA_ARCH" -ge 120 ] && [ "$NVCC_MAJOR" -eq 12 ] && [ "$NVCC_MINOR" -lt 8 ]; then NEED_UPGRADE=1; fi && ` +
					`if [ "$NEED_UPGRADE" -eq 1 ]; then ` +
					`echo "Upgrading CUDA toolkit for compute_$CUDA_ARCH..." && ` +
					`wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && ` +
					`dpkg -i cuda-keyring_1.1-1_all.deb && ` +
					`apt-get update -qq && ` +
					`DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-12-8 && ` +
					`export PATH=/usr/local/cuda-12.8/bin:$PATH && ` +
					`ln -sfn /usr/local/cuda-12.8 /usr/local/cuda && ` +
					`echo "Upgraded to $(nvcc --version | grep release)"; ` +
					`else echo "CUDA toolkit OK"; fi`,
			},
		},
		{
			Name: "build-llama-cpp",
			Commands: []string{
				`if [ -f /usr/local/bin/.llama-server-cuda ]; then echo "llama-server (CUDA) already built, skipping"; exit 0; fi`,
				"[ -d /opt/llama.cpp ] && cd /opt/llama.cpp && git pull || git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
				`CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')`,
				"rm -rf /opt/llama.cpp/build",
				`cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"`,
				"cd /opt/llama.cpp && cmake --build build --config Release -j$(nproc) -t llama-server",
				"cp /opt/llama.cpp/build/bin/llama-server /usr/local/bin/",
				"touch /usr/local/bin/.llama-server-cuda",
			},
		},
		{
			Name: "build-tanrenai-gpu",
			Commands: []string{
				"[ -f /usr/local/go/bin/go ] || curl -fsSL https://go.dev/dl/go1.25.0.linux-amd64.tar.gz | tar -C /usr/local -xzf -",
				"export PATH=$PATH:/usr/local/go/bin",
				"[ -d /opt/tanrenai ] && cd /opt/tanrenai && git fetch && git reset --hard origin/main || git clone --depth 1 https://github.com/ThatCatDev/tanrenai.git /opt/tanrenai",
				"cd /opt/tanrenai/gpu && /usr/local/go/bin/go clean -cache 2>/dev/null; /usr/local/go/bin/go build -o /usr/local/bin/tanrenai-gpu .",
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
			"killall tanrenai-gpu 2>/dev/null; sleep 1",
			fmt.Sprintf("sh -c 'nohup /usr/local/bin/tanrenai-gpu serve --host 0.0.0.0 --port %d > /var/log/tanrenai-gpu.log 2>&1 &'", gpuPort),
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
