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
			Name: "storage-setup",
			Commands: []string{
				`bash -c '
mkdir -p /root/.local/share/tanrenai

# Find the largest writable mount point for model storage
# Skip pseudo-filesystems, read-only mounts, and tiny mounts
BEST_PATH=""
BEST_AVAIL=0

for MP in /dev/shm /workspace /mnt/* /data /scratch /tmp; do
  [ -d "$MP" ] || continue
  # Check if writable
  touch "$MP/.tanrenai-test" 2>/dev/null || continue
  rm -f "$MP/.tanrenai-test"
  # Get available space in KB
  AVAIL=$(df --output=avail "$MP" 2>/dev/null | tail -1 | tr -d " ")
  [ -z "$AVAIL" ] && continue
  # Must have at least 50GB free
  if [ "$AVAIL" -gt 50000000 ] && [ "$AVAIL" -gt "$BEST_AVAIL" ]; then
    BEST_AVAIL=$AVAIL
    BEST_PATH=$MP
  fi
done

if [ -n "$BEST_PATH" ]; then
  mkdir -p "$BEST_PATH/tanrenai-models"
  rm -rf /root/.local/share/tanrenai/models
  ln -sfn "$BEST_PATH/tanrenai-models" /root/.local/share/tanrenai/models
  AVAIL_GB=$((BEST_AVAIL / 1024 / 1024))
  echo "Models dir: $BEST_PATH/tanrenai-models (${AVAIL_GB}GB free)"
else
  mkdir -p /root/.local/share/tanrenai/models
  echo "Models dir: /root/.local/share/tanrenai/models (overlay — WARNING: limited space)"
fi
'`,
			},
		},
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
				"rm -rf /opt/tanrenai && git clone --depth 1 https://github.com/ThatCatDev/tanrenai.git /opt/tanrenai",
				"ls /opt/tanrenai/gpu/internal/models/hfresolve.go && echo 'hfresolve.go present' || echo 'WARNING: hfresolve.go MISSING'",
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
			fmt.Sprintf(`python3 -c "
import subprocess, time, os, signal
# Kill old GPU server
try:
    r = subprocess.run(['pgrep', '-f', 'tanrenai-gpu serve'], capture_output=True, text=True)
    for pid in r.stdout.strip().split('\n'):
        if pid and int(pid) != os.getpid():
            print(f'Stopping old GPU server (PID {pid})...')
            os.kill(int(pid), signal.SIGTERM)
    time.sleep(2)
except: pass
# Start new GPU server
print('Starting GPU server on port %d...')
subprocess.Popen(['nohup', '/usr/local/bin/tanrenai-gpu', 'serve', '--host', '0.0.0.0', '--port', '%d'],
    stdout=open('/var/log/tanrenai-gpu.log', 'w'), stderr=subprocess.STDOUT,
    start_new_session=True, preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
time.sleep(3)
import urllib.request
try:
    urllib.request.urlopen('http://localhost:%d/health')
    print('GPU server healthy')
except: print('WARNING: health check failed')
"`, gpuPort, gpuPort, gpuPort),
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
