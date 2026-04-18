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

// DefaultGPUImage is the pre-built Docker image containing llama-server (CUDA)
// and tanrenai-gpu. Built by CI and pushed to Docker Hub.
const DefaultGPUImage = "thatcatdev/tanrenai-gpu:latest"

// GPUServerSetupStages returns the stages to set up a GPU server on a remote host.
func GPUServerSetupStages(networkCmds []string, model string, gpuPort int) []SetupStage {
	stages := []SetupStage{
		{
			Name: "storage-setup",
			Commands: []string{
				`bash -c '
mkdir -p /root/.local/share/tanrenai

# Find the best writable storage for models
# Prefer persistent disk over RAM-backed tmpfs
BEST_PATH=""
BEST_AVAIL=0
BEST_PERSISTENT=0

for MP in /root/.local/share /workspace /mnt/* /data /scratch /dev/shm /tmp; do
  [ -d "$MP" ] || continue
  # Check if writable
  touch "$MP/.tanrenai-test" 2>/dev/null || continue
  rm -f "$MP/.tanrenai-test"
  # Get available space in KB
  AVAIL=$(df --output=avail "$MP" 2>/dev/null | tail -1 | tr -d " ")
  [ -z "$AVAIL" ] && continue
  # Must have at least 50GB free
  [ "$AVAIL" -gt 50000000 ] || continue
  # Check if persistent (not tmpfs/devtmpfs)
  FSTYPE=$(df --output=fstype "$MP" 2>/dev/null | tail -1 | tr -d " ")
  PERSISTENT=0
  case "$FSTYPE" in tmpfs|devtmpfs) PERSISTENT=0 ;; *) PERSISTENT=1 ;; esac
  # Prefer persistent storage, then largest
  if [ "$PERSISTENT" -gt "$BEST_PERSISTENT" ] || \
     ([ "$PERSISTENT" -eq "$BEST_PERSISTENT" ] && [ "$AVAIL" -gt "$BEST_AVAIL" ]); then
    BEST_AVAIL=$AVAIL
    BEST_PATH=$MP
    BEST_PERSISTENT=$PERSISTENT
  fi
done

if [ -n "$BEST_PATH" ]; then
  mkdir -p "$BEST_PATH/tanrenai-models"
  rm -rf /root/.local/share/tanrenai/models
  ln -sfn "$BEST_PATH/tanrenai-models" /root/.local/share/tanrenai/models
  AVAIL_GB=$((BEST_AVAIL / 1024 / 1024))
  FSTYPE=$(df --output=fstype "$BEST_PATH" 2>/dev/null | tail -1 | tr -d " ")
  echo "Models dir: $BEST_PATH/tanrenai-models (${AVAIL_GB}GB free, $FSTYPE)"
else
  mkdir -p /root/.local/share/tanrenai/models
  echo "Models dir: /root/.local/share/tanrenai/models (overlay — WARNING: limited space)"
fi
'`,
			},
		},
		{
			Name: "verify-binaries",
			Commands: []string{
				// Image already has llama-server and tanrenai-gpu installed.
				// Just ensure the tanrenai data dirs and symlinks are in place.
				`mkdir -p /root/.local/share/tanrenai/bin`,
				`ln -sf /usr/local/bin/llama-server /root/.local/share/tanrenai/bin/llama-server`,
				`llama-server --version`,
				`tanrenai-gpu version`,
			},
		},
	}

	if model != "" {
		stages = append(stages, SetupStage{
			Name: "pull-model",
			Commands: []string{
				"mkdir -p /root/.local/share/tanrenai/models",
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
		_, _ = fmt.Fprintf(output, "\n=== %s ===\n", stage.Name)

		cmd := strings.Join(stage.Commands, " && ")
		if err := ssh.Run(ctx, cmd, output); err != nil {
			return fmt.Errorf("stage %q failed: %w", stage.Name, err)
		}
	}

	return nil
}
