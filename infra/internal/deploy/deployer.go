package deploy

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/ThatCatDev/tanrenai/infra/internal/config"
	"github.com/ThatCatDev/tanrenai/infra/internal/network"
	"github.com/ThatCatDev/tanrenai/infra/internal/remote"
	"github.com/ThatCatDev/tanrenai/infra/internal/tui"
	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// Result holds the output of a successful deploy.
type Result struct {
	InstanceID int
	GPUURL     string // e.g. "http://100.64.0.5:11435" or "http://ssh-host:11435"
	GPUName    string
	CostPerHr  float64
}

// Deployer orchestrates the full GPU server deployment pipeline.
type Deployer struct {
	vastai  *vastai.Client
	network network.Provider
	cfg     config.Config
	output  io.Writer
	verbose bool
}

// New creates a new Deployer.
func New(vastaiClient *vastai.Client, netProvider network.Provider, cfg config.Config, output io.Writer, verbose bool) *Deployer {
	return &Deployer{
		vastai:  vastaiClient,
		network: netProvider,
		cfg:     cfg,
		verbose: verbose,
		output:  output,
	}
}

// Run executes the full deploy pipeline.
func (d *Deployer) Run(ctx context.Context) (*Result, error) {
	// 1. Resolve instance
	inst, err := d.resolveInstance(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolve instance: %w", err)
	}
	fmt.Fprintf(d.output, "Instance %d (%s) — $%.3f/hr\n", inst.ID, inst.GPUName, inst.CostPerHr)

	// 2. Wait for instance to be running with SSH details
	instID := fmt.Sprintf("%d", inst.ID)
	err = tui.RunWithSpinner(d.output, fmt.Sprintf("Waiting for instance %d to be ready", inst.ID), func() error {
		waitCtx, waitCancel := context.WithTimeout(ctx, 5*time.Minute)
		defer waitCancel()

		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-waitCtx.Done():
				return fmt.Errorf("timed out waiting for instance (status: %s)", inst.Status)
			case <-ticker.C:
				updated, err := d.vastai.GetInstance(waitCtx, instID)
				if err != nil {
					slog.Debug("polling instance", "err", err)
					continue
				}
				slog.Debug("instance poll", "status", updated.Status, "ssh_host", updated.SSHHost, "ssh_port", updated.SSHPort)
				if updated.Status == "running" && updated.SSHHost != "" && updated.SSHPort != 0 {
					*inst = *updated
					return nil
				}
			}
		}
	})
	if err != nil {
		return nil, fmt.Errorf("instance not ready: %w", err)
	}

	// 3. Wait for SSH to accept connections
	sshCtx, sshCancel := context.WithTimeout(ctx, 3*time.Minute)
	defer sshCancel()
	err = tui.RunWithSpinner(d.output, fmt.Sprintf("Waiting for SSH on %s:%d", inst.SSHHost, inst.SSHPort), func() error {
		return remote.WaitForSSH(sshCtx, inst.SSHHost, inst.SSHPort)
	})
	if err != nil {
		return nil, fmt.Errorf("SSH not ready: %w", err)
	}

	// 3. Connect via SSH
	var sshClient *remote.SSHClient
	err = tui.RunWithSpinner(d.output, "Connecting via SSH", func() error {
		var connErr error
		sshClient, connErr = remote.Connect(ctx, inst.SSHHost, inst.SSHPort, "root")
		return connErr
	})
	if err != nil {
		return nil, fmt.Errorf("SSH connect: %w", err)
	}
	defer sshClient.Close()

	// 4. Generate network auth key
	var authKey string
	if d.network.Name() != "none" {
		err = tui.RunWithSpinner(d.output, fmt.Sprintf("Generating %s auth key", d.network.Name()), func() error {
			var keyErr error
			authKey, keyErr = d.network.GenerateAuthKey(ctx)
			return keyErr
		})
		if err != nil {
			return nil, fmt.Errorf("generate auth key: %w", err)
		}
	}

	// 5. Setup GPU server
	hostname := fmt.Sprintf("tanrenai-gpu-%d", inst.ID)
	networkCmds := d.network.InstallCommands(authKey, hostname)
	stages := remote.GPUServerSetupStages(networkCmds, d.cfg.Model, d.cfg.GPUPort)

	for _, stage := range stages {
		stageName := stage.Name
		stageRef := stage

		if d.verbose {
			fmt.Fprintf(d.output, "\n=== %s ===\n", stageName)
			if err := remote.RunStages(ctx, sshClient, []remote.SetupStage{stageRef}, d.output); err != nil {
				return nil, fmt.Errorf("setup stage %q: %w", stageName, err)
			}
		} else {
			var buf bytes.Buffer
			err = tui.RunWithSpinner(d.output, fmt.Sprintf("Setup: %s", stageName), func() error {
				return remote.RunStages(ctx, sshClient, []remote.SetupStage{stageRef}, &buf)
			})
			if err != nil {
				// Show captured output on failure
				fmt.Fprintf(d.output, "\n--- %s output ---\n%s\n", stageName, buf.String())
				return nil, fmt.Errorf("setup stage %q: %w", stageName, err)
			}
		}
	}

	// 6. Determine GPU URL
	var gpuIP string
	if d.network.Name() != "none" {
		err = tui.RunWithSpinner(d.output, fmt.Sprintf("Waiting for %s peer %s", d.network.Name(), hostname), func() error {
			peerCtx, peerCancel := context.WithTimeout(ctx, 3*time.Minute)
			defer peerCancel()
			var peerErr error
			gpuIP, peerErr = d.network.WaitForPeer(peerCtx, hostname)
			return peerErr
		})
		if err != nil {
			return nil, fmt.Errorf("wait for peer: %w", err)
		}
	} else {
		gpuIP = inst.SSHHost
	}

	gpuURL := fmt.Sprintf("http://%s:%d", gpuIP, d.cfg.GPUPort)

	// 7. Health check
	err = tui.RunWithSpinner(d.output, fmt.Sprintf("Health check %s/health", gpuURL), func() error {
		return d.healthCheck(ctx, gpuURL)
	})
	if err != nil {
		slog.Warn("health check failed (server may still be starting)", "err", err)
	}

	return &Result{
		InstanceID: inst.ID,
		GPUURL:     gpuURL,
		GPUName:    inst.GPUName,
		CostPerHr:  inst.CostPerHr,
	}, nil
}

func (d *Deployer) resolveInstance(ctx context.Context) (*vastai.Instance, error) {
	if d.cfg.VastaiInstance != "" {
		slog.Info("using existing instance", "id", d.cfg.VastaiInstance)
		inst, err := d.vastai.GetInstance(ctx, d.cfg.VastaiInstance)
		if err != nil {
			return nil, err
		}
		if inst.Status == "exited" {
			fmt.Fprintf(d.output, "Starting stopped instance %d...\n", inst.ID)
			if err := d.vastai.StartInstance(ctx, d.cfg.VastaiInstance); err != nil {
				return nil, fmt.Errorf("start instance: %w", err)
			}
			// Re-fetch to get updated status
			time.Sleep(3 * time.Second)
			inst, err = d.vastai.GetInstance(ctx, d.cfg.VastaiInstance)
			if err != nil {
				return nil, err
			}
		}
		return inst, nil
	}

	// Fetch existing instances and show interactive picker
	fmt.Fprintf(d.output, "Fetching instances...\n")
	instances, err := d.vastai.ListInstances(ctx)
	if err != nil {
		return nil, fmt.Errorf("list instances: %w", err)
	}

	choice, err := tui.PickInstance(instances)
	if err != nil {
		return nil, err
	}

	if choice.Instance != nil {
		inst := choice.Instance
		if inst.Status == "exited" {
			fmt.Fprintf(d.output, "Starting stopped instance %d...\n", inst.ID)
			id := fmt.Sprintf("%d", inst.ID)
			if err := d.vastai.StartInstance(ctx, id); err != nil {
				return nil, fmt.Errorf("start instance: %w", err)
			}
			time.Sleep(3 * time.Second)
			inst, err = d.vastai.GetInstance(ctx, id)
			if err != nil {
				return nil, err
			}
		}
		return inst, nil
	}

	// User chose "Create new instance"
	return d.createNewInstance(ctx)
}

func (d *Deployer) createNewInstance(ctx context.Context) (*vastai.Instance, error) {
	searchLabel := fmt.Sprintf("Searching offers (%.0f GB RAM, %.0f GB disk, $%.2f/hr max)", d.cfg.MinGPURAM, d.cfg.DiskGB, d.cfg.MaxCostPerHr)
	if d.cfg.GPUName != "" {
		searchLabel = fmt.Sprintf("Searching offers (%s, %.0f GB RAM, %.0f GB disk, $%.2f/hr max)", d.cfg.GPUName, d.cfg.MinGPURAM, d.cfg.DiskGB, d.cfg.MaxCostPerHr)
	}

	var offers []vastai.Offer
	err := tui.RunWithSpinner(d.output, searchLabel, func() error {
		var searchErr error
		offers, searchErr = d.vastai.SearchOffers(ctx, vastai.SearchQuery{
			GPUName:      d.cfg.GPUName,
			MinGPURAM:    d.cfg.MinGPURAM,
			MaxCostPerHr: d.cfg.MaxCostPerHr,
			MinDiskGB:    d.cfg.DiskGB,
		})
		return searchErr
	})
	if err != nil {
		return nil, fmt.Errorf("search offers: %w", err)
	}
	if len(offers) == 0 {
		return nil, fmt.Errorf("no offers found matching criteria (try --max-cost or --min-gpu-ram)")
	}

	choice, err := tui.PickOffer(offers)
	if err != nil {
		return nil, err
	}

	var inst *vastai.Instance
	err = tui.RunWithSpinner(d.output,
		fmt.Sprintf("Creating instance from offer %d (%s, %.0f GB disk)", choice.Offer.ID, choice.Offer.GPUName, d.cfg.DiskGB),
		func() error {
			var createErr error
			inst, createErr = d.vastai.CreateInstance(ctx, choice.Offer.ID, vastai.CreateOpts{
				DiskGB: d.cfg.DiskGB,
			})
			return createErr
		})
	if err != nil {
		return nil, fmt.Errorf("create instance: %w", err)
	}

	return inst, nil
}

func (d *Deployer) healthCheck(ctx context.Context, gpuURL string) error {
	hctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-hctx.Done():
			return hctx.Err()
		case <-ticker.C:
			req, err := http.NewRequestWithContext(hctx, http.MethodGet, gpuURL+"/health", nil)
			if err != nil {
				continue
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				continue
			}
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
	}
}
