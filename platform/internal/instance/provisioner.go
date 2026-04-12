package instance

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/ThatCatDev/tanrenai/platform/internal/database"
	"github.com/ThatCatDev/tanrenai/platform/internal/network"
	"github.com/ThatCatDev/tanrenai/platform/internal/vastai"
)

const gpuPort = 11435

// Provisioner handles the full lifecycle of creating a vast.ai instance.
type Provisioner struct {
	db        *database.DB
	image     string // Docker image for GPU instances
	encKey    string // encryption key for decrypting vast.ai API keys
	headscale *network.HeadscaleProvider // nil if Headscale not configured
}

// NewProvisioner creates a new instance provisioner.
func NewProvisioner(db *database.DB, image, encKey string, headscale *network.HeadscaleProvider) *Provisioner {
	return &Provisioner{db: db, image: image, encKey: encKey, headscale: headscale}
}

// Provision creates a new GPU instance for the user based on model requirements.
func (p *Provisioner) Provision(ctx context.Context, user *database.User, modelSize string) (*database.Instance, error) {
	if user.VastaiKeyEnc == nil {
		return nil, fmt.Errorf("no vast.ai API key configured — set one via POST /api/user/vastai-key")
	}

	apiKey, err := database.Decrypt(user.VastaiKeyEnc, p.encKey)
	if err != nil {
		return nil, fmt.Errorf("decrypt vast.ai API key: %w", err)
	}

	client := vastai.NewClient(string(apiKey))

	// Estimate resources
	vram, err := vastai.VRAMForModelSize(modelSize)
	if err != nil {
		return nil, fmt.Errorf("estimate VRAM: %w", err)
	}
	disk, err := vastai.DiskForModelSize(modelSize)
	if err != nil {
		return nil, fmt.Errorf("estimate disk: %w", err)
	}

	slog.Info("provisioning instance",
		"user", user.Email, "model_size", modelSize,
		"vram_gb", vram, "disk_gb", disk,
		"max_cost", user.MaxCostPerHr, "gpu_pref", user.PreferredGPU)

	// Create DB record
	inst := &database.Instance{
		UserID:         user.ID,
		Status:         "provisioning",
		ProvisionState: "searching",
	}
	if err := p.db.CreateInstance(ctx, inst); err != nil {
		return nil, fmt.Errorf("create instance record: %w", err)
	}

	// Search for offers
	offers, err := client.SearchOffers(ctx, vastai.SearchQuery{
		GPUName:      user.PreferredGPU,
		MinGPURAM:    vram,
		MaxCostPerHr: user.MaxCostPerHr,
		MinDiskGB:    disk,
	})
	if err != nil {
		_ = p.db.UpdateInstanceStatus(ctx, inst.ID, "destroyed", "failed")
		return nil, fmt.Errorf("search offers: %w", err)
	}
	if len(offers) == 0 {
		_ = p.db.UpdateInstanceStatus(ctx, inst.ID, "destroyed", "failed")
		return nil, fmt.Errorf("no GPU offers found matching requirements (%.0f GB VRAM, %.0f GB disk, $%.2f/hr max)", vram, disk, user.MaxCostPerHr)
	}

	offer := offers[0]
	slog.Info("selected offer", "offer_id", offer.ID, "gpu", offer.GPUName, "cost", offer.CostPerHr)

	// Generate Headscale auth key and OnStart script
	_ = p.db.UpdateInstanceStatus(ctx, inst.ID, "provisioning", "creating")

	createOpts := vastai.CreateOpts{
		Image:  p.image,
		DiskGB: disk,
	}

	// Unique hostname for this instance on the Headscale network
	hostname := fmt.Sprintf("gpu-%s-%s", sanitizeHostname(user.Email), inst.ID[:8])

	if p.headscale != nil {
		slog.Info("generating Headscale auth key", "hostname", hostname)
		authKey, err := p.headscale.GenerateAuthKey(ctx)
		if err != nil {
			_ = p.db.UpdateInstanceStatus(ctx, inst.ID, "destroyed", "failed")
			return nil, fmt.Errorf("generate Headscale auth key: %w", err)
		}
		createOpts.OnStart = p.headscale.OnStartScript(authKey, hostname, gpuPort)
	}

	// Create vast.ai instance
	vastInst, err := client.CreateInstance(ctx, offer.ID, createOpts)
	if err != nil {
		_ = p.db.UpdateInstanceStatus(ctx, inst.ID, "destroyed", "failed")
		return nil, fmt.Errorf("create vast.ai instance: %w", err)
	}

	_ = p.db.UpdateInstanceGPU(ctx, inst.ID, vastInst.ID, offer.GPUName, "", offer.CostPerHr)
	inst.VastaiInstanceID = vastInst.ID
	inst.GPUName = offer.GPUName
	inst.CostPerHr = offer.CostPerHr

	// Resolve GPU URL via Headscale or direct IP
	_ = p.db.UpdateInstanceStatus(ctx, inst.ID, "provisioning", "booting")

	var gpuURL string
	if p.headscale != nil {
		slog.Info("waiting for Headscale peer", "hostname", hostname)
		peerCtx, peerCancel := context.WithTimeout(ctx, 5*time.Minute)
		defer peerCancel()

		ip, err := p.headscale.WaitForPeer(peerCtx, hostname)
		if err != nil {
			slog.Error("Headscale peer not found", "error", err, "hostname", hostname)
			_ = p.db.UpdateInstanceStatus(ctx, inst.ID, "provisioning", "booting")
			return inst, fmt.Errorf("Headscale peer %q not found: %w", hostname, err)
		}
		gpuURL = fmt.Sprintf("http://%s:%d", ip, gpuPort)
	} else {
		gpuURL = fmt.Sprintf("http://%s:%d", vastInst.SSHHost, gpuPort)
	}

	_ = p.db.UpdateInstanceGPU(ctx, inst.ID, vastInst.ID, offer.GPUName, gpuURL, offer.CostPerHr)
	inst.GPUURL = gpuURL

	// Wait for GPU server health
	if err := p.waitForHealthy(ctx, gpuURL, 10*time.Minute); err != nil {
		slog.Error("GPU server not healthy", "error", err, "gpu_url", gpuURL)
		_ = p.db.UpdateInstanceStatus(ctx, inst.ID, "provisioning", "booting")
		return inst, fmt.Errorf("GPU server not healthy: %w", err)
	}

	_ = p.db.UpdateInstanceStatus(ctx, inst.ID, "running", "ready")
	inst.Status = "running"
	inst.ProvisionState = "ready"

	slog.Info("instance provisioned", "gpu", offer.GPUName, "cost", offer.CostPerHr, "url", gpuURL, "hostname", hostname)
	return inst, nil
}

// Destroy destroys a user's active instance.
func (p *Provisioner) Destroy(ctx context.Context, user *database.User, inst *database.Instance) error {
	if inst.VastaiInstanceID == 0 {
		_ = p.db.DestroyInstance(ctx, inst.ID)
		return nil
	}

	if user.VastaiKeyEnc == nil {
		_ = p.db.DestroyInstance(ctx, inst.ID)
		return fmt.Errorf("no vast.ai API key to destroy instance")
	}

	apiKey, err := database.Decrypt(user.VastaiKeyEnc, p.encKey)
	if err != nil {
		_ = p.db.DestroyInstance(ctx, inst.ID)
		return fmt.Errorf("decrypt API key: %w", err)
	}

	client := vastai.NewClient(string(apiKey))
	instID := fmt.Sprintf("%d", inst.VastaiInstanceID)

	_ = p.db.UpdateInstanceStatus(ctx, inst.ID, "destroying", "")

	slog.Info("destroying instance", "vastai_id", inst.VastaiInstanceID)
	if err := client.DestroyInstance(ctx, instID); err != nil {
		slog.Error("failed to destroy vast.ai instance", "error", err, "id", instID)
	}

	_ = p.db.DestroyInstance(ctx, inst.ID)
	return nil
}

func (p *Provisioner) waitForHealthy(ctx context.Context, gpuURL string, timeout time.Duration) error {
	deadline := time.After(timeout)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	healthURL := gpuURL + "/health"

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-deadline:
			return fmt.Errorf("timeout after %s", timeout)
		case <-ticker.C:
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
			if err != nil {
				continue
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				continue
			}
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
	}
}

// sanitizeHostname converts an email to a valid hostname component.
func sanitizeHostname(email string) string {
	// Take the part before @ and replace non-alphanumeric chars
	parts := make([]byte, 0, len(email))
	for i := 0; i < len(email) && i < 16; i++ {
		c := email[i]
		if c == '@' {
			break
		}
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') {
			parts = append(parts, c)
		} else if c >= 'A' && c <= 'Z' {
			parts = append(parts, c+32) // lowercase
		} else {
			parts = append(parts, '-')
		}
	}
	return string(parts)
}
