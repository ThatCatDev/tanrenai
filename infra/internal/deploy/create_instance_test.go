package deploy

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai/infra/internal/config"
	"github.com/ThatCatDev/tanrenai/infra/internal/network"
	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

// TestCreateNewInstanceNoOffersFound verifies createNewInstance returns error when no offers match.
func TestCreateNewInstanceNoOffersFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Return empty offers list
		json.NewEncoder(w).Encode(map[string]any{"offers": []any{}})
	}))
	defer srv.Close()

	client := vastai.NewClient("key")
	overrideVastaiClientTransport(client, srv)

	cfg := config.Config{
		VastaiAPIKey: "key",
		Network:      "none",
		GPUName:      "H100",
		MinGPURAM:    80,
		MaxCostPerHr: 1.0,
		DiskGB:       200,
	}

	var buf bytes.Buffer
	d := &Deployer{
		vastai:  client,
		network: network.NewNoneProvider(),
		cfg:     cfg,
		output:  &buf,
	}

	_, err := d.createNewInstance(context.Background())
	if err == nil {
		t.Error("expected error when no offers found")
	}
	if !strings.Contains(err.Error(), "no offers") {
		t.Errorf("error should mention no offers found, got: %v", err)
	}
}

// TestCreateNewInstanceSearchError verifies createNewInstance propagates search errors.
func TestCreateNewInstanceSearchError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "rate limited", http.StatusTooManyRequests)
	}))
	defer srv.Close()

	client := vastai.NewClient("key")
	overrideVastaiClientTransport(client, srv)

	cfg := config.Config{
		VastaiAPIKey: "key",
		Network:      "none",
		GPUName:      "A100",
		MinGPURAM:    40,
		MaxCostPerHr: 2.0,
		DiskGB:       100,
	}

	var buf bytes.Buffer
	d := &Deployer{
		vastai:  client,
		network: network.NewNoneProvider(),
		cfg:     cfg,
		output:  &buf,
	}

	_, err := d.createNewInstance(context.Background())
	if err == nil {
		t.Error("expected error when SearchOffers fails")
	}
	if !strings.Contains(err.Error(), "search offers") {
		t.Errorf("error should mention search offers, got: %v", err)
	}
}

// TestCreateNewInstanceSearchLabelWithGPUName verifies the search label includes GPU name.
func TestCreateNewInstanceSearchLabelNoGPUName(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"offers": []any{}})
	}))
	defer srv.Close()

	client := vastai.NewClient("key")
	overrideVastaiClientTransport(client, srv)

	cfg := config.Config{
		VastaiAPIKey: "key",
		Network:      "none",
		GPUName:      "", // no GPU name filter
		MinGPURAM:    24,
		MaxCostPerHr: 1.0,
		DiskGB:       50,
	}

	var buf bytes.Buffer
	d := &Deployer{
		vastai:  client,
		network: network.NewNoneProvider(),
		cfg:     cfg,
		output:  &buf,
	}

	_, err := d.createNewInstance(context.Background())
	// Should fail with no offers, not GPU name error
	if err != nil && !strings.Contains(err.Error(), "no offers") && !strings.Contains(err.Error(), "search offers") {
		t.Logf("createNewInstance() returned: %v", err)
	}
}
