package deploy

import (
	"bytes"
	"testing"

	"github.com/ThatCatDev/tanrenai/infra/internal/config"
	"github.com/ThatCatDev/tanrenai/infra/internal/network"
	"github.com/ThatCatDev/tanrenai/infra/internal/vastai"
)

func TestNewDeployer(t *testing.T) {
	client := vastai.NewClient("test-key")
	net := network.NewNoneProvider()
	cfg := config.Defaults()
	var buf bytes.Buffer

	d := New(client, net, cfg, &buf, false)
	if d == nil {
		t.Fatal("New() returned nil")
	}
	if d.vastai == nil {
		t.Error("deployer vastai client is nil")
	}
	if d.network == nil {
		t.Error("deployer network provider is nil")
	}
	if d.output == nil {
		t.Error("deployer output writer is nil")
	}
	if d.verbose != false {
		t.Error("deployer verbose should be false")
	}
}

func TestNewDeployerVerbose(t *testing.T) {
	client := vastai.NewClient("key")
	net := network.NewNoneProvider()
	cfg := config.Defaults()
	var buf bytes.Buffer

	d := New(client, net, cfg, &buf, true)
	if !d.verbose {
		t.Error("deployer verbose should be true")
	}
}

func TestResultFields(t *testing.T) {
	r := Result{
		InstanceID: 12345,
		GPUURL:     "http://100.64.0.1:11435",
		GPUName:    "RTX 4090",
		CostPerHr:  0.456,
	}
	if r.InstanceID != 12345 {
		t.Errorf("InstanceID = %d, want 12345", r.InstanceID)
	}
	if r.GPUURL != "http://100.64.0.1:11435" {
		t.Errorf("GPUURL = %q, want \"http://100.64.0.1:11435\"", r.GPUURL)
	}
	if r.GPUName != "RTX 4090" {
		t.Errorf("GPUName = %q, want \"RTX 4090\"", r.GPUName)
	}
	if r.CostPerHr != 0.456 {
		t.Errorf("CostPerHr = %f, want 0.456", r.CostPerHr)
	}
}

func TestNewDeployerConfigPropagated(t *testing.T) {
	client := vastai.NewClient("key")
	net := network.NewHeadscaleProvider("https://hs.example.com", "apikey", "user")
	cfg := config.Config{
		GPUPort:      9999,
		Model:        "qwen2.5:72b",
		Network:      "headscale",
		MaxCostPerHr: 2.5,
	}
	var buf bytes.Buffer

	d := New(client, net, cfg, &buf, false)
	if d.cfg.GPUPort != 9999 {
		t.Errorf("cfg.GPUPort = %d, want 9999", d.cfg.GPUPort)
	}
	if d.cfg.Model != "qwen2.5:72b" {
		t.Errorf("cfg.Model = %q, want \"qwen2.5:72b\"", d.cfg.Model)
	}
	if d.cfg.MaxCostPerHr != 2.5 {
		t.Errorf("cfg.MaxCostPerHr = %f, want 2.5", d.cfg.MaxCostPerHr)
	}
	if d.network.Name() != "headscale" {
		t.Errorf("network.Name() = %q, want \"headscale\"", d.network.Name())
	}
}
