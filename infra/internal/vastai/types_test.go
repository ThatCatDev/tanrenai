package vastai

import (
	"encoding/json"
	"testing"
)

func TestInstanceJSONRoundtrip(t *testing.T) {
	original := Instance{
		ID:        42,
		Status:    "running",
		SSHHost:   "1.2.3.4",
		SSHPort:   22,
		CurState:  "running",
		GPUName:   "RTX 4090",
		NumGPUs:   1,
		CostPerHr: 0.456,
	}

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}

	var got Instance
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}

	if got.ID != original.ID {
		t.Errorf("ID = %d, want %d", got.ID, original.ID)
	}
	if got.Status != original.Status {
		t.Errorf("Status = %q, want %q", got.Status, original.Status)
	}
	if got.SSHHost != original.SSHHost {
		t.Errorf("SSHHost = %q, want %q", got.SSHHost, original.SSHHost)
	}
	if got.SSHPort != original.SSHPort {
		t.Errorf("SSHPort = %d, want %d", got.SSHPort, original.SSHPort)
	}
	if got.GPUName != original.GPUName {
		t.Errorf("GPUName = %q, want %q", got.GPUName, original.GPUName)
	}
	if got.NumGPUs != original.NumGPUs {
		t.Errorf("NumGPUs = %d, want %d", got.NumGPUs, original.NumGPUs)
	}
	if got.CostPerHr != original.CostPerHr {
		t.Errorf("CostPerHr = %f, want %f", got.CostPerHr, original.CostPerHr)
	}
}

func TestInstanceJSONFieldNames(t *testing.T) {
	// Verify the JSON field names match what vast.ai API sends
	raw := `{
		"id": 99,
		"actual_status": "loading",
		"ssh_host": "10.0.0.1",
		"ssh_port": 2222,
		"cur_state": "loading",
		"gpu_name": "A100 SXM4",
		"num_gpus": 2,
		"dph_total": 1.23
	}`

	var inst Instance
	if err := json.Unmarshal([]byte(raw), &inst); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}
	if inst.ID != 99 {
		t.Errorf("ID = %d, want 99", inst.ID)
	}
	if inst.Status != "loading" {
		t.Errorf("Status = %q, want \"loading\" (maps to actual_status)", inst.Status)
	}
	if inst.SSHHost != "10.0.0.1" {
		t.Errorf("SSHHost = %q, want \"10.0.0.1\"", inst.SSHHost)
	}
	if inst.SSHPort != 2222 {
		t.Errorf("SSHPort = %d, want 2222", inst.SSHPort)
	}
	if inst.NumGPUs != 2 {
		t.Errorf("NumGPUs = %d, want 2", inst.NumGPUs)
	}
	if inst.CostPerHr != 1.23 {
		t.Errorf("CostPerHr = %f, want 1.23", inst.CostPerHr)
	}
}

func TestOfferJSONFieldNames(t *testing.T) {
	raw := `{
		"id": 7,
		"gpu_name": "RTX 3090",
		"num_gpus": 1,
		"gpu_ram": 24576,
		"dph_total": 0.25,
		"disk_space": 500,
		"reliability": 0.99,
		"dlperf": 120.5,
		"cuda_max_good": 12.4
	}`

	var offer Offer
	if err := json.Unmarshal([]byte(raw), &offer); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}
	if offer.ID != 7 {
		t.Errorf("ID = %d, want 7", offer.ID)
	}
	if offer.GPUName != "RTX 3090" {
		t.Errorf("GPUName = %q, want \"RTX 3090\"", offer.GPUName)
	}
	if offer.GPURAMTotal != 24576 {
		t.Errorf("GPURAMTotal = %f, want 24576", offer.GPURAMTotal)
	}
	if offer.CostPerHr != 0.25 {
		t.Errorf("CostPerHr = %f, want 0.25", offer.CostPerHr)
	}
	if offer.DiskSpace != 500 {
		t.Errorf("DiskSpace = %f, want 500", offer.DiskSpace)
	}
	if offer.Reliability != 0.99 {
		t.Errorf("Reliability = %f, want 0.99", offer.Reliability)
	}
	if offer.CUDAVersion != 12.4 {
		t.Errorf("CUDAVersion = %f, want 12.4", offer.CUDAVersion)
	}
}

func TestSearchQueryZeroValues(t *testing.T) {
	var q SearchQuery
	if q.GPUName != "" {
		t.Errorf("zero GPUName = %q, want \"\"", q.GPUName)
	}
	if q.MinGPURAM != 0 {
		t.Errorf("zero MinGPURAM = %f, want 0", q.MinGPURAM)
	}
	if q.MaxCostPerHr != 0 {
		t.Errorf("zero MaxCostPerHr = %f, want 0", q.MaxCostPerHr)
	}
}

func TestCreateOptsDefaults(t *testing.T) {
	var opts CreateOpts
	if opts.Image != "" {
		t.Errorf("zero Image = %q, want \"\"", opts.Image)
	}
	if opts.DiskGB != 0 {
		t.Errorf("zero DiskGB = %f, want 0", opts.DiskGB)
	}
	if opts.OnStart != "" {
		t.Errorf("zero OnStart = %q, want \"\"", opts.OnStart)
	}
}
