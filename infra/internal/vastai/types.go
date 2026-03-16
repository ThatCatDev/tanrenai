package vastai

// Instance represents a vast.ai instance.
type Instance struct {
	ID        int     `json:"id"`
	Status    string  `json:"actual_status"` // running, exited, loading, etc.
	SSHHost   string  `json:"ssh_host"`
	SSHPort   int     `json:"ssh_port"`
	CurState  string  `json:"cur_state"`
	GPUName   string  `json:"gpu_name"`
	NumGPUs   int     `json:"num_gpus"`
	CostPerHr float64 `json:"dph_total"`
}

// Offer represents a vast.ai machine offer.
type Offer struct {
	ID          int     `json:"id"`
	GPUName     string  `json:"gpu_name"`
	NumGPUs     int     `json:"num_gpus"`
	GPURAMTotal float64 `json:"gpu_ram"`
	CostPerHr   float64 `json:"dph_total"`
	DiskSpace   float64 `json:"disk_space"`
	Reliability float64 `json:"reliability"`
	DLPerf      float64 `json:"dlperf"`
	CUDAVersion float64 `json:"cuda_max_good"`
}

// SearchQuery defines filters for searching vast.ai offers.
type SearchQuery struct {
	GPUName      string  // filter by GPU name (e.g. "A100", "RTX_4090")
	MinGPURAM    float64 // minimum GPU RAM in GB
	MaxCostPerHr float64 // max $/hr
	MinDiskGB    float64 // minimum disk space
}

// CreateOpts defines options for creating a vast.ai instance.
type CreateOpts struct {
	Image   string  // Docker image (e.g. "nvidia/cuda:12.4.1-devel-ubuntu22.04")
	DiskGB  float64 // disk space to allocate
	OnStart string  // onstart script
}
