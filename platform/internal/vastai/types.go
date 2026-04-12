package vastai

// Instance represents a vast.ai instance.
type Instance struct {
	ID        int     `json:"id"`
	Status    string  `json:"actual_status"`
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
	GPUName      string
	MinGPURAM    float64
	MaxCostPerHr float64
	MinDiskGB    float64
}

// CreateOpts defines options for creating a vast.ai instance.
type CreateOpts struct {
	Image   string
	DiskGB  float64
	OnStart string
}
