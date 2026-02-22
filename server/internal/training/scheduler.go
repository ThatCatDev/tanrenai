package training

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/thatcatdev/tanrenai/server/internal/config"
)

// ScheduleConfig is the schedule.json format for periodic training.
type ScheduleConfig struct {
	Interval        string `json:"interval"`          // e.g. "24h"
	MinNewEntries   int    `json:"min_new_entries"`    // minimum new memories before training
	AutoMerge       bool   `json:"auto_merge"`         // auto-merge after training
	LastRunAt       string `json:"last_run_at"`        // RFC3339 timestamp of last run
	LastMemoryCount int    `json:"last_memory_count"`  // memory count at last run
}

// DefaultScheduleConfig returns sensible defaults.
func DefaultScheduleConfig() ScheduleConfig {
	return ScheduleConfig{
		Interval:      "24h",
		MinNewEntries: 50,
		AutoMerge:     false,
	}
}

// Scheduler runs periodic fine-tuning in the background.
type Scheduler struct {
	manager   *Manager
	baseModel string
	cfg       ScheduleConfig
	cancel    context.CancelFunc
	wg        sync.WaitGroup
}

// NewScheduler creates a Scheduler.
func NewScheduler(manager *Manager, baseModel string) *Scheduler {
	cfg := loadScheduleConfig()
	return &Scheduler{
		manager:   manager,
		baseModel: baseModel,
		cfg:       cfg,
	}
}

// Start begins the periodic training loop.
func (s *Scheduler) Start(interval time.Duration) {
	ctx, cancel := context.WithCancel(context.Background())
	s.cancel = cancel

	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				s.tick(ctx)
			}
		}
	}()
}

// Stop stops the scheduler and waits for the background goroutine to exit.
func (s *Scheduler) Stop() {
	if s.cancel != nil {
		s.cancel()
		s.wg.Wait()
	}
}

func (s *Scheduler) tick(ctx context.Context) {
	// Check if enough new memories have been added since the last run.
	currentCount := s.manager.memStore.Count()
	newEntries := currentCount - s.cfg.LastMemoryCount

	if newEntries < s.cfg.MinNewEntries {
		return
	}

	log.Printf("[scheduler] %d new memories since last run, starting fine-tune", newEntries)

	runCfg := DefaultRunConfig()
	run, err := s.manager.Prepare(ctx, s.baseModel, runCfg)
	if err != nil {
		log.Printf("[scheduler] prepare failed: %v", err)
		return
	}

	if err := s.manager.Train(ctx, run.ID); err != nil {
		log.Printf("[scheduler] train failed: %v", err)
		return
	}

	// Update schedule state
	s.cfg.LastRunAt = time.Now().Format(time.RFC3339)
	s.cfg.LastMemoryCount = currentCount
	saveScheduleConfig(s.cfg)

	log.Printf("[scheduler] training run %s started", run.ID)
}

func schedulePath() string {
	return filepath.Join(config.TrainingDir(), "schedule.json")
}

func loadScheduleConfig() ScheduleConfig {
	data, err := os.ReadFile(schedulePath())
	if err != nil {
		return DefaultScheduleConfig()
	}
	var cfg ScheduleConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return DefaultScheduleConfig()
	}
	return cfg
}

func saveScheduleConfig(cfg ScheduleConfig) {
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return
	}
	dir := filepath.Dir(schedulePath())
	os.MkdirAll(dir, 0755)
	os.WriteFile(schedulePath(), data, 0644)
}

// SaveScheduleConfig persists the schedule config (exported for use in cmd).
func SaveScheduleConfig(cfg ScheduleConfig) {
	saveScheduleConfig(cfg)
}

// LoadScheduleConfig loads the schedule config from disk.
func LoadScheduleConfig() ScheduleConfig {
	return loadScheduleConfig()
}

// FormatScheduleInfo returns a human-readable summary of the schedule state.
func FormatScheduleInfo(cfg ScheduleConfig) string {
	return fmt.Sprintf("interval=%s min_new=%d auto_merge=%v last_run=%s last_count=%d",
		cfg.Interval, cfg.MinNewEntries, cfg.AutoMerge, cfg.LastRunAt, cfg.LastMemoryCount)
}
