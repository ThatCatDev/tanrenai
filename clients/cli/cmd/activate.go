package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/spf13/cobra"
)

var activateCmd = &cobra.Command{
	Use:   "activate <code>",
	Short: "Redeem an invite code to activate your plan",
	Long: "Redeem a single-use activation code to start your trial. " +
		"You need an active plan before you can use the hosted GPU service.",
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		body, err := json.Marshal(map[string]string{"code": args[0]})
		if err != nil {
			return err
		}
		result, err := platformPost("/api/activate", bytes.NewReader(body))
		if err != nil {
			return err
		}
		tier, _ := result["tier"].(string)
		if tier == "" {
			tier = "your plan"
		}
		fmt.Printf("Activated — you're on the %s plan. You can now run `tanrenai run`.\n", tier)
		return nil
	},
}

func init() {
	rootCmd.AddCommand(activateCmd)
}
