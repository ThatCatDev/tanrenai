package cmd

import (
	"context"
	"log/slog"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/ThatCatDev/tanrenai/platform/internal/config"
	"github.com/ThatCatDev/tanrenai/platform/internal/database"
	"github.com/ThatCatDev/tanrenai/platform/internal/server"
)

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the platform service",
	RunE: func(cmd *cobra.Command, args []string) error {
		cfg := config.Defaults()

		if host, _ := cmd.Flags().GetString("host"); host != "" {
			cfg.Host = host
		}
		if port, _ := cmd.Flags().GetInt("port"); port != 0 {
			cfg.Port = port
		}
		if dsn, _ := cmd.Flags().GetString("database-url"); dsn != "" {
			cfg.DatabaseURL = dsn
		}
		if issuer, _ := cmd.Flags().GetString("oidc-issuer"); issuer != "" {
			cfg.OIDCIssuer = issuer
		}
		if clientID, _ := cmd.Flags().GetString("oidc-client-id"); clientID != "" {
			cfg.OIDCClientID = clientID
		}
		if secret, _ := cmd.Flags().GetString("oidc-client-secret"); secret != "" {
			cfg.OIDCClientSecret = secret
		}
		if key, _ := cmd.Flags().GetString("encryption-key"); key != "" {
			cfg.EncryptionKey = key
		}
		if origin, _ := cmd.Flags().GetString("frontend-origin"); origin != "" {
			cfg.FrontendOrigin = origin
		}
		if hsURL, _ := cmd.Flags().GetString("headscale-url"); hsURL != "" {
			cfg.HeadscaleURL = hsURL
		}
		if hsAPI, _ := cmd.Flags().GetString("headscale-api-key"); hsAPI != "" {
			cfg.HeadscaleAPI = hsAPI
		}
		if hsUser, _ := cmd.Flags().GetString("headscale-user"); hsUser != "" {
			cfg.HeadscaleUser = hsUser
		}

		// Generate encryption key if not provided
		if cfg.EncryptionKey == "" {
			key, err := database.GenerateEncryptionKey()
			if err != nil {
				return err
			}
			cfg.EncryptionKey = key
			slog.Warn("no encryption key provided, generated ephemeral key (API keys will not survive restarts)")
		}

		ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
		defer stop()

		// Connect to database
		db, err := database.New(ctx, cfg.DatabaseURL)
		if err != nil {
			return err
		}
		defer db.Close()
		slog.Info("database connected")

		srv := server.New(cfg, db)
		return srv.Start(ctx)
	},
}

func init() {
	serveCmd.Flags().String("host", "", "bind address")
	serveCmd.Flags().Int("port", 0, "listen port")
	serveCmd.Flags().String("database-url", "", "PostgreSQL connection string")
	serveCmd.Flags().String("oidc-issuer", "", "OIDC issuer URL")
	serveCmd.Flags().String("oidc-client-id", "", "OIDC client ID")
	serveCmd.Flags().String("oidc-client-secret", "", "OIDC client secret")
	serveCmd.Flags().String("encryption-key", "", "32-byte hex key for encrypting API keys")
	serveCmd.Flags().String("frontend-origin", "", "frontend URL for CORS")
	serveCmd.Flags().String("headscale-url", "", "Headscale server URL")
	serveCmd.Flags().String("headscale-api-key", "", "Headscale API key")
	serveCmd.Flags().String("headscale-user", "", "Headscale user name")

	rootCmd.AddCommand(serveCmd)
}
