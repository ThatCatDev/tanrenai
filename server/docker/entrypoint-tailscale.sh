#!/bin/sh
set -e

# Start tailscaled in the background
tailscaled --state=/var/lib/tailscale/tailscaled.state --socket=/var/run/tailscale/tailscaled.sock &

# Wait for socket
sleep 2

# Join the network
if [ -n "$TAILSCALE_AUTH_KEY" ]; then
    TAILSCALE_ARGS="--authkey=$TAILSCALE_AUTH_KEY"
    if [ -n "$TAILSCALE_LOGIN_SERVER" ]; then
        TAILSCALE_ARGS="$TAILSCALE_ARGS --login-server=$TAILSCALE_LOGIN_SERVER"
    fi
    if [ -n "$TAILSCALE_HOSTNAME" ]; then
        TAILSCALE_ARGS="$TAILSCALE_ARGS --hostname=$TAILSCALE_HOSTNAME"
    fi
    tailscale up $TAILSCALE_ARGS
    echo "Tailscale connected: $(tailscale ip -4)"
fi

# Run the server
exec tanrenai-server "$@"
