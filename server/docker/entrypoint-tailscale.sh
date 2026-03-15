#!/bin/sh
set -e

echo "=== tanrenai-server entrypoint ==="

mkdir -p /var/run/tailscale /var/lib/tailscale

TS="tailscale --socket=/var/run/tailscale/tailscaled.sock"

# --- Resolve GPU URL before starting tailscale ---

GPU_URL=""

# If no --gpu-url in args, try to discover from Headscale API (direct HTTP, no tunnel needed)
if ! echo "$@" | grep -q "\-\-gpu-url"; then
    if [ -n "$HEADSCALE_API_KEY" ] && [ -n "$HEADSCALE_URL" ]; then
        GPU_HOSTNAME="${GPU_HOSTNAME:-tanrenai-gpu-}"
        echo "Discovering GPU server on Headscale network (matching: $GPU_HOSTNAME)..."
        GPU_NODE=$(curl -sf -H "Authorization: Bearer $HEADSCALE_API_KEY" \
            "$HEADSCALE_URL/api/v1/node" | \
            jq -r ".nodes[] | select(.givenName | startswith(\"$GPU_HOSTNAME\")) | .ipAddresses[0]" | head -1)

        if [ -n "$GPU_NODE" ]; then
            GPU_PORT="${GPU_PORT:-11435}"
            GPU_URL="http://$GPU_NODE:$GPU_PORT"
            echo "Found GPU server: $GPU_URL"
        else
            echo "WARNING: No GPU node matching '$GPU_HOSTNAME*' found on Headscale network"
        fi
    fi
fi

# --- Generate tailscale auth key if needed ---

if [ -z "$TAILSCALE_AUTH_KEY" ] && [ -n "$HEADSCALE_API_KEY" ] && [ -n "$HEADSCALE_URL" ]; then
    echo "Generating auth key from Headscale API..."
    HEADSCALE_USER="${HEADSCALE_USER:-tanrenai}"

    USER_ID=$(curl -sf -H "Authorization: Bearer $HEADSCALE_API_KEY" \
        "$HEADSCALE_URL/api/v1/user" | \
        jq -r ".users[] | select(.name==\"$HEADSCALE_USER\") | .id")

    if [ -z "$USER_ID" ]; then
        echo "ERROR: Headscale user '$HEADSCALE_USER' not found"
        exit 1
    fi
    echo "Found user '$HEADSCALE_USER' (id: $USER_ID)"

    EXPIRY=$(date -u -d "@$(($(date +%s) + 86400))" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)
    TAILSCALE_AUTH_KEY=$(curl -sf -X POST \
        -H "Authorization: Bearer $HEADSCALE_API_KEY" \
        -H "Content-Type: application/json" \
        "$HEADSCALE_URL/api/v1/preauthkey" \
        -d "{\"user\":\"$USER_ID\",\"reusable\":false,\"ephemeral\":true,\"expiration\":\"$EXPIRY\"}" | \
        jq -r '.preAuthKey.key')

    if [ -z "$TAILSCALE_AUTH_KEY" ] || [ "$TAILSCALE_AUTH_KEY" = "null" ]; then
        echo "ERROR: Failed to generate auth key from Headscale"
        exit 1
    fi
    echo "Auth key generated."
    TAILSCALE_LOGIN_SERVER="$HEADSCALE_URL"
fi

# --- Build server args ---

SERVER_ARGS="$@"
if [ -n "$GPU_URL" ]; then
    SERVER_ARGS="$SERVER_ARGS --gpu-url $GPU_URL"
fi

# --- Start the server in the background ---

echo "Starting tanrenai-server..."
tanrenai-server $SERVER_ARGS &
SERVER_PID=$!

# Give the server a moment to bind its port
sleep 1

# --- Now start tailscale (after server port is bound) ---

tailscaled --state=/var/lib/tailscale/tailscaled.state \
    --socket=/var/run/tailscale/tailscaled.sock \
    --tun=userspace-networking 2>&1 &

echo "Waiting for tailscaled..."
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    [ -S /var/run/tailscale/tailscaled.sock ] && break
    sleep 1
done

if [ -n "$TAILSCALE_AUTH_KEY" ]; then
    TAILSCALE_ARGS="--authkey=$TAILSCALE_AUTH_KEY"
    if [ -n "$TAILSCALE_LOGIN_SERVER" ]; then
        TAILSCALE_ARGS="$TAILSCALE_ARGS --login-server=$TAILSCALE_LOGIN_SERVER"
    fi
    if [ -n "$TAILSCALE_HOSTNAME" ]; then
        TAILSCALE_ARGS="$TAILSCALE_ARGS --hostname=$TAILSCALE_HOSTNAME"
    fi
    $TS up $TAILSCALE_ARGS
    echo "Tailscale connected: $($TS ip -4)"
else
    echo "WARNING: No TAILSCALE_AUTH_KEY set, skipping tailscale join"
fi

# Wait for the server process
wait $SERVER_PID
