#!/bin/sh
set -e

# Tanrenai installer for Linux and macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/ThatCatDev/tanrenai/main/installers/install.sh | sh

REPO="ThatCatDev/tanrenai"
INSTALL_DIR="${TANRENAI_INSTALL_DIR:-$HOME/.local/bin}"

# Detect OS and architecture
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "$OS" in
  linux)  OS="linux" ;;
  darwin) OS="darwin" ;;
  *)      echo "Error: unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
  x86_64|amd64)   ARCH="amd64" ;;
  aarch64|arm64)   ARCH="arm64" ;;
  *)               echo "Error: unsupported architecture: $ARCH"; exit 1 ;;
esac

echo "Detected: ${OS}/${ARCH}"

# Get latest CLI version.
#
# The repo also publishes the VS Code extension on a SEPARATE tag line
# (`vscode-v*`). GitHub's /releases/latest returns the most recent release of
# ANY kind, so once a `vscode-v*` release is cut it becomes "latest" — and it
# has no `tanrenai-cli-*` assets, which made this installer 404. Instead, list
# releases (newest first) and pick the newest CLI tag (vMAJOR.MINOR...). Pin a
# specific version with TANRENAI_VERSION=v1.2.3 to skip the lookup.
echo "Fetching latest release..."
VERSION="${TANRENAI_VERSION:-$(curl -fsSL "https://api.github.com/repos/${REPO}/releases?per_page=100" \
  | grep '"tag_name"' \
  | sed 's/.*"tag_name": *"//;s/".*//' \
  | grep -E '^v[0-9]+\.[0-9]+' \
  | head -n 1)}"

if [ -z "$VERSION" ]; then
  echo "Error: could not determine latest version"
  exit 1
fi

echo "Latest version: ${VERSION}"

ASSET="tanrenai-cli-${OS}-${ARCH}.tar.gz"
URL="https://github.com/${REPO}/releases/download/${VERSION}/${ASSET}"

# Download and extract
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading ${ASSET}..."
curl -fsSL "$URL" -o "${TMPDIR}/${ASSET}"

echo "Extracting..."
tar xzf "${TMPDIR}/${ASSET}" -C "$TMPDIR"

# Install
mkdir -p "$INSTALL_DIR"

EXTRACTED="tanrenai-cli-${OS}-${ARCH}"

# Copy binary
cp "${TMPDIR}/${EXTRACTED}/tanrenai-cli" "${INSTALL_DIR}/tanrenai"
chmod +x "${INSTALL_DIR}/tanrenai"

# Copy bundled llama-server if present (preserve existing GPU libraries)
if [ -d "${TMPDIR}/${EXTRACTED}/bin" ] && [ "$(ls -A "${TMPDIR}/${EXTRACTED}/bin")" ]; then
  TANRENAI_BIN="${INSTALL_DIR}/../share/tanrenai/bin"
  mkdir -p "$TANRENAI_BIN"
  for f in "${TMPDIR}/${EXTRACTED}/bin/"*; do
    fname="$(basename "$f")"
    # Skip shared libraries that already exist (preserves user-built GPU/CUDA libs)
    case "$fname" in
      *.so|*.so.*|*.dylib)
        if [ -e "${TANRENAI_BIN}/${fname}" ]; then
          continue
        fi
        ;;
    esac
    cp "$f" "${TANRENAI_BIN}/${fname}"
  done
  chmod +x "${TANRENAI_BIN}/"* 2>/dev/null || true
  echo "Installed llama-server to $(cd "${TANRENAI_BIN}" && pwd)"
fi

echo ""
echo "Installed tanrenai ${VERSION} to ${INSTALL_DIR}/tanrenai"

# Check PATH
case ":$PATH:" in
  *":${INSTALL_DIR}:"*)
    echo ""
    echo "tanrenai is ready! Run: tanrenai --local list"
    ;;
  *)
    echo ""
    echo "Add ${INSTALL_DIR} to your PATH:"
    SHELL_NAME=$(basename "${SHELL:-/bin/sh}")
    case "$SHELL_NAME" in
      zsh)  echo "  echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.zshrc && source ~/.zshrc" ;;
      fish) echo "  fish_add_path ${INSTALL_DIR}" ;;
      *)    echo "  echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.bashrc && source ~/.bashrc" ;;
    esac
    echo ""
    echo "Then run: tanrenai --local list"
    ;;
esac
