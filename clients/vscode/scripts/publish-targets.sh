#!/usr/bin/env bash
#
# Publish the VS Code extension as PER-PLATFORM .vsix packages, each bundling
# the matching tanrenai CLI binary at dist/bin/<target>/tanrenai[.exe] where
# resolveCliPath() looks for it. Without this the published extension ships no
# CLI and falls back to whatever `tanrenai` is on the user's PATH — which can
# be an incompatible older version (e.g. one that still required init.model).
#
# Run from clients/vscode (semantic-release's exec step does this). The version
# in package.json must already be set by the time we publish; semantic-release's
# prepare step handles that.
#
# Env:
#   VSCE_PAT      required to publish (Marketplace token)
#   OVSX_PAT      optional — Open VSX is skipped if unset
#   PACKAGE_ONLY  set to "1" to build per-target .vsix files into dist/ without
#                 publishing (no tokens needed) — used by `make vscode-package-targets`
set -euo pipefail

VSCODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLI_DIR="$(cd "$VSCODE_DIR/../cli" && pwd)"
BIN="$VSCODE_DIR/dist/bin"

# vsce/ovsx target : GOOS : GOARCH : ext   (target name == Node platform-arch,
# which is exactly what resolveCliPath() builds from process.platform/arch).
TARGETS="
linux-x64:linux:amd64:
linux-arm64:linux:arm64:
darwin-x64:darwin:amd64:
darwin-arm64:darwin:arm64:
win32-x64:windows:amd64:.exe
"

# Bundle the JS once (platform-independent); each target packages the same JS
# plus its own binary.
npm run build

for spec in $TARGETS; do
  target="${spec%%:*}"; rest="${spec#*:}"
  goos="${rest%%:*}"; rest="${rest#*:}"
  goarch="${rest%%:*}"; ext="${rest##*:}"

  # Keep only this target's binary in dist/bin so each .vsix stays lean.
  rm -rf "$BIN"
  mkdir -p "$BIN/$target"
  echo "→ building CLI $goos/$goarch → dist/bin/$target/tanrenai$ext"
  (
    cd "$CLI_DIR"
    CGO_ENABLED=0 GOOS="$goos" GOARCH="$goarch" \
      go build -trimpath -ldflags="-s -w" -o "$BIN/$target/tanrenai$ext" .
  )

  if [ "${PACKAGE_ONLY:-}" = "1" ]; then
    echo "→ vsce package --target $target"
    npx vsce package --target "$target" --no-dependencies -o "$VSCODE_DIR/dist/tanrenai-$target.vsix"
    continue
  fi

  echo "→ vsce publish --target $target"
  npx vsce publish --target "$target" --no-dependencies -p "$VSCE_PAT"

  if [ -n "${OVSX_PAT:-}" ]; then
    echo "→ ovsx publish --target $target"
    npx --yes ovsx publish --target "$target" --no-dependencies -p "$OVSX_PAT"
  else
    echo "  OVSX_PAT not set; skipping Open VSX for $target"
  fi
done

rm -rf "$BIN"
if [ "${PACKAGE_ONLY:-}" = "1" ]; then
  echo "Packaged per-target .vsix files into dist/."
else
  echo "Published all targets."
fi
