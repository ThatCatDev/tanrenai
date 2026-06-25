.PHONY: build install uninstall clean test vscode-bundle vscode-package vscode-package-targets

PREFIX ?= $(HOME)/.local

# ── Build ────────────────────────────────────────────────────────────────

build:
	cd clients/cli && go build -o tanrenai .

build-gpu:
	cd gpu && go build -o tanrenai-gpu .

build-server:
	cd server && go build -o tanrenai-server .

build-infra:
	cd infra && go build -o tanrenai-infra .

build-all: build build-gpu build-server build-infra

# ── Install ──────────────────────────────────────────────────────────────

install: build
	@mkdir -p $(PREFIX)/bin
	cp clients/cli/tanrenai $(PREFIX)/bin/tanrenai
	@echo "Installed to $(PREFIX)/bin/tanrenai"
	@echo ""
	@if echo "$$PATH" | tr ':' '\n' | grep -qx "$(PREFIX)/bin"; then \
		echo "tanrenai is ready — run: tanrenai run <model>"; \
	else \
		echo "Add $(PREFIX)/bin to your PATH:"; \
		echo "  echo 'export PATH=\"$(PREFIX)/bin:\$$PATH\"' >> ~/.zshrc"; \
		echo "  source ~/.zshrc"; \
	fi

uninstall:
	rm -f $(PREFIX)/bin/tanrenai
	@echo "Removed $(PREFIX)/bin/tanrenai"

# ── Test ─────────────────────────────────────────────────────────────────

test:
	cd clients/shared && go test ./...
	cd clients/cli && go test ./...
	cd gpu && go test ./...
	cd server && go test ./...
	cd infra && go test ./...

# ── VS Code extension bundling ────────────────────────────────────────────
#
# Cross-compile the CLI for every platform the VS Code extension needs to
# support, dropping the binaries into clients/vscode/dist/bin/<plat>-<arch>/
# where the extension's resolveCliPath() expects to find them.
#
# Directory names match Node's `process.platform`/`process.arch`:
#   linux/amd64    → dist/bin/linux-x64/tanrenai
#   linux/arm64    → dist/bin/linux-arm64/tanrenai
#   darwin/amd64   → dist/bin/darwin-x64/tanrenai
#   darwin/arm64   → dist/bin/darwin-arm64/tanrenai
#   windows/amd64  → dist/bin/win32-x64/tanrenai.exe

VSCODE_BIN := clients/vscode/dist/bin

# Each target is "node-platform/node-arch=go-os/go-arch[/extension]".
VSCODE_TARGETS := \
	linux-x64=linux/amd64 \
	linux-arm64=linux/arm64 \
	darwin-x64=darwin/amd64 \
	darwin-arm64=darwin/arm64 \
	win32-x64=windows/amd64/.exe

vscode-bundle:
	@for spec in $(VSCODE_TARGETS); do \
		dir=$${spec%%=*}; \
		rest=$${spec#*=}; \
		goos=$${rest%%/*}; rest2=$${rest#*/}; \
		goarch=$${rest2%%/*}; \
		ext=""; case "$$rest2" in */.exe) ext=".exe";; esac; \
		out=$(VSCODE_BIN)/$$dir; \
		mkdir -p "$$out"; \
		echo "  → $$goos/$$goarch  $$out/tanrenai$$ext"; \
		(cd clients/cli && CGO_ENABLED=0 GOOS=$$goos GOARCH=$$goarch \
			go build -trimpath -ldflags="-s -w" -o ../../$$out/tanrenai$$ext .) \
			|| exit 1; \
	done
	@echo "Bundled CLI binaries into $(VSCODE_BIN)/"

# Build a single universal .vsix that includes ALL bundled CLI binaries.
# Handy for local sideloading; the Marketplace release uses per-target packages
# (see vscode-package-targets) so each download only carries its own binary.
vscode-package: vscode-bundle
	cd clients/vscode && npm install --no-audit --no-fund && npm run build && npx vsce package

# Build per-platform .vsix files (one binary each) into clients/vscode/dist/ —
# the same path CI publishes. No Marketplace tokens needed (PACKAGE_ONLY).
vscode-package-targets:
	cd clients/vscode && npm install --no-audit --no-fund && PACKAGE_ONLY=1 bash scripts/publish-targets.sh

# ── Clean ────────────────────────────────────────────────────────────────

clean:
	rm -f clients/cli/tanrenai gpu/tanrenai-gpu server/tanrenai-server infra/tanrenai-infra
	rm -rf $(VSCODE_BIN)
