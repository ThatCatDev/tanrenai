.PHONY: build install uninstall clean test

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

# ── Clean ────────────────────────────────────────────────────────────────

clean:
	rm -f clients/cli/tanrenai gpu/tanrenai-gpu server/tanrenai-server infra/tanrenai-infra
