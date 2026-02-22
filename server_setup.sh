#!/usr/bin/env bash
set -e

# Install dependencies
apt update && apt install -y git cmake build-essential libcurl4-openssl-dev

# Install Go
wget https://go.dev/dl/go1.24.4.linux-amd64.tar.gz
tar -C /usr/local -xzf go1.24.4.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc

# Build llama-server with CUDA
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --target llama-server -j$(nproc)
mkdir -p ~/.local/share/tanrenai/bin
cp build/bin/llama-server ~/.local/share/tanrenai/bin/
cd ~

# Clone and build tanrenai
git clone https://github.com/ThatCatDev/tanrenai.git
cd tanrenai/server
go build -o tanrenai .

# Download models
# Llama 3.3 70B — dense 70B, fits in A100 80GB (~42GB)
./tanrenai pull https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf

# Embedding model for memory/RAG
./tanrenai pull https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF/resolve/main/all-MiniLM-L6-v2.Q8_0.gguf

echo ""
echo "Setup complete. Run with:"
echo "  cd ~/tanrenai/server"
echo "  ./tanrenai run Llama-3.3-70B-Instruct-Q4_K_M --agent --memory --ctx-size 16384 --gpu-layers -1"
