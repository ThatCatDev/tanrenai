# ── Stage 1: Build llama-server with CUDA ──────────────────────────────
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS llama-build

ARG LLAMA_CPP_REF=master
# Common CUDA architectures:
#   89  = RTX 4090 / RTX PRO 6000
#   90  = H100 / A100
#   100 = Blackwell B200
#   120 = Blackwell Ultra
ARG CUDA_ARCHITECTURES="89;90;100;120"

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git build-essential curl && \
    curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz \
      | tar -xzf - -C /usr/local --strip-components=1

RUN git clone --depth 1 --branch "$LLAMA_CPP_REF" https://github.com/ggml-org/llama.cpp /opt/llama.cpp

WORKDIR /opt/llama.cpp
RUN cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j"$(nproc)" -t llama-server

# ── Stage 2: Build tanrenai-gpu ────────────────────────────────────────
FROM golang:1.25-bookworm AS go-build

WORKDIR /src
COPY gpu/ gpu/
RUN cd gpu && CGO_ENABLED=0 go build -o /tanrenai-gpu .

# ── Stage 3: Runtime image ─────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ca-certificates curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# llama-server binary and shared libs
COPY --from=llama-build /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama-build /opt/llama.cpp/build/bin/*.so* /usr/local/lib/
RUN ldconfig

# tanrenai-gpu binary
COPY --from=go-build /tanrenai-gpu /usr/local/bin/tanrenai-gpu

# Set up tanrenai data dirs
RUN mkdir -p /root/.local/share/tanrenai/bin && \
    ln -s /usr/local/bin/llama-server /root/.local/share/tanrenai/bin/llama-server

ENV LD_LIBRARY_PATH=/usr/local/lib
EXPOSE 11435

ENTRYPOINT ["tanrenai-gpu"]
CMD ["serve", "--host", "0.0.0.0", "--port", "11435"]
