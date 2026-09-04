#!/bin/bash
# EIE — build macOS Apple Silicon (Metal), binaire statique et portable.
# Fonctionne aussi en cross-compilation depuis un Mac Intel.
set -e
cd "$(dirname "$0")/.."
cmake -B build-arm64 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DGGML_NATIVE=OFF \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=OFF \
  -DLLAMA_OPENSSL=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build-arm64 --target eie-server -j "$(sysctl -n hw.ncpu)"
echo "Binaire : build-arm64/eie-server"
