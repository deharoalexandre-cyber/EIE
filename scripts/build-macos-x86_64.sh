#!/bin/bash
# EIE: build macOS Intel (CPU + Accelerate, Metal désactivé : iGPU Intel = résultats faux),
# binaire statique et portable (GGML_NATIVE=OFF : pas de -march=native).
set -e
cd "$(dirname "$0")/.."
cmake -B build-x86_64 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_METAL=OFF -DGGML_ACCELERATE=ON -DGGML_BLAS=OFF \
  -DLLAMA_OPENSSL=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build-x86_64 --target eie-server -j "$(sysctl -n hw.ncpu)"
echo "Binaire : build-x86_64/eie-server"
