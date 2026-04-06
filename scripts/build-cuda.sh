#!/bin/bash
set -e
echo "Building EIE with CUDA backend..."
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
echo "Build complete: ./build/eie-server"
