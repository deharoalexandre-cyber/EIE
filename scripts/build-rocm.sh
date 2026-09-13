#!/bin/bash
set -e
echo "Building EIE with ROCm/HIP backend..."
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
echo "Build complete: ./build/eie-server"
