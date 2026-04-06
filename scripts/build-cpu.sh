#!/bin/bash
set -e
echo "Building EIE with CPU backend..."
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
echo "Build complete: ./build/eie-server"
