#!/bin/bash
set -e
echo "Building EIE with CPU backend..."
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$JOBS"
echo "Build complete: ./build/eie-server"
