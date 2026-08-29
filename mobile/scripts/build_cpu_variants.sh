#!/usr/bin/env bash
# Build ggml/llama/mtmd pour Android arm64 avec dispatch runtime des variantes CPU
# (GGML_CPU_ALL_VARIANTS + BACKEND_DL) -> SD888 (dotprod) ET SD 8Gen3+ (i8mm).
# mtmd rebuild dans ce schema (ne linke plus libggml-cpu.so).
cd /workspace || exit 2
cp -f docs/backend/snapdragon/CMakeUserPresets.json .
mkdir -p /workspace/_bl
FLAGS="-fvectorize -ffp-model=fast -fno-finite-math-only -D_GNU_SOURCE"

cmake --preset arm64-android-snapdragon-release -B /tmp/build-av \
  -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON -DGGML_HEXAGON=OFF -DGGML_OPENCL=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_HTML=OFF -DLLAMA_BUILD_COMMON=ON \
  -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$FLAGS" > /workspace/_bl/cfg.log 2>&1
echo "configure exit $?"

cmake --build /tmp/build-av -j --target llama mtmd llama-common ggml ggml-base ggml-opencl \
  ggml-cpu-android_armv8.0_1 ggml-cpu-android_armv8.2_1 ggml-cpu-android_armv8.2_2 ggml-cpu-android_armv8.6_1 \
  ggml-cpu-android_armv9.0_1 ggml-cpu-android_armv9.2_1 ggml-cpu-android_armv9.2_2 > /workspace/_bl/build.log 2>&1
BR=$?
echo "build exit $BR"
tail -30 /workspace/_bl/build.log

if [ "$BR" -eq 0 ]; then
  rm -rf /workspace/pkg-av2 && mkdir -p /workspace/pkg-av2
  cp /tmp/build-av/bin/*.so /workspace/pkg-av2/ 2>/dev/null
  echo "=== .so recuperes ==="
  ls -la /workspace/pkg-av2 | grep -i '\.so'
fi
echo "=== SCRIPT DONE ==="
