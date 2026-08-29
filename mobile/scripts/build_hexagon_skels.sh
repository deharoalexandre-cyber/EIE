#!/usr/bin/env bash
# Build du backend Hexagon (ARM) + skels DSP (libggml-htp-vXX.so) pour la Piste B NPU.
# Cibles device : Z Flip 6 (8 Gen 3 = HTP v75), S25 Ultra (8 Elite = v79).
cd /workspace || exit 2
cp -f docs/backend/snapdragon/CMakeUserPresets.json .
mkdir -p /workspace/_bl
FLAGS="-fvectorize -ffp-model=fast -fno-finite-math-only -D_GNU_SOURCE"

cmake --preset arm64-android-snapdragon-release -B /tmp/build-hex \
  -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON -DGGML_HEXAGON=ON -DGGML_OPENCL=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_HTML=OFF \
  -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$FLAGS" > /workspace/_bl/cfg-hex.log 2>&1
echo "configure exit $?"
grep -iE "hexagon|htp" /workspace/_bl/cfg-hex.log | tail -8

cmake --build /tmp/build-hex -j --target ggml-hexagon htp-v68 htp-v75 htp-v79 > /workspace/_bl/build-hex.log 2>&1
BR=$?
echo "build exit $BR"
tail -25 /workspace/_bl/build-hex.log

if [ "$BR" -eq 0 ]; then
  rm -rf /workspace/pkg-hex && mkdir -p /workspace/pkg-hex
  find /tmp/build-hex \( -name "libggml-hexagon.so" -o -name "libggml-htp-*.so" \) -exec cp -v {} /workspace/pkg-hex/ \; 2>/dev/null
  # les skels atterrissent dans le binary dir de ggml-hexagon (BUILD_BYPRODUCTS)
  cp -v /tmp/build-hex/ggml/src/ggml-hexagon/libggml-htp-*.so /workspace/pkg-hex/ 2>/dev/null
  echo "=== livrables ==="
  ls -la /workspace/pkg-hex
fi
echo "=== SCRIPT DONE ==="
