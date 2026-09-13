#!/bin/bash
# EIE: assemble un bundle macOS distribuable depuis un répertoire de build.
# Usage : bash scripts/bundle-macos.sh arm64|x86_64 [build-dir]
set -e
cd "$(dirname "$0")/.."
ARCH="${1:?arm64|x86_64}"
BUILD="${2:-build-$ARCH}"
[ -x "$BUILD/eie-server" ] || { echo "$BUILD/eie-server manquant (voir scripts/build-macos-arm64.sh)"; exit 1; }
REV="$(git rev-parse --short HEAD)"
NAME="eie-macos-$ARCH-$REV"
OUT="dist/$NAME"
rm -rf "$OUT" && mkdir -p "$OUT/presets"
cp "$BUILD/eie-server" "$OUT/"
cp presets/macos-cpu.yaml presets/macos-silicon.yaml "$OUT/presets/"
cp scripts/install-macos.sh "$OUT/install-macos.sh"
cp docs/macos.md "$OUT/README.md"
cp LICENSE NOTICE "$OUT/" 2>/dev/null || true
(cd "$OUT" && shasum -a 256 eie-server presets/*.yaml install-macos.sh > SHA256SUMS)
(cd dist && tar -czf "$NAME.tar.gz" "$NAME")
shasum -a 256 "dist/$NAME.tar.gz"
