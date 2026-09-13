#!/bin/bash
# EIE — installation d'un bundle macOS (Intel ou Apple Silicon).
# Usage : bash install-macos.sh [--download-models] [--dest DIR]
#   - copie eie-server et les presets dans DIR (défaut ~/Elyne)
#   - télécharge les modèles de référence sur demande (--download-models, ~3,7 Go)
#   - installe un LaunchAgent (moteur démarré avec la session, port 8090)
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/Elyne"
DOWNLOAD=0
for a in "$@"; do
  case "$a" in
    --download-models) DOWNLOAD=1 ;;
    --dest=*) DEST="${a#--dest=}" ;;
    --dest) shift; DEST="$1" ;;
  esac
done
ARCH="$(uname -m)"
case "$ARCH" in
  arm64)  PRESET="macos-silicon.yaml" ;;
  x86_64) PRESET="macos-cpu.yaml" ;;
  *) echo "Architecture non prise en charge : $ARCH"; exit 1 ;;
esac
[ -x "$SRC/eie-server" ] || { echo "eie-server introuvable à côté de ce script."; exit 1; }
BIN_ARCH="$(file "$SRC/eie-server" | grep -oE 'x86_64|arm64' | head -1)"
[ "$BIN_ARCH" = "$ARCH" ] || { echo "Ce bundle est pour $BIN_ARCH, cette machine est $ARCH."; exit 1; }

echo "== EIE macOS ($ARCH) → $DEST"
mkdir -p "$DEST/models" "$DEST/presets" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
cp "$SRC/eie-server" "$DEST/eie-server" && chmod +x "$DEST/eie-server"
cp "$SRC"/presets/*.yaml "$DEST/presets/"
xattr -cr "$DEST" 2>/dev/null || true

# Modèles de référence (mêmes fichiers que les reçus macOS du dépôt)
GEMMA_URL="https://huggingface.co/lmstudio-community/gemma-4-E2B-it-QAT-GGUF/resolve/main/gemma-4-E2B-it-QAT-Q4_0.gguf"
BGE_URL="https://huggingface.co/gpustack/bge-m3-GGUF/resolve/main/bge-m3-Q8_0.gguf"
GEMMA_SHA="aa6eb6d481b583a304269649d467fbff66267f7d9c480b18911bdcf081102790"
BGE_SHA="950f4a8e5e19477a6d3c26d2f162233c20002c601f75e4b002e3239997821167"
fetch() { # url dest sha
  if [ -f "$2" ]; then echo "présent : $(basename "$2")"; return; fi
  echo "téléchargement : $(basename "$2")"
  curl -L --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
  local got; got="$(shasum -a 256 "$2" | cut -d' ' -f1)"
  [ "$got" = "$3" ] || { echo "SHA-256 inattendu pour $(basename "$2") : $got"; exit 1; }
  echo "SHA-256 vérifié."
}
if [ "$DOWNLOAD" = 1 ]; then
  fetch "$GEMMA_URL" "$DEST/models/gemma-4-E2B-it-QAT-Q4_0.gguf" "$GEMMA_SHA"
  fetch "$BGE_URL"   "$DEST/models/bge-m3-Q8_0.gguf" "$BGE_SHA"
elif [ -z "$(ls "$DEST/models"/*.gguf 2>/dev/null)" ]; then
  echo "Aucun modèle dans $DEST/models. Relance avec --download-models, ou dépose-y tes GGUF."
fi

PLIST="$HOME/Library/LaunchAgents/com.elyne.eie.plist"
launchctl bootout "gui/$(id -u)/com.elyne.eie" 2>/dev/null || true
cat > "$PLIST" << PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.elyne.eie</string>
    <key>ProgramArguments</key>
    <array>
        <string>$DEST/eie-server</string>
        <string>--config</string>
        <string>$DEST/presets/$PRESET</string>
    </array>
    <key>WorkingDirectory</key><string>$DEST</string>
    <key>ProcessType</key><string>Interactive</string>
    <key>Nice</key><integer>-10</integer>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>$HOME/Library/Logs/eie.log</string>
    <key>StandardErrorPath</key><string>$HOME/Library/Logs/eie.log</string>
</dict>
</plist>
PL
launchctl bootstrap "gui/$(id -u)" "$PLIST"
printf "Démarrage du moteur"
for i in $(seq 1 90); do
  if curl -s -m 2 http://127.0.0.1:8090/health > /dev/null 2>&1; then
    echo " — prêt : $(curl -s -m 2 http://127.0.0.1:8090/health)"; exit 0
  fi
  printf "."; sleep 2
done
echo; echo "Le moteur ne répond pas : tail -30 ~/Library/Logs/eie.log"; exit 1
