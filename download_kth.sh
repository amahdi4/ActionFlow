#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
KTH_DIR="$ROOT_DIR/data/kth"
DATA_DIR="$KTH_DIR/raw"
BASE_URL="https://www.csc.kth.se/cvap/actions"
SEQUENCES_FILE="$KTH_DIR/00sequences.txt"
ACTIONS=("boxing" "handclapping" "handwaving" "jogging" "running" "walking")

mkdir -p "$DATA_DIR"

for action in "${ACTIONS[@]}"; do
    AVI_DIR="$DATA_DIR/$action"
    ZIP_FILE="$DATA_DIR/${action}.zip"

    if [ -d "$AVI_DIR" ] && find "$AVI_DIR" -maxdepth 1 -name "*.avi" | grep -q .; then
        echo "[SKIP] $action already downloaded"
        continue
    fi

    mkdir -p "$AVI_DIR"
    echo "[DOWNLOADING] $action.zip"
    curl -fL --progress-bar -o "$ZIP_FILE" "$BASE_URL/${action}.zip"
    echo "[EXTRACTING] $action.zip"
    unzip -q -o "$ZIP_FILE" -d "$AVI_DIR"
done

if [ -f "$SEQUENCES_FILE" ]; then
    echo "[SKIP] 00sequences.txt already downloaded"
else
    echo "[DOWNLOADING] 00sequences.txt"
    curl -fL --progress-bar -o "$SEQUENCES_FILE" "$BASE_URL/00sequences.txt"
fi

echo "Download complete."
