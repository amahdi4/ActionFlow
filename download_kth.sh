#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$ROOT_DIR/data/kth/raw"
BASE_URL="https://www.csc.kth.se/cvap/actions"
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

echo "Download complete."
