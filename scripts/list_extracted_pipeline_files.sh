#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/extracted_content_pipeline"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Missing directory: $TARGET_DIR"
  exit 1
fi

echo "Scaffold root: $TARGET_DIR"
rg --files "$TARGET_DIR"
