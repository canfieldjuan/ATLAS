#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
import json
from pathlib import Path
import sys

manifest = Path("extracted_llm_infrastructure/manifest.json")
obj = json.loads(manifest.read_text())
status = 0
for mapping in obj["mappings"]:
    src = Path(mapping["source"])
    dst = Path(mapping["target"])
    if not src.exists():
        print(f"MISSING SOURCE: {src} (referenced by manifest target {dst})")
        status = 1
        continue
    if not dst.exists() or src.read_bytes() != dst.read_bytes():
        print(f"OUT OF SYNC: {dst}")
        status = 1

if status:
    print("Validation failed: run scripts/sync_extracted_llm_infrastructure.sh")
    sys.exit(1)

print("Validation passed: extracted_llm_infrastructure matches atlas_brain sources")
PY
