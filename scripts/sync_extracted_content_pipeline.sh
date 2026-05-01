#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
import json
from pathlib import Path

manifest = Path("extracted_content_pipeline/manifest.json")
obj = json.loads(manifest.read_text())
for mapping in obj["mappings"]:
    src = Path(mapping["source"])
    dst = Path(mapping["target"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())

print("extracted_content_pipeline refreshed from atlas_brain sources")
PY
