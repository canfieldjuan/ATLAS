#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <product_dir>" >&2
  exit 2
fi

PRODUCT_DIR="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

python - "$PRODUCT_DIR" <<'PY'
import json
import sys
from pathlib import Path

product = Path(sys.argv[1])
manifest = product / "manifest.json"
if not manifest.exists():
    print(f"ERROR missing manifest: {manifest}", file=sys.stderr)
    sys.exit(2)

obj = json.loads(manifest.read_text())
owned = {entry["target"] for entry in obj.get("owned", [])}
status = 0

for mapping in obj["mappings"]:
    if mapping["target"] in owned:
        continue
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
    print(f"Validation failed: run extracted/_shared/scripts/sync_extracted.sh {product}")
    sys.exit(1)

print(f"Validation passed: mapped files for {product} match atlas_brain sources")
PY

python "$ROOT_DIR/extracted/_shared/scripts/forbid_hard_atlas_imports.py" "$PRODUCT_DIR"
