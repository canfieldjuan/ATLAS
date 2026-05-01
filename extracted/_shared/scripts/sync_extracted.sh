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
errors: list[str] = []
copied = 0

for mapping in obj["mappings"]:
    if mapping["target"] in owned:
        continue
    src = Path(mapping["source"])
    dst = Path(mapping["target"])
    if not src.exists():
        errors.append(
            f"missing source: {src} (target would be {dst}); "
            "fix the manifest entry before re-running sync"
        )
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    copied += 1

if errors:
    for line in errors:
        print(f"ERROR {line}")
    print(f"Sync failed: {len(errors)} missing source(s); {copied} target(s) refreshed")
    sys.exit(1)

print(f"{product} refreshed from atlas_brain sources ({copied} files)")
PY
