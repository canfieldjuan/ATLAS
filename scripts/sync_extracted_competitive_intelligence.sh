#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
import json
import sys
from pathlib import Path

manifest = Path("extracted_competitive_intelligence/manifest.json")
obj = json.loads(manifest.read_text())
owned = {entry["target"] for entry in obj.get("owned", [])}
errors: list[str] = []
copied = 0
for mapping in obj["mappings"]:
    src = Path(mapping["source"])
    dst = Path(mapping["target"])
    if mapping["target"] in owned:
        continue
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

print(f"extracted_competitive_intelligence refreshed from atlas_brain sources ({copied} files)")
PY
