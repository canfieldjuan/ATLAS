#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
import json
import sys
from pathlib import Path

manifest = Path("extracted_llm_infrastructure/manifest.json")
obj = json.loads(manifest.read_text())
errors: list[str] = []
copied = 0
for mapping in obj["mappings"]:
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

print(f"extracted_llm_infrastructure refreshed from atlas_brain sources ({copied} files)")
PY
