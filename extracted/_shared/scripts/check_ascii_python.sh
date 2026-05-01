#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <product_dir>" >&2
  exit 2
fi

PRODUCT_DIR="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

mapfile -t files < <(python - "$PRODUCT_DIR" <<'PY'
import json
import sys
from pathlib import Path

product = Path(sys.argv[1])
obj = json.loads((product / "manifest.json").read_text())
for mapping in obj["mappings"]:
    target = mapping["target"]
    if target.endswith(".py"):
        print(target)
for entry in obj.get("owned", []):
    target = entry["target"]
    if target.endswith(".py"):
        print(target)
PY
)

status=0
for file in "${files[@]}"; do
  if ! python - "$file" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = path.read_bytes()
violations = [(idx, b) for idx, b in enumerate(data) if b > 0x7F]
if violations:
    print(path)
    for idx, b in violations[:20]:
        print(f"  byte_offset={idx} value=0x{b:02X}")
    sys.exit(1)
PY
  then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "ASCII check failed"
  exit 1
fi

echo "ASCII check passed for $PRODUCT_DIR Python files"
