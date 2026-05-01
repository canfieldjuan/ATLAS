#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Collect every .py file under the scaffold dynamically. The list comes
# from the manifest's Python targets so a future addition gets covered
# without touching this script.
mapfile -t files < <(python - <<'PY'
import json
from pathlib import Path
manifest = Path("extracted_llm_infrastructure/manifest.json")
obj = json.loads(manifest.read_text())
for mapping in obj["mappings"]:
    target = mapping["target"]
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
# Report 0-based byte offsets so editors / hex tools (which seek to a
# zero-based index) point to the right byte without an off-by-one.
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

echo "ASCII check passed for extracted_llm_infrastructure Python files"
