#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

status=0
while IFS= read -r file; do
  if ! python - "$file" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = path.read_bytes()
violations = [(idx + 1, b) for idx, b in enumerate(data) if b > 0x7F]
if violations:
    print(path)
    for idx, b in violations[:20]:
        print(f"  byte_offset={idx} value=0x{b:02X}")
    sys.exit(1)
PY
  then
    status=1
  fi
done < <(printf '%s\n' \
  extracted_content_pipeline/autonomous/tasks/blog_post_generation.py \
  extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py \
  extracted_content_pipeline/autonomous/tasks/complaint_content_generation.py \
  extracted_content_pipeline/autonomous/tasks/complaint_enrichment.py \
  extracted_content_pipeline/autonomous/tasks/article_enrichment.py)

if [[ "$status" -ne 0 ]]; then
  echo "ASCII check failed"
  exit 1
fi

echo "ASCII check passed for extracted_content_pipeline Python files"
