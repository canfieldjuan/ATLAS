#!/usr/bin/env python3
"""CLI wrapper for support-ticket generated-content evaluation."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extracted_content_pipeline.support_ticket_generated_content_eval import (
    evaluate_support_ticket_generated_content,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
