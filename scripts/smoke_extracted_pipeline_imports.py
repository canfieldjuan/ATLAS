#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = [
    "extracted_content_pipeline.autonomous.tasks.blog_post_generation",
    "extracted_content_pipeline.autonomous.tasks.b2b_blog_post_generation",
    "extracted_content_pipeline.autonomous.tasks.complaint_content_generation",
    "extracted_content_pipeline.autonomous.tasks.complaint_enrichment",
    "extracted_content_pipeline.autonomous.tasks.article_enrichment",
    "extracted_content_pipeline.autonomous.tasks.b2b_vendor_briefing",
    "extracted_content_pipeline.autonomous.tasks.b2b_campaign_generation",
    # Podcast repurposing surface (single-pass, host-installable).
    "extracted_content_pipeline.podcast_ports",
    "extracted_content_pipeline.podcast_transcript_data",
    "extracted_content_pipeline.podcast_postgres_import",
    "extracted_content_pipeline.podcast_extraction",
    "extracted_content_pipeline.podcast_postgres",
    "extracted_content_pipeline.podcast_postgres_extraction",
    "extracted_content_pipeline.podcast_idea_data",
    "extracted_content_pipeline.podcast_repurpose_generation",
    "extracted_content_pipeline.podcast_postgres_repurpose",
    "extracted_content_pipeline.podcast_example",
    "extracted_content_pipeline.services.podcast_quality",
    "extracted_content_pipeline.api.campaign_webhooks",
    "extracted_content_pipeline.api.b2b_campaigns",
    "extracted_content_pipeline.api.seller_campaigns",
    "extracted_content_pipeline.campaign_postgres_seller_opportunities",
]


def main() -> int:
    failed: list[str] = []
    for module in MODULES:
        try:
            importlib.import_module(module)
            print(f"OK {module}")
        except Exception as exc:
            print(f"FAIL {module}: {exc}")
            failed.append(module)

    if failed:
        print(f"Import smoke failed for {len(failed)} module(s)")
        return 1

    print("Import smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
