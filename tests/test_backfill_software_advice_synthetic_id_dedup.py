from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "backfill_software_advice_synthetic_id_dedup.py"
    spec = importlib.util.spec_from_file_location("backfill_software_advice_synthetic_id_dedup", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_plan_updates_targets_only_synthetic_low_value_duplicates():
    module = _load_module()

    planned = module._plan_updates([
        {
            "id": "stable-1",
            "vendor_name": "ClickUp",
            "source_review_id": "Capterra___7035855",
            "enrichment_status": "enriched",
            "review_text": "Good value for the money and easy to get started.",
            "summary": "Good value for the money",
            "pros": None,
            "cons": None,
            "reviewer_title": None,
            "reviewer_company": None,
            "reviewer_company_norm": None,
            "company_size_raw": None,
            "reviewer_industry": None,
            "parser_version": "software_advice:2",
            "raw_metadata": {},
            "review_content_hash": "hash-1",
        },
        {
            "id": "synthetic-1",
            "vendor_name": "ClickUp",
            "source_review_id": "49dc014ba1f66cc9",
            "enrichment_status": "raw_only",
            "review_text": "Good value for the money and easy to get started.",
            "summary": None,
            "pros": None,
            "cons": None,
            "reviewer_title": None,
            "reviewer_company": None,
            "reviewer_company_norm": None,
            "company_size_raw": "51-200",
            "reviewer_industry": "Consumer Goods",
            "parser_version": "software_advice:3",
            "raw_metadata": {},
            "review_content_hash": "hash-1",
        },
        {
            "id": "synthetic-2",
            "vendor_name": "ClickUp",
            "source_review_id": "2324651b3db6979a",
            "enrichment_status": "enriched",
            "review_text": "Good value for the money and easy to get started.",
            "summary": None,
            "pros": None,
            "cons": None,
            "reviewer_title": None,
            "reviewer_company": None,
            "reviewer_company_norm": None,
            "company_size_raw": "11-50",
            "reviewer_industry": "IT",
            "parser_version": "software_advice:3",
            "raw_metadata": {},
            "review_content_hash": "hash-1",
        },
    ])

    assert len(planned) == 1
    assert planned[0]["duplicate_review_id"] == "synthetic-1"
    assert planned[0]["canonical_review_id"] == "stable-1"
    assert planned[0]["canonical_source_review_id"] == "Capterra___7035855"


def test_plan_updates_skips_groups_without_stable_ids():
    module = _load_module()

    planned = module._plan_updates([
        {
            "id": "synthetic-1",
            "vendor_name": "Zendesk",
            "source_review_id": "e9f46737ecf2953a",
            "enrichment_status": "raw_only",
            "review_text": "Works well for simple ticket flows.",
            "summary": None,
            "pros": None,
            "cons": None,
            "reviewer_title": None,
            "reviewer_company": None,
            "reviewer_company_norm": None,
            "company_size_raw": None,
            "reviewer_industry": None,
            "parser_version": "software_advice:3",
            "raw_metadata": {},
            "review_content_hash": "hash-2",
        }
    ])

    assert planned == []
