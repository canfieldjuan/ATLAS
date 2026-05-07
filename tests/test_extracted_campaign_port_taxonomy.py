from __future__ import annotations

from typing import get_args

from extracted_content_pipeline.campaign_ports import TargetMode


def test_target_mode_alias_covers_known_runtime_and_asset_modes() -> None:
    assert {
        "vendor_retention",
        "challenger_intel",
        "churning_company",
        "amazon_seller",
        "vendor",
        "account",
        "opportunity",
        "marketing_campaign",
    } <= set(get_args(TargetMode))
