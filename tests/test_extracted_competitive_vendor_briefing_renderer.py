from __future__ import annotations

import importlib

import pytest


vendor_briefing = importlib.import_module(
    "extracted_competitive_intelligence.templates.email.vendor_briefing"
)


WITNESS_COUNT = 6
CUSTOM_WITNESS_LIMIT = 2


@pytest.fixture(autouse=True)
def _reset_renderer_config():
    vendor_briefing.configure_reasoning_witness_highlight_limit(None)
    yield
    vendor_briefing.configure_reasoning_witness_highlight_limit(None)


def _witness_rows(count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(count):
        rows.append(
            {
                "excerpt_text": f"Verbatim fallback witness {idx}",
                "phrase_verbatim": True,
                "source": "g2",
                "review_id": f"review-{idx}",
            }
        )
    return rows


def test_renderer_default_witness_limit_matches_atlas_contract() -> None:
    selected = vendor_briefing._selected_reasoning_anchors(
        {"reasoning_witness_highlights": _witness_rows(WITNESS_COUNT)}
    )

    assert len(selected) == vendor_briefing.DEFAULT_REASONING_WITNESS_HIGHLIGHT_LIMIT


def test_renderer_witness_limit_is_host_configurable() -> None:
    vendor_briefing.configure_reasoning_witness_highlight_limit(CUSTOM_WITNESS_LIMIT)

    selected = vendor_briefing._selected_reasoning_anchors(
        {"reasoning_witness_highlights": _witness_rows(WITNESS_COUNT)}
    )

    assert len(selected) == CUSTOM_WITNESS_LIMIT


def test_renderer_witness_limit_reset_restores_default() -> None:
    vendor_briefing.configure_reasoning_witness_highlight_limit(CUSTOM_WITNESS_LIMIT)
    vendor_briefing.configure_reasoning_witness_highlight_limit(None)

    selected = vendor_briefing._selected_reasoning_anchors(
        {"reasoning_witness_highlights": _witness_rows(WITNESS_COUNT)}
    )

    assert len(selected) == vendor_briefing.DEFAULT_REASONING_WITNESS_HIGHLIGHT_LIMIT
