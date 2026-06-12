"""Token-set cluster-preview skip threshold (#1454).

Token-set clustering is pairwise (quadratic in rows), so above
MAX_TOKEN_SET_CLUSTER_ROWS the preview is skipped and reported instead of
wedging the worker. Explicit-category rows always cluster.
"""

from __future__ import annotations

import time

import pytest

from extracted_content_pipeline import support_ticket_clustering as clustering
from extracted_content_pipeline.support_ticket_clustering import (
    MAX_TOKEN_SET_CLUSTER_ROWS,
    assign_support_ticket_clusters_with_diagnostics,
    support_ticket_cluster_quality,
)
from extracted_content_pipeline.support_ticket_input_package import (
    build_support_ticket_input_package,
)


def _token_set_row(index: int) -> dict[str, str]:
    return {
        "ticket_id": f"ticket-{index}",
        "description": (
            f"Widget batch {index} misroutes parcel labels during nightly "
            f"depot transfer window {index % 7}"
        ),
    }


def test_token_set_rows_below_threshold_cluster_normally() -> None:
    rows = [
        {"ticket_id": "ticket-1", "description": "How do I export data?"},
        {"ticket_id": "ticket-2", "description": "Exporting data fails for me"},
    ]

    annotated, diagnostics = assign_support_ticket_clusters_with_diagnostics(rows)

    assert diagnostics == {
        "token_set_row_count": 2,
        "max_token_set_rows": MAX_TOKEN_SET_CLUSTER_ROWS,
        "cluster_preview_skipped": False,
    }
    assert all(row["support_ticket_cluster"] for row in annotated)


def test_token_set_rows_above_threshold_skip_preview_but_keep_rows() -> None:
    rows = [_token_set_row(index) for index in range(4)]
    rows.append({
        "ticket_id": "ticket-explicit",
        "description": "Billing question",
        "pain_category": "billing",
    })

    annotated, diagnostics = assign_support_ticket_clusters_with_diagnostics(
        rows,
        max_token_set_rows=3,
    )

    assert diagnostics == {
        "token_set_row_count": 4,
        "max_token_set_rows": 3,
        "cluster_preview_skipped": True,
    }
    assert len(annotated) == 5
    token_set_rows = annotated[:4]
    assert all("support_ticket_cluster" not in row for row in token_set_rows)
    explicit_row = annotated[4]
    assert explicit_row["support_ticket_cluster"] == "billing"
    assert explicit_row["support_ticket_cluster_source"] == "explicit"
    quality = support_ticket_cluster_quality(annotated)
    assert quality["uncategorized_row_count"] == 4
    assert quality["clustered_row_count"] == 1


def test_package_surfaces_cluster_preview_skip_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clustering, "MAX_TOKEN_SET_CLUSTER_ROWS", 3)
    rows = [_token_set_row(index) for index in range(5)]

    package = build_support_ticket_input_package(rows)

    assert package.metadata["cluster_preview_skipped"] is True
    assert package.metadata["cluster_preview_token_set_row_count"] == 5
    assert package.metadata["cluster_quality"]["uncategorized_row_count"] == 5
    skip_warnings = [
        warning
        for warning in package.warnings
        if warning["code"] == "cluster_preview_skipped_large_upload"
    ]
    assert len(skip_warnings) == 1
    assert skip_warnings[0]["row_count"] == 5
    assert skip_warnings[0]["max_token_set_rows"] == 3
    assert package.metadata["included_row_count"] == 5


def test_package_omits_skip_metadata_below_threshold() -> None:
    rows = [_token_set_row(index) for index in range(3)]

    package = build_support_ticket_input_package(rows)

    assert "cluster_preview_skipped" not in package.metadata
    assert all(
        warning["code"] != "cluster_preview_skipped_large_upload"
        for warning in package.warnings
    )


def test_cluster_preview_skip_bounds_runtime_at_25k_rows() -> None:
    rows = [_token_set_row(index) for index in range(25_000)]

    started = time.perf_counter()
    annotated, diagnostics = assign_support_ticket_clusters_with_diagnostics(rows)
    elapsed = time.perf_counter() - started

    assert diagnostics["cluster_preview_skipped"] is True
    assert diagnostics["token_set_row_count"] == 25_000
    assert len(annotated) == 25_000
    # The quadratic pairwise path measured ~6.7s at just 2k rows (#1454);
    # with the preview skipped this is a linear tokenize pass. The generous
    # bound discriminates against any quadratic regression without being
    # flaky on slow CI runners.
    assert elapsed < 30.0
