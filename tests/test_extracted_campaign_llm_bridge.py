from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_content_llm_bridges_target_extracted_llm_infrastructure() -> None:
    bridge_paths = [
        "extracted_content_pipeline/pipelines/llm.py",
        "extracted_content_pipeline/services/b2b/anthropic_batch.py",
        "extracted_content_pipeline/services/llm/anthropic.py",
    ]

    for path in bridge_paths:
        text = _read(path)
        assert "extracted_llm_infrastructure" in text
        assert "from atlas_brain" not in text


def test_content_pipeline_still_owns_notify_bridge() -> None:
    text = _read("extracted_content_pipeline/pipelines/notify.py")

    assert "send_pipeline_notification" in text
    assert "EXTRACTED_PIPELINE_STANDALONE" in text
