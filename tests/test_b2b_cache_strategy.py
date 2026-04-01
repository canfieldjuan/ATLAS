from pathlib import Path

from atlas_brain.services.b2b.cache_strategy import (
    exact_cache_namespaces,
    iter_core_b2b_cache_strategies,
    require_b2b_cache_strategy,
)


def _read_text(path: str) -> str:
    return Path(path).read_text()


def test_cache_strategy_registry_has_unique_stage_ids_and_namespaces():
    strategies = iter_core_b2b_cache_strategies()
    stage_ids = [strategy.stage_id for strategy in strategies]
    assert len(stage_ids) == len(set(stage_ids))

    namespaces = [strategy.namespace for strategy in strategies if strategy.namespace]
    assert len(namespaces) == len(set(namespaces))
    assert tuple(sorted(namespaces)) == tuple(sorted(exact_cache_namespaces()))


def test_exact_cache_strategies_reference_cache_namespace_and_helpers():
    helper_markers = (
        "prepare_b2b_exact_stage_request(",
        "prepare_b2b_exact_skill_stage_request(",
        "bind_b2b_exact_stage_request(",
        "lookup_b2b_exact_stage_text(",
        "store_b2b_exact_stage_text(",
        "run_b2b_exact_stage(",
    )
    for strategy in iter_core_b2b_cache_strategies():
        if strategy.mode != "exact":
            continue
        text = _read_text(strategy.file_path)
        assert strategy.stage_id in text
        assert any(marker in text for marker in helper_markers)
        assert "lookup_cached_text(" not in text
        assert "store_cached_text(" not in text
        assert "build_request_envelope(" not in text
        assert "build_skill_request_envelope(" not in text


def test_semantic_cache_strategy_uses_semantic_cache_and_evidence_hash():
    strategy = next(
        item for item in iter_core_b2b_cache_strategies()
        if item.stage_id == "b2b_battle_cards.sales_copy"
    )
    text = _read_text(strategy.file_path)
    assert "SemanticCache" in text
    assert "compute_evidence_hash" in text


def test_evidence_hash_strategies_use_hash_based_reuse():
    for stage_id in (
        "b2b_reasoning_synthesis.vendor",
        "b2b_reasoning_synthesis.cross_vendor",
    ):
        strategy = next(
            item for item in iter_core_b2b_cache_strategies()
            if item.stage_id == stage_id
        )
        text = _read_text(strategy.file_path)
        assert "evidence_hash" in text
    text = _read_text(
        "atlas_brain/autonomous/tasks/_b2b_cross_vendor_synthesis.py"
    )
    assert "compute_cross_vendor_evidence_hash" in text


def test_any_b2b_llm_content_file_is_declared_in_cache_registry():
    registered = {
        Path(strategy.file_path).as_posix()
        for strategy in iter_core_b2b_cache_strategies()
    }
    llm_markers = (
        "call_llm_with_skill(",
        ".chat(",
        "chat_async(",
    )
    for path in sorted(Path("atlas_brain/autonomous/tasks").glob("b2b_*.py")):
        text = _read_text(path.as_posix())
        if any(marker in text for marker in llm_markers):
            assert path.as_posix() in registered, path.as_posix()


def test_registry_lookup_requires_declared_strategy():
    strategy = require_b2b_cache_strategy("b2b_campaign_generation.content")
    assert strategy.namespace == "b2b_campaign_generation.content"
