"""Slim YAML-driven evidence engine for the extracted reasoning core.

Loads rules from `evidence_map.yaml` (shipped alongside this module via
PR-C1a) and provides the conclusions + suppression evaluation surface
that products consume to decide whether a vendor's evidence is
sufficient and which report sections should be suppressed or degraded.

This is the slim-core split documented in PR #82's audit. The
per-review enrichment surface (`compute_urgency`, `override_pain`,
`derive_recommend`, `derive_price_complaint`, `derive_budget_authority`)
is **not** included here -- it stays atlas-side until PR-C1e moves it
to `atlas_brain/reasoning/review_enrichment.py`. Reasoning core's
public boundary is the conclusions / suppression surface plus the
public `EvidenceDecision` / `ConclusionResult` / `SuppressionResult`
types defined in `extracted_reasoning_core.types`.

Public surface:

  - `EvidenceEngine` -- the engine class.
  - `EvidenceEngine.evaluate_conclusion(rule_id, evidence)` -- single
    rule, returns one `ConclusionResult`.
  - `EvidenceEngine.evaluate_conclusions(evidence)` -- iterates the
    full `_conclusions` rule-set and returns a list of
    `ConclusionResult`. Honors the `insufficient_data` short-circuit.
  - `EvidenceEngine.evaluate_suppression(section, evidence)` -- per-
    section render-suppression decision returned as
    `SuppressionResult`.
  - `EvidenceEngine.get_confidence_tier`,
    `EvidenceEngine.get_confidence_label` -- review-count based
    confidence labeling helpers.
  - `get_evidence_engine(map_path=None)` -- cached factory.
  - `reload_evidence_engine()` -- force-reload the cached engine
    after on-disk YAML edits.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .types import ConclusionResult, SuppressionResult

logger = logging.getLogger("extracted_reasoning_core.evidence_engine")

_DEFAULT_MAP_PATH = Path(__file__).parent / "evidence_map.yaml"


class EvidenceEngine:
    """Conclusion + suppression evaluator driven by a rules dict.

    Rules can be supplied in three shapes:

      - YAML/JSON file (``EvidenceEngine(map_path)``) -- loaded from
        disk; ``.json`` files are parsed with stdlib ``json``, all
        other suffixes use ``yaml.safe_load`` (yaml is imported lazily
        so the standalone extracted-pipeline CI -- which intentionally
        omits PyYAML -- can still import this module).
      - In-memory dict (``EvidenceEngine.from_rules(rules)``) -- no
        filesystem I/O. Used by callers that ship rules as Python
        literals, e.g. the content-pipeline wrapper.
      - Default (``EvidenceEngine()`` / no args) -- loads the YAML map
        shipped next to this module. Atlas-side use only; standalone
        callers should use ``from_rules`` instead.

    Slim-core split: per-review enrichment lives in
    `atlas_brain.reasoning.review_enrichment` (will be carved out in
    PR-C1e). This class only exposes the conclusions + suppression
    surface that the public reasoning core needs.
    """

    def __init__(self, map_path: str | Path | None = None) -> None:
        path = Path(map_path) if map_path else _DEFAULT_MAP_PATH
        with open(path, "rb") as f:
            raw_bytes = f.read()
        rules = _parse_rules_bytes(raw_bytes, path)
        self._init_from_rules(rules, str(path), raw_bytes)

    @classmethod
    def from_rules(cls, rules: dict[str, Any]) -> "EvidenceEngine":
        """Construct an engine directly from an in-memory rules dict.

        Used by callers (notably the content-pipeline wrapper) that
        ship rules as Python literals and do not want a filesystem
        dependency. The hash is computed over a canonical JSON dump of
        the dict so two callers with identical rules get identical
        ``map_hash`` fingerprints.
        """
        inst = cls.__new__(cls)
        canonical = json.dumps(rules, sort_keys=True).encode("utf-8")
        inst._init_from_rules(rules, "in-memory", canonical)
        return inst

    def _init_from_rules(
        self,
        rules: dict[str, Any],
        map_path_str: str,
        raw_bytes: bytes,
    ) -> None:
        self._rules: dict[str, Any] = rules
        self.map_hash: str = hashlib.sha256(raw_bytes).hexdigest()[:16]
        self.map_path: str = map_path_str
        # Enrichment rules are loaded for completeness (the atlas YAML
        # carries them for the atlas-side enrichment module) but this
        # slim core never reads them. No regex pre-compilation here --
        # those live in review_enrichment.py.
        self._enrichment = self._rules.get("enrichment", {})
        self._conclusions = self._rules.get("conclusions", {})
        self._suppression = self._rules.get("suppression", {})
        self._confidence_tiers = self._rules.get("confidence_tiers", {})

    # ------------------------------------------------------------------
    # Conclusion evaluation
    # ------------------------------------------------------------------

    def evaluate_conclusions(
        self,
        vendor_evidence: dict[str, Any],
    ) -> list[ConclusionResult]:
        """Evaluate every conclusion rule against vendor-level evidence.

        Honors the `insufficient_data` short-circuit: when the
        insufficient-data trigger fires, only that result is returned
        (matches atlas behavior).
        """
        results: list[ConclusionResult] = []

        # insufficient_data short-circuit
        insuf = self._conclusions.get("insufficient_data")
        if insuf:
            triggers = insuf.get("trigger", [])
            if triggers and all(
                self._check_requirement(t, vendor_evidence) for t in triggers
            ):
                results.append(
                    ConclusionResult(
                        conclusion_id="insufficient_data",
                        met=True,
                        confidence="insufficient",
                        fallback_label=insuf.get("label"),
                        fallback_action=insuf.get("action"),
                    )
                )
                return results

        for cid, spec in self._conclusions.items():
            if cid == "insufficient_data":
                continue
            results.append(self._evaluate_one_conclusion(cid, spec, vendor_evidence))

        return results

    def evaluate_conclusion(
        self,
        conclusion_id: str,
        vendor_evidence: dict[str, Any],
    ) -> ConclusionResult:
        """Evaluate a single conclusion rule by id.

        Public per the audit's Collision 4 resolution: callers can
        evaluate one rule without iterating the whole catalog. Returns
        a `ConclusionResult` with `met=False` and `confidence="insufficient"`
        when the rule id is unknown -- the caller decides whether that is
        an error or a no-op.
        """
        if conclusion_id == "insufficient_data":
            spec = self._conclusions.get("insufficient_data") or {}
            triggers = spec.get("trigger", [])
            all_met = bool(triggers) and all(
                self._check_requirement(t, vendor_evidence) for t in triggers
            )
            return ConclusionResult(
                conclusion_id="insufficient_data",
                met=all_met,
                confidence="insufficient",
                fallback_label=spec.get("label") if all_met else None,
                fallback_action=spec.get("action") if all_met else None,
            )

        spec = self._conclusions.get(conclusion_id)
        if not spec:
            return ConclusionResult(
                conclusion_id=conclusion_id,
                met=False,
                confidence="insufficient",
            )
        return self._evaluate_one_conclusion(conclusion_id, spec, vendor_evidence)

    def _evaluate_one_conclusion(
        self,
        conclusion_id: str,
        spec: dict[str, Any],
        vendor_evidence: dict[str, Any],
    ) -> ConclusionResult:
        """Evaluate one non-insufficient_data conclusion rule.

        Shared between `evaluate_conclusions` (iteration) and
        `evaluate_conclusion` (singular).
        """
        requires = spec.get("requires", [])
        all_met = all(
            self._check_requirement(r, vendor_evidence) for r in requires
        )

        confidence = (
            spec.get("confidence_when_met", "medium") if all_met else "insufficient"
        )

        # Amplifiers can boost confidence
        if all_met:
            for amp in spec.get("amplifiers", []):
                if self._check_requirement(amp, vendor_evidence) and amp.get(
                    "boost_confidence"
                ):
                    if confidence == "medium":
                        confidence = "high"

        fallback = spec.get("fallback", {})
        return ConclusionResult(
            conclusion_id=conclusion_id,
            met=all_met,
            confidence=confidence,
            fallback_label=fallback.get("label") if not all_met else None,
            fallback_action=fallback.get("action") if not all_met else None,
        )

    # ------------------------------------------------------------------
    # Suppression evaluation
    # ------------------------------------------------------------------

    def evaluate_suppression(
        self,
        section: str,
        evidence: dict[str, Any],
    ) -> SuppressionResult:
        """Per-section render-suppression decision."""
        spec = self._suppression.get(section)
        if not spec:
            return SuppressionResult()

        for rule in spec.get("suppress_when", []):
            if self._check_suppression_rule(rule, evidence):
                return SuppressionResult(
                    suppress=True,
                    fallback_label=spec.get("fallback_label"),
                )

        for rule in spec.get("degrade_when", []):
            if self._check_suppression_rule(rule, evidence):
                return SuppressionResult(
                    degrade=True,
                    disclaimer=spec.get("disclaimer"),
                )

        return SuppressionResult()

    # ------------------------------------------------------------------
    # Confidence labeling
    # ------------------------------------------------------------------

    def get_confidence_tier(self, total_reviews: Any) -> str:
        """Return confidence tier label for a review count.

        Coerces strict-numeric-shaped inputs (``120``, ``"120"``,
        ``"1,234"``, ``"28%"``) via ``_numeric_value`` so a single
        string-shaped review count does not silently fall through to
        "insufficient". Mixed-text inputs like ``"120 reviews"`` are
        not parsed -- callers should pre-strip non-numeric content.
        """
        review_count = _numeric_value(total_reviews) or 0.0
        for tier_name in ("high", "medium", "low"):
            tier = self._confidence_tiers.get(tier_name, {})
            min_reviews = _numeric_value(tier.get("min_reviews")) or 0.0
            if review_count >= min_reviews:
                return tier_name
        return "insufficient"

    def get_confidence_label(self, total_reviews: int) -> str:
        """Return human-readable confidence label."""
        tier_name = self.get_confidence_tier(total_reviews)
        tier = self._confidence_tiers.get(tier_name, {})
        return tier.get("label", tier_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_field(data: dict[str, Any], field_path: str) -> Any:
        """Resolve a dotted field path against a nested dict."""
        parts = field_path.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _check_condition_simple(
        self, cond: dict[str, Any], context: dict[str, Any],
    ) -> bool:
        """Check a simple {field, eq/lte/gte/in} condition.

        Carried from atlas's evidence_engine; kept here because it is
        the shared shape used by some YAML rules even on the slim-core
        side. The atlas-side enrichment module also uses this shape.
        """
        field = cond.get("field", "")
        value = self._resolve_field(context, field)
        if "eq" in cond:
            return value == cond["eq"]
        if "lte" in cond:
            value_num = _numeric_value(value)
            cond_num = _numeric_value(cond["lte"])
            return value_num is not None and cond_num is not None and value_num <= cond_num
        if "gte" in cond:
            value_num = _numeric_value(value)
            cond_num = _numeric_value(cond["gte"])
            return value_num is not None and cond_num is not None and value_num >= cond_num
        if "in" in cond:
            return value in cond["in"]
        return False

    def _check_requirement(
        self, req: dict[str, Any], evidence: dict[str, Any],
    ) -> bool:
        """Check a conclusion requirement against vendor evidence."""
        field = req.get("field", "")
        value = self._resolve_field(evidence, field)
        op = req.get("operator", "eq")

        if op in {"gte", "gt", "lte", "lt"}:
            value_num = _numeric_value(value)
            rule_num = _numeric_value(req.get("value"))
            if value_num is None or rule_num is None:
                return False
            if op == "gte":
                return value_num >= rule_num
            if op == "gt":
                return value_num > rule_num
            if op == "lte":
                return value_num <= rule_num
            return value_num < rule_num

        if op == "eq":
            return value == req.get("value")
        if op == "in":
            return value in req.get("values", [])
        if op == "exists":
            return value is not None and value != ""
        if op == "min_count":
            expected = _numeric_value(req.get("value"))
            return (
                isinstance(value, (list, tuple, set, dict))
                and expected is not None
                and len(value) >= expected
            )
        return False

    def _check_suppression_rule(
        self, rule: dict[str, Any], evidence: dict[str, Any],
    ) -> bool:
        """Check a suppression condition.

        Accepts two shapes:
          - shorthand: ``{field: ..., lt: 5}`` (the canonical form in
            core's `evidence_map.yaml`)
          - explicit:  ``{field: ..., operator: lt, value: 5}`` (the
            form some content-pipeline rules use; carried forward in
            PR-C1i so a single suppression-section rule shape works
            on both pipelines without forcing a YAML rewrite)
        """
        # Explicit-operator form delegates to _check_requirement.
        if "operator" in rule:
            return self._check_requirement(rule, evidence)

        field = rule.get("field", "")
        value = self._resolve_field(evidence, field)

        for op_key in ("lt", "lte", "gt", "gte"):
            if op_key in rule:
                value_num = _numeric_value(value)
                rule_num = _numeric_value(rule[op_key])
                if value_num is None or rule_num is None:
                    return False
                if op_key == "lt":
                    return value_num < rule_num
                if op_key == "lte":
                    return value_num <= rule_num
                if op_key == "gt":
                    return value_num > rule_num
                return value_num >= rule_num
        if "eq" in rule:
            return value == rule["eq"]

        return False


# ------------------------------------------------------------------
# Module-level cached factory
# ------------------------------------------------------------------


_engine: EvidenceEngine | None = None
_engine_path: str | None = None


def get_evidence_engine(map_path: str | Path | None = None) -> EvidenceEngine:
    """Return a cached `EvidenceEngine` instance.

    The cache is keyed by resolved path, so passing a different
    `map_path` reloads. In-process YAML edits with the same path
    require `reload_evidence_engine()` to pick up changes.

    Atlas's factory consulted `settings.b2b_churn.evidence_map_path`
    when no path was passed. Reasoning core stays config-free here:
    callers pass an explicit path or get the default YAML shipped
    next to this module. Atlas-side wiring (PR-C1f) re-introduces the
    settings lookup in the atlas adapter.
    """
    global _engine, _engine_path

    resolved = str(map_path) if map_path else str(_DEFAULT_MAP_PATH)

    if _engine is not None and _engine_path == resolved:
        return _engine

    try:
        _engine = EvidenceEngine(resolved)
    except FileNotFoundError:
        if resolved != str(_DEFAULT_MAP_PATH):
            logger.error(
                "Evidence map not found at %s, falling back to default",
                resolved,
            )
            resolved = str(_DEFAULT_MAP_PATH)
            _engine = EvidenceEngine(resolved)
        else:
            raise
    _engine_path = resolved
    logger.info(
        "Evidence engine loaded: %s (hash=%s, %d conclusions, %d suppression sections)",
        resolved,
        _engine.map_hash,
        len(_engine._conclusions),
        len(_engine._suppression),
    )
    return _engine


def reload_evidence_engine() -> EvidenceEngine:
    """Force-reload the cached engine (e.g. after on-disk YAML edits)."""
    global _engine, _engine_path
    _engine = None
    _engine_path = None
    return get_evidence_engine()


def _parse_rules_bytes(raw_bytes: bytes, path: Path) -> dict[str, Any]:
    """Parse rule bytes from disk into a dict.

    ``.json`` files use stdlib ``json`` so the standalone extracted-
    pipeline CI (which omits PyYAML) can still load JSON-shaped rule
    files in tests. All other suffixes use ``yaml.safe_load`` -- yaml
    is imported lazily here so the module imports cleanly even when
    PyYAML is not installed.
    """
    if path.suffix.lower() == ".json":
        loaded = json.loads(raw_bytes.decode("utf-8"))
    else:
        try:
            import yaml  # type: ignore[import-untyped]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"PyYAML is required to load {path} (suffix: {path.suffix}). "
                f"Install pyyaml, switch to a .json rule file, or use "
                f"EvidenceEngine.from_rules(...) to pass rules as a dict."
            ) from exc
        loaded = yaml.safe_load(raw_bytes)
    if not isinstance(loaded, dict):
        raise ValueError(f"Evidence map must decode to a dict: {path}")
    return loaded


def _numeric_value(value: Any) -> float | None:
    """Coerce common numeric-shaped values to float, returning None on bad input.

    Handles ints, floats, percent-suffixed strings ("28%"), thousands
    separators ("1,234"), and bare decimal strings ("7.2"). Booleans
    return None to avoid `True == 1.0` collisions in numeric comparisons.

    Carried forward from `extracted_content_pipeline.reasoning.evidence_engine`
    in PR-C1i so the slim core can absorb the same defensive coercion the
    drifted content_pipeline fork already had.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if not stripped:
            return None
        if stripped.endswith("%"):
            stripped = stripped[:-1]
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


__all__ = [
    "EvidenceEngine",
    "get_evidence_engine",
    "reload_evidence_engine",
]
