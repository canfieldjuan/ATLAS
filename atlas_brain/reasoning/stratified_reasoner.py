"""Stratified Reasoning Engine -- core 3-mode dispatch.

Three cognitive modes:
    Recall       - Semantic cache hit, fresh + same evidence hash  (~0 tokens)
    Reconstitute - Cache hit but evidence changed <30% delta       (~30% tokens)
    Reason       - Full LLM synthesis (no cache or large delta)    (~100% tokens)

Also integrates:
    - Metacognitive monitor (surprise detection, exploration budget)
    - Hierarchical tier context (inherited priors from higher tiers)
    - Differential engine (evidence diff classification)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .differential import EvidenceDiff, classify_evidence, reconstitute
from .episodic_store import (
    ConclusionNode,
    EpisodicStore,
    EvidenceNode,
    ReasoningTrace,
)
from .semantic_cache import CacheEntry, SemanticCache, compute_evidence_hash

logger = logging.getLogger("atlas.reasoning.stratified")

# System prompt for Phase 1 (inline; replaced by skill file in WS6)
_REASON_SYSTEM_PROMPT = """\
You are a B2B churn intelligence analyst. Given evidence about a software \
vendor, produce a structured JSON conclusion.

Output JSON with these fields:
{
  "archetype": "<one of: pricing_shock, feature_gap, acquisition_decay, \
leadership_redesign, integration_break, support_collapse, \
category_disruption, compliance_gap, mixed, stable>",
  "confidence": <0.0-1.0>,
  "executive_summary": "<2-3 sentence assessment>",
  "key_signals": ["<signal1>", "<signal2>", ...],
  "risk_level": "<low|medium|high|critical>",
  "falsification_conditions": ["<what would prove this wrong>"],
  "uncertainty_sources": ["<what data is missing or weak>"]
}

Be precise. Cite specific numbers from the evidence. If data is insufficient \
for a confident assessment, say so and lower confidence accordingly.\
"""


@dataclass
class ReasoningResult:
    """Result of a stratified reasoning analysis."""

    mode: str           # "recall", "reconstitute", "reason"
    conclusion: dict[str, Any]
    confidence: float
    pattern_sig: str
    evidence_hash: str
    tokens_used: int
    cached: bool
    trace_id: str | None = None


class StratifiedReasoner:
    """Core dispatch engine: recall -> reconstitute -> reason."""

    def __init__(
        self,
        cache: SemanticCache,
        episodic: EpisodicStore,
        metacognition: Any | None = None,
    ):
        self._cache = cache
        self._episodic = episodic
        self._meta = metacognition  # MetacognitiveMonitor (optional)

    async def analyze(
        self,
        vendor_name: str,
        evidence: dict[str, Any],
        *,
        product_category: str = "",
        force_reason: bool = False,
        tier_context: dict[str, Any] | None = None,
    ) -> ReasoningResult:
        """Main entry point. Decides: recall, reconstitute, or reason.

        Flow:
            1. Check metacognition for forced exploration/surprise
            2. Try recall (semantic cache, same evidence hash)
            3. If evidence changed, try reconstitute (diff < 30%)
            4. Fall back to full reason
        """
        ev_hash = compute_evidence_hash(evidence)
        pattern_sig = self._build_pattern_sig(vendor_name, ev_hash)

        # 0. Metacognitive overrides
        if not force_reason and self._meta:
            if self._meta.should_force_exploration():
                logger.info("Exploration sample forced for %s", vendor_name)
                force_reason = True

        # 1. Try recall (unless forced)
        if not force_reason:
            cached = await self._recall(pattern_sig, ev_hash)
            if cached is not None:
                await self._log_metacognition("recall", 0, cached.conclusion.get("archetype", ""))
                return cached

        # 2. Try reconstitute: cache hit on vendor but evidence changed
        if not force_reason:
            reconstituted = await self._try_reconstitute(
                vendor_name, evidence, ev_hash, pattern_sig, product_category,
            )
            if reconstituted is not None:
                await self._log_metacognition(
                    "reconstitute", reconstituted.tokens_used,
                    reconstituted.conclusion.get("archetype", ""),
                )
                return reconstituted

        # 3. Find similar traces for LLM context
        prior_traces = await self._find_prior_context(evidence, vendor_name)

        # 4. Full reason
        result = await self._reason(
            vendor_name, evidence, prior_traces,
            product_category=product_category,
            pattern_sig=pattern_sig,
            evidence_hash=ev_hash,
            tier_context=tier_context,
        )
        await self._log_metacognition("reason", result.tokens_used, result.conclusion.get("archetype", ""))
        return result

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    async def _recall(self, pattern_sig: str, evidence_hash: str) -> ReasoningResult | None:
        """Try semantic cache. Return None on miss or stale."""
        entry = await self._cache.lookup(pattern_sig)
        if entry is None:
            return None

        # If evidence changed since cache was stored, don't recall
        if entry.evidence_hash and entry.evidence_hash != evidence_hash:
            logger.info(
                "Cache hit for %s but evidence changed (old=%s new=%s)",
                pattern_sig, entry.evidence_hash, evidence_hash,
            )
            return None

        return ReasoningResult(
            mode="recall",
            conclusion=entry.conclusion,
            confidence=entry.effective_confidence or entry.confidence,
            pattern_sig=pattern_sig,
            evidence_hash=evidence_hash,
            tokens_used=0,
            cached=True,
        )

    # ------------------------------------------------------------------
    # Reconstitute (Phase 2)
    # ------------------------------------------------------------------

    async def _try_reconstitute(
        self,
        vendor_name: str,
        new_evidence: dict[str, Any],
        new_ev_hash: str,
        new_pattern_sig: str,
        product_category: str,
    ) -> ReasoningResult | None:
        """Try to patch an existing cached conclusion with the evidence delta.

        Looks for a prior cache entry for the same vendor (any evidence hash).
        If found and the diff ratio is < 30%, uses the LLM to patch only the
        delta instead of re-reasoning from scratch.
        """
        # Find any active cache entry for this vendor
        entries = await self._cache.lookup_by_class("", vendor_name=vendor_name, limit=1)
        if not entries:
            return None

        entry = entries[0]
        if entry.effective_confidence is not None and entry.effective_confidence < 0.3:
            return None  # too stale to reconstitute from

        # Reconstruct old evidence from the stored evidence hash
        # We need the actual old evidence for diffing -- check episodic store
        old_evidence = await self._load_old_evidence(entry.pattern_sig)
        if not old_evidence:
            return None

        # Classify the diff
        diff = classify_evidence(old_evidence, new_evidence)
        logger.info("Diff for %s: %s", vendor_name, diff.summary())

        if not diff.should_reconstitute:
            return None  # delta too large, need full reason

        # Patch the conclusion with only the delta
        updated_conclusion, tokens = await reconstitute(
            vendor_name, entry.conclusion, diff, new_evidence,
        )

        if not updated_conclusion or "error" in updated_conclusion:
            return None

        new_confidence = updated_conclusion.get("confidence", entry.confidence)
        archetype = updated_conclusion.get("archetype", entry.conclusion_type)

        # Update semantic cache with new conclusion + evidence hash
        updated_entry = CacheEntry(
            pattern_sig=new_pattern_sig,
            pattern_class=archetype,
            vendor_name=vendor_name,
            product_category=product_category,
            conclusion=updated_conclusion,
            confidence=new_confidence,
            reasoning_steps=entry.reasoning_steps,
            boundary_conditions=entry.boundary_conditions,
            falsification_conditions=updated_conclusion.get("falsification_conditions", []),
            uncertainty_sources=updated_conclusion.get("uncertainty_sources", []),
            conclusion_type=archetype,
            evidence_hash=new_ev_hash,
        )
        try:
            await self._cache.store(updated_entry)
            # Invalidate the old entry if pattern_sig changed
            if entry.pattern_sig != new_pattern_sig:
                await self._cache.invalidate(entry.pattern_sig, reason="superseded by reconstitute")
        except Exception:
            logger.warning("Failed to update cache after reconstitute", exc_info=True)

        return ReasoningResult(
            mode="reconstitute",
            conclusion=updated_conclusion,
            confidence=new_confidence,
            pattern_sig=new_pattern_sig,
            evidence_hash=new_ev_hash,
            tokens_used=tokens,
            cached=False,
        )

    async def _load_old_evidence(self, pattern_sig: str) -> dict[str, Any] | None:
        """Load the evidence from a prior episodic trace for diffing."""
        try:
            traces = await self._episodic.get_traces_for_vendor("", limit=1)
            # Better: search by pattern_sig
            driver = await self._episodic._get_driver()
            async with driver.session() as session:
                result = await session.run(
                    """
                    MATCH (t:ReasoningTrace {pattern_sig: $sig, group_id: $gid})
                          -[:SUPPORTED_BY]->(e:EvidenceNode)
                    RETURN e.type AS type, e.source AS source, e.value AS value
                    """,
                    sig=pattern_sig,
                    gid="b2b-reasoning",
                )
                evidence = {}
                async for record in result:
                    key = record["type"] or record["source"]
                    val = record["value"]
                    if key in evidence:
                        if isinstance(evidence[key], list):
                            evidence[key].append(val)
                        else:
                            evidence[key] = [evidence[key], val]
                    else:
                        evidence[key] = val
                return evidence if evidence else None
        except Exception:
            logger.debug("Failed to load old evidence for %s", pattern_sig, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Find prior context
    # ------------------------------------------------------------------

    async def _find_prior_context(
        self, evidence: dict[str, Any], vendor_name: str
    ) -> list[ReasoningTrace]:
        """Find similar episodic traces to provide context to the LLM."""
        try:
            summary = self._evidence_summary(evidence, vendor_name)
            embedding = self._episodic.embed_text(summary)
            traces = await self._episodic.find_similar(embedding, limit=3)
            if traces:
                logger.info(
                    "Found %d similar traces (best=%.3f)",
                    len(traces), traces[0].similarity_score or 0,
                )
            return traces
        except Exception:
            logger.warning("Episodic similarity search failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Reason
    # ------------------------------------------------------------------

    async def _reason(
        self,
        vendor_name: str,
        evidence: dict[str, Any],
        prior_traces: list[ReasoningTrace],
        *,
        product_category: str = "",
        pattern_sig: str = "",
        evidence_hash: str = "",
        tier_context: dict[str, Any] | None = None,
    ) -> ReasoningResult:
        """Full LLM synthesis. Store result in both memories."""
        from ..pipelines.llm import get_pipeline_llm, parse_json_response
        from ..services.protocols import Message

        # Build user payload
        payload = {
            "vendor_name": vendor_name,
            "product_category": product_category,
            "evidence": evidence,
        }
        if tier_context and tier_context.get("inherited_priors"):
            payload["tier_context"] = tier_context
        if prior_traces:
            payload["similar_patterns"] = [
                {
                    "vendor": t.vendor_name,
                    "archetype": t.conclusion_type,
                    "confidence": t.confidence,
                    "similarity": t.similarity_score,
                }
                for t in prior_traces
            ]

        llm = get_pipeline_llm(workload="vllm", auto_activate_ollama=True)
        if llm is None:
            logger.error("No LLM available for stratified reasoning")
            return ReasoningResult(
                mode="reason",
                conclusion={"error": "no_llm_available"},
                confidence=0.0,
                pattern_sig=pattern_sig,
                evidence_hash=evidence_hash,
                tokens_used=0,
                cached=False,
            )

        messages = [
            Message(role="system", content=_REASON_SYSTEM_PROMPT),
            Message(
                role="user",
                content=json.dumps(payload, separators=(",", ":"), default=str),
            ),
        ]

        t0 = time.monotonic()
        try:
            result = llm.chat(messages=messages, max_tokens=2048, temperature=0.3)
            text = result.get("response", "").strip()
            usage = result.get("usage", {})
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

            # Clean think tags
            import re
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            conclusion = parse_json_response(text)
        except Exception:
            logger.exception("LLM reasoning failed for %s", vendor_name)
            return ReasoningResult(
                mode="reason",
                conclusion={"error": "llm_failed"},
                confidence=0.0,
                pattern_sig=pattern_sig,
                evidence_hash=evidence_hash,
                tokens_used=0,
                cached=False,
            )

        confidence = conclusion.get("confidence", 0.5)
        archetype = conclusion.get("archetype", "unknown")
        duration_ms = (time.monotonic() - t0) * 1000

        logger.info(
            "Reasoned %s -> %s (conf=%.2f, %d tokens, %.0fms)",
            vendor_name, archetype, confidence, tokens, duration_ms,
        )

        # Store in semantic cache
        cache_entry = CacheEntry(
            pattern_sig=pattern_sig,
            pattern_class=archetype,
            vendor_name=vendor_name,
            product_category=product_category,
            conclusion=conclusion,
            confidence=confidence,
            reasoning_steps=[],
            boundary_conditions={},
            falsification_conditions=conclusion.get("falsification_conditions", []),
            uncertainty_sources=conclusion.get("uncertainty_sources", []),
            conclusion_type=archetype,
            evidence_hash=evidence_hash,
        )
        try:
            await self._cache.store(cache_entry)
        except Exception:
            logger.warning("Failed to store semantic cache for %s", pattern_sig, exc_info=True)

        # Store in episodic memory
        trace_id = None
        try:
            evidence_summary = self._evidence_summary(evidence, vendor_name)
            embedding = self._episodic.embed_text(evidence_summary)

            evidence_nodes = self._build_evidence_nodes(evidence)
            conclusion_node = ConclusionNode(
                claim=conclusion.get("executive_summary", ""),
                confidence=confidence,
                evidence_chain=[e.id for e in evidence_nodes],
            )

            trace = ReasoningTrace(
                vendor_name=vendor_name,
                category=product_category,
                conclusion_type=archetype,
                confidence=confidence,
                pattern_sig=pattern_sig,
                trace_embedding=embedding,
                evidence=evidence_nodes,
                conclusions=[conclusion_node],
            )
            trace_id = await self._episodic.store_trace(trace)
        except Exception:
            logger.warning("Failed to store episodic trace for %s", vendor_name, exc_info=True)

        return ReasoningResult(
            mode="reason",
            conclusion=conclusion,
            confidence=confidence,
            pattern_sig=pattern_sig,
            evidence_hash=evidence_hash,
            tokens_used=tokens,
            cached=False,
            trace_id=trace_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_pattern_sig(self, vendor_name: str, evidence_hash: str) -> str:
        """Deterministic signature: vendor + evidence hash."""
        safe_name = vendor_name.lower().replace(" ", "_").replace(".", "")
        return f"{safe_name}:{evidence_hash}"

    @staticmethod
    def _evidence_summary(evidence: dict[str, Any], vendor_name: str) -> str:
        """Build a text summary of evidence for embedding."""
        parts = [f"Vendor: {vendor_name}"]
        for key, val in evidence.items():
            if isinstance(val, (list, dict)):
                parts.append(f"{key}: {json.dumps(val, default=str)[:200]}")
            else:
                parts.append(f"{key}: {val}")
        return " | ".join(parts)[:1000]

    @staticmethod
    def _build_evidence_nodes(evidence: dict[str, Any]) -> list[EvidenceNode]:
        """Convert evidence dict into EvidenceNode list."""
        import uuid

        nodes = []
        for key, val in evidence.items():
            if isinstance(val, list):
                for i, item in enumerate(val[:10]):
                    nodes.append(EvidenceNode(
                        id=str(uuid.uuid4()),
                        type=key,
                        source=key,
                        value=str(item)[:500],
                    ))
            else:
                nodes.append(EvidenceNode(
                    id=str(uuid.uuid4()),
                    type=key,
                    source=key,
                    value=str(val)[:500],
                ))
        return nodes

    # ------------------------------------------------------------------
    # Metacognition
    # ------------------------------------------------------------------

    async def _log_metacognition(self, mode: str, tokens_used: int, conclusion_type: str = "") -> None:
        """Record reasoning outcome via the metacognitive monitor."""
        if self._meta:
            self._meta.record(mode, tokens_used, conclusion_type)
            return

        # Fallback: direct DB write (Phase 1 compat)
        try:
            pool = self._cache._pool
            today = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            tomorrow = today.replace(hour=23, minute=59, second=59)

            col_map = {"recall": "recall_hits", "reconstitute": "reconstitute_hits", "reason": "full_reasons"}
            col = col_map.get(mode, "full_reasons")
            saved = {"recall": 2000, "reconstitute": 1400, "reason": 0}.get(mode, 0)

            await pool.execute(
                f"""
                INSERT INTO reasoning_metacognition (
                    period_start, period_end, total_queries,
                    {col}, total_tokens_saved, total_tokens_spent
                ) VALUES ($1, $2, 1, 1, $3, $4)
                ON CONFLICT (period_start) DO UPDATE SET
                    total_queries = reasoning_metacognition.total_queries + 1,
                    {col} = reasoning_metacognition.{col} + 1,
                    total_tokens_saved = reasoning_metacognition.total_tokens_saved + $3,
                    total_tokens_spent = reasoning_metacognition.total_tokens_spent + $4
                """,
                today,
                tomorrow,
                saved,
                tokens_used,
            )
        except Exception:
            logger.debug("Metacognition logging failed", exc_info=True)
