"""Stratified Reasoning Engine -- core 3-mode dispatch.

Phase 1: Recall (semantic cache) and Reason (full LLM) only.
Reconstitute (differential patching) will be added in Phase 2.

Flow:
    1. Build pattern_sig from vendor + evidence hash
    2. Try Recall (semantic cache lookup)
       - Hit with effective_confidence > threshold -> return cached
    3. Find similar episodic traces (vector search for LLM context)
    4. Full Reason via LLM -> store in both memories -> return
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

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
    """Core dispatch engine: recall -> (reconstitute P2) -> reason."""

    def __init__(self, cache: SemanticCache, episodic: EpisodicStore):
        self._cache = cache
        self._episodic = episodic

    async def analyze(
        self,
        vendor_name: str,
        evidence: dict[str, Any],
        *,
        product_category: str = "",
        force_reason: bool = False,
    ) -> ReasoningResult:
        """Main entry point. Decides recall or reason."""
        ev_hash = compute_evidence_hash(evidence)
        pattern_sig = self._build_pattern_sig(vendor_name, ev_hash)

        # 1. Try recall (unless forced)
        if not force_reason:
            cached = await self._recall(pattern_sig, ev_hash)
            if cached is not None:
                await self._log_metacognition("recall", 0)
                return cached

        # 2. Find similar traces for LLM context
        prior_traces = await self._find_prior_context(evidence, vendor_name)

        # 3. Full reason
        result = await self._reason(
            vendor_name, evidence, prior_traces,
            product_category=product_category,
            pattern_sig=pattern_sig,
            evidence_hash=ev_hash,
        )
        await self._log_metacognition("reason", result.tokens_used)
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

    async def _log_metacognition(self, mode: str, tokens_used: int) -> None:
        """Increment counters in the reasoning_metacognition table."""
        try:
            pool = self._cache._pool
            today = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            tomorrow = today.replace(hour=23, minute=59, second=59)

            col = "recall_hits" if mode == "recall" else "full_reasons"
            saved = 2000 if mode == "recall" else 0  # estimated avg tokens for a full reason

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
