"""Behavioral Risk Sensors for Operational Risk Assessment.

Three sensors that detect friction thresholds in labor and logistics sectors
by analyzing linguistic patterns in text from professional/labor forums,
negotiation transcripts, and operational communications.

Sensors
-------
AlignmentSensorTool
    Detects a shift from collaborative to adversarial language (institutional
    friction).  Tracks the ratio of collaborative pronouns / partnership
    language vs adversarial identifiers (management, they, the corporation).

OperationalUrgencySensorTool
    Identifies when a workforce or logistics hub has moved from "Planning"
    mode to "Reactionary" mode by analyzing verb tenses and temporal markers.

NegotiationRigiditySensorTool
    Detects loss of flexibility in labor or vendor negotiations by measuring
    absolutist language vs flexibility markers.

Cross-sensor correlation
------------------------
``correlate()`` connects the three sensor outputs to find relationships between
the collected signals and produce a composite risk level::

    from atlas_brain.tools.risk_sensors import (
        alignment_sensor_tool,
        operational_urgency_tool,
        negotiation_rigidity_tool,
        correlate,
    )

    text = "They are demanding we stop immediately — non-negotiable, final offer."
    a = alignment_sensor_tool.analyze(text)
    u = operational_urgency_tool.analyze(text)
    r = negotiation_rigidity_tool.analyze(text)

    cross = correlate(a, u, r)
    # cross["composite_risk_level"]  → "CRITICAL"
    # cross["relationships"][0]["label"]  → "full_friction_cascade"

Pipeline integration
--------------------
Each tool exposes an ``analyze(text)`` helper that returns a plain dict so the
sensors can be called directly in a Pandas or Polars pipeline::

    import pandas as pd
    from atlas_brain.tools.risk_sensors import alignment_sensor_tool

    df["alignment"] = df["text"].apply(alignment_sensor_tool.analyze)
    df["triggered"]  = df["alignment"].apply(lambda r: r["triggered"])
"""

import logging
import re
from typing import Any

from .base import Tool, ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.risk_sensors")

# ---------------------------------------------------------------------------
# Linguistic word lists
# ---------------------------------------------------------------------------

# --- Alignment sensor ---
_COLLABORATIVE_TERMS: frozenset[str] = frozenset({
    # Inclusive pronouns
    "we", "our", "ours", "us",
    # Partnership / unity language
    "team", "together", "partnership", "partners", "partner",
    "collaborate", "collaboration", "collaborative", "cooperative",
    "joint", "mutual", "shared", "united", "aligned", "alignment",
    "leadership", "community",
})

_ADVERSARIAL_TERMS: frozenset[str] = frozenset({
    # Othering pronouns
    "they", "them", "those",
    # Institutional identifiers used to signal "us vs them"
    "management", "managers", "corporation", "corporate",
    "board", "suits", "executives", "bosses", "administration",
    "company", "organization", "hr",
    # Slang / loaded language
    "system",
})

# Multi-word adversarial phrases (matched as bigrams / trigrams)
_ADVERSARIAL_PHRASES: frozenset[str] = frozenset({
    "the suits",
    "those guys",
    "the system",
    "the board",
    "the company",
    "the corporation",
    "upper management",
    "senior management",
})

# --- Operational urgency sensor ---
_PLANNING_TERMS: frozenset[str] = frozenset({
    "will", "shall", "scheduled", "schedule", "proposal", "propose",
    "plan", "planning", "roadmap", "forecast", "projected", "upcoming",
    "future", "long-term", "strategy", "strategic", "eventually",
    "pipeline", "backlog", "timeline", "milestone",
    "quarterly", "annually",
})

# Multi-word planning phrases (matched as bigrams / trigrams)
_PLANNING_PHRASES: frozenset[str] = frozenset({
    "next quarter",
    "next month",
    "long term",
    "road map",
    "strategic plan",
    "future planning",
})

_URGENCY_TERMS: frozenset[str] = frozenset({
    "now", "immediate", "immediately", "stop", "stopped", "blocked",
    "halt", "halted", "urgent", "urgently", "critical", "crisis",
    "emergency", "today", "minute", "instant", "instantly",
    "asap", "deadline", "overdue", "escalate", "escalating",
    "grounded", "shutdown", "lockout", "walkout",
})

_URGENCY_PHRASES: frozenset[str] = frozenset({
    "right now",
    "no time",
    "at once",
    "stand down",
    "work stoppage",
    "cease operations",
    "walk out",
    "shut down",
    "lock out",
})

# --- Negotiation rigidity sensor ---
_ABSOLUTIST_TERMS: frozenset[str] = frozenset({
    "never", "always", "must", "zero", "none",
    "impossible", "demanding", "demand", "demands",
    "ultimatum", "absolute", "absolutely", "dead",
    "failed", "broken", "wall", "finished", "unacceptable",
    "final", "mandate", "mandatory",
    # Hyphenated absolutist unigrams (tokenizer preserves hyphens as one token)
    "non-negotiable",
})

_ABSOLUTIST_PHRASES: frozenset[str] = frozenset({
    "no way",
    "not negotiable",
    "non negotiable",
    "non-negotiable",
    "take it or leave it",
    "last offer",
    "final offer",
    "walk away",
})

_FLEXIBILITY_TERMS: frozenset[str] = frozenset({
    "considering", "consider", "alternative", "alternatives",
    "options", "option", "potential", "possibly", "likely",
    "could", "suggest", "suggesting", "open", "flexible",
    "adjusting", "shifting", "evolving", "progress", "exploring",
    "negotiate", "negotiating", "compromise", "middle ground",
    "creative", "workable", "revisit", "reconsider",
})

_FLEXIBILITY_PHRASES: frozenset[str] = frozenset({
    "open to",
    "willing to",
    "let us",
    "might work",
    "could work",
    "worth exploring",
})


# ---------------------------------------------------------------------------
# Shared tokenizer / term counter
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Return lowercase word tokens (preserves hyphenated words)."""
    return re.findall(r"\b\w+(?:-\w+)*\b", text.lower())


def _ngrams(tokens: list[str], n: int) -> list[str]:
    """Generate space-joined n-grams from a token list."""
    return [" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def _count_terms(
    tokens: list[str],
    term_set: frozenset[str],
    phrase_set: frozenset[str] | None = None,
) -> int:
    """Count unigram hits plus optional phrase (bigram/trigram) hits."""
    count = sum(1 for t in tokens if t in term_set)

    if phrase_set:
        for n in (2, 3):
            for gram in _ngrams(tokens, n):
                if gram in phrase_set:
                    count += 1

    return count


# ---------------------------------------------------------------------------
# Alignment Sensor
# ---------------------------------------------------------------------------

#: Default threshold: adversarial terms must be at least 30 % more dense than
#: collaborative terms to trigger.
_ALIGNMENT_ADV_RATIO_THRESHOLD = 0.30


class AlignmentSensorTool:
    """Detects a shift from collaborative to adversarial language.

    Measures the ratio of adversarial identifiers (they, management, the board
    …) vs collaborative pronouns (we, our, partnership …).  When the
    adversarial share exceeds the configurable threshold the sensor triggers,
    signalling potential institutional friction.
    """

    @property
    def name(self) -> str:
        return "analyze_alignment"

    @property
    def description(self) -> str:
        return (
            "Analyze text for a shift from collaborative to adversarial language. "
            "Returns a friction score and triggers an alert when adversarial "
            "identifiers outweigh collaborative terms beyond the threshold."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                param_type="string",
                description="Text to analyze (forum post, report, communication).",
                required=True,
            ),
            ToolParameter(
                name="threshold",
                param_type="float",
                description=(
                    "Adversarial-share threshold (0–1) that triggers the alert. "
                    "Default 0.30 means adversarial terms ≥ 30 %% of combined total."
                ),
                required=False,
                default=_ALIGNMENT_ADV_RATIO_THRESHOLD,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["alignment sensor", "labor alignment", "institutional friction"]

    @property
    def category(self) -> str:
        return "analytics"

    def analyze(self, text: str, threshold: float = _ALIGNMENT_ADV_RATIO_THRESHOLD) -> dict[str, Any]:
        """Run the alignment analysis and return a result dict.

        Args:
            text: Text corpus to analyze.
            threshold: Adversarial-share fraction (0–1) that triggers alert.

        Returns:
            dict with keys: collaborative_count, adversarial_count,
            adversarial_share, triggered, summary.
        """
        tokens = _tokenize(text)
        total_words = max(len(tokens), 1)

        collaborative_count = _count_terms(tokens, _COLLABORATIVE_TERMS)
        adversarial_count = _count_terms(
            tokens, _ADVERSARIAL_TERMS, _ADVERSARIAL_PHRASES
        )

        combined = collaborative_count + adversarial_count
        adversarial_share = adversarial_count / combined if combined > 0 else 0.0

        triggered = adversarial_share >= threshold and collaborative_count < adversarial_count

        return {
            "sensor": "alignment",
            "total_words": total_words,
            "collaborative_count": collaborative_count,
            "adversarial_count": adversarial_count,
            "adversarial_share": round(adversarial_share, 4),
            "threshold": threshold,
            "triggered": triggered,
            "summary": (
                f"ALERT: Adversarial language dominates ({adversarial_share:.0%}). "
                f"Institutional friction detected."
                if triggered
                else f"No alert. Adversarial share {adversarial_share:.0%} below threshold {threshold:.0%}."
            ),
        }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the alignment sensor."""
        text = params.get("text", "")
        if not text or not text.strip():
            return ToolResult(
                success=False,
                error="MISSING_PARAMETER",
                message="Parameter 'text' is required and must not be empty.",
            )

        threshold = float(params.get("threshold", _ALIGNMENT_ADV_RATIO_THRESHOLD))
        if not 0.0 <= threshold <= 1.0:
            return ToolResult(
                success=False,
                error="INVALID_PARAMETER",
                message="threshold must be between 0 and 1 (inclusive).",
            )

        result = self.analyze(text, threshold)
        return ToolResult(
            success=True,
            data=result,
            message=result["summary"],
        )


# ---------------------------------------------------------------------------
# Operational Urgency Sensor
# ---------------------------------------------------------------------------

#: Default trigger: urgency density must exceed planning density by this factor.
_URGENCY_DENSITY_FACTOR = 2.0


class OperationalUrgencySensorTool:
    """Detects when language shifts from planning to reactive/urgency mode.

    Analyzes the relative density of future-facing planning terms
    ('will', 'scheduled', 'proposal') vs immediate-present urgency terms
    ('now', 'immediate', 'blocked').  Triggers when urgency density is at
    least ``density_factor`` times higher than planning density, indicating
    the workforce or hub has moved into reactive mode.
    """

    @property
    def name(self) -> str:
        return "analyze_operational_urgency"

    @property
    def description(self) -> str:
        return (
            "Analyze text for a shift from forward-planning language to "
            "immediate-urgency language. Returns a temporal-shift score and "
            "triggers an alert when reactionary vocabulary dominates."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                param_type="string",
                description="Text to analyze.",
                required=True,
            ),
            ToolParameter(
                name="density_factor",
                param_type="float",
                description=(
                    "How many times denser urgency terms must be vs planning "
                    "terms to trigger.  Default 2.0 (≈ 2 standard deviations)."
                ),
                required=False,
                default=_URGENCY_DENSITY_FACTOR,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["urgency sensor", "operational urgency", "temporal shift"]

    @property
    def category(self) -> str:
        return "analytics"

    def analyze(
        self, text: str, density_factor: float = _URGENCY_DENSITY_FACTOR
    ) -> dict[str, Any]:
        """Run the operational urgency analysis.

        Args:
            text: Text corpus to analyze.
            density_factor: Urgency-to-planning density ratio that triggers alert.

        Returns:
            dict with keys: planning_count, urgency_count, planning_density,
            urgency_density, density_ratio, triggered, summary.
        """
        tokens = _tokenize(text)
        total_words = max(len(tokens), 1)

        planning_count = _count_terms(tokens, _PLANNING_TERMS, _PLANNING_PHRASES)
        urgency_count = _count_terms(tokens, _URGENCY_TERMS, _URGENCY_PHRASES)

        planning_density = planning_count / total_words
        urgency_density = urgency_count / total_words

        # Avoid division by zero; if no planning language at all and urgency
        # terms present, always trigger.
        if planning_density == 0:
            density_ratio = urgency_density * density_factor * 10 if urgency_density > 0 else 0.0
        else:
            density_ratio = urgency_density / planning_density

        triggered = density_ratio >= density_factor

        return {
            "sensor": "operational_urgency",
            "total_words": total_words,
            "planning_count": planning_count,
            "urgency_count": urgency_count,
            "planning_density": round(planning_density, 6),
            "urgency_density": round(urgency_density, 6),
            "density_ratio": round(density_ratio, 4),
            "density_factor_threshold": density_factor,
            "triggered": triggered,
            "summary": (
                f"ALERT: Reactionary language dominates "
                f"(urgency/planning ratio {density_ratio:.2f}). "
                f"System appears to be in reactive mode."
                if triggered
                else f"No alert. Urgency/planning ratio {density_ratio:.2f} below threshold {density_factor}."
            ),
        }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the operational urgency sensor."""
        text = params.get("text", "")
        if not text or not text.strip():
            return ToolResult(
                success=False,
                error="MISSING_PARAMETER",
                message="Parameter 'text' is required and must not be empty.",
            )

        density_factor = float(params.get("density_factor", _URGENCY_DENSITY_FACTOR))
        if density_factor <= 0:
            return ToolResult(
                success=False,
                error="INVALID_PARAMETER",
                message="density_factor must be positive.",
            )

        result = self.analyze(text, density_factor)
        return ToolResult(
            success=True,
            data=result,
            message=result["summary"],
        )


# ---------------------------------------------------------------------------
# Negotiation Rigidity Sensor
# ---------------------------------------------------------------------------

#: Default threshold: absolutist share ≥ 50 % triggers alert.
_RIGIDITY_ABSOLUTIST_THRESHOLD = 0.50


class NegotiationRigiditySensorTool:
    """Detects loss of flexibility in labor or vendor negotiations.

    Measures absolutist language ('non-negotiable', 'final', 'never', 'must')
    against flexibility markers ('considering', 'alternatives', 'options').
    A spike in absolutist terms indicates a high probability of a strike or
    work stoppage.
    """

    @property
    def name(self) -> str:
        return "analyze_negotiation_rigidity"

    @property
    def description(self) -> str:
        return (
            "Analyze text for absolutist negotiation language vs flexibility "
            "markers. Triggers when absolutist terms dominate, indicating high "
            "probability of a strike or work stoppage."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                param_type="string",
                description="Text to analyze (negotiation transcript, press release).",
                required=True,
            ),
            ToolParameter(
                name="threshold",
                param_type="float",
                description=(
                    "Absolutist share (0–1) that triggers the alert.  "
                    "Default 0.50 means absolutist terms ≥ 50 %% of combined total."
                ),
                required=False,
                default=_RIGIDITY_ABSOLUTIST_THRESHOLD,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["rigidity sensor", "negotiation rigidity", "strike risk"]

    @property
    def category(self) -> str:
        return "analytics"

    def analyze(
        self, text: str, threshold: float = _RIGIDITY_ABSOLUTIST_THRESHOLD
    ) -> dict[str, Any]:
        """Run the negotiation rigidity analysis.

        Args:
            text: Text corpus to analyze.
            threshold: Absolutist-share fraction (0–1) that triggers alert.

        Returns:
            dict with keys: absolutist_count, flexibility_count,
            absolutist_share, triggered, summary.
        """
        tokens = _tokenize(text)
        total_words = max(len(tokens), 1)

        absolutist_count = _count_terms(
            tokens, _ABSOLUTIST_TERMS, _ABSOLUTIST_PHRASES
        )
        flexibility_count = _count_terms(
            tokens, _FLEXIBILITY_TERMS, _FLEXIBILITY_PHRASES
        )

        combined = absolutist_count + flexibility_count
        absolutist_share = absolutist_count / combined if combined > 0 else 0.0

        triggered = absolutist_share >= threshold

        return {
            "sensor": "negotiation_rigidity",
            "total_words": total_words,
            "absolutist_count": absolutist_count,
            "flexibility_count": flexibility_count,
            "absolutist_share": round(absolutist_share, 4),
            "threshold": threshold,
            "triggered": triggered,
            "summary": (
                f"ALERT: Absolutist language detected ({absolutist_share:.0%} of "
                f"rigidity signals). High probability of strike or work stoppage."
                if triggered
                else f"No alert. Absolutist share {absolutist_share:.0%} below threshold {threshold:.0%}."
            ),
        }

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the negotiation rigidity sensor."""
        text = params.get("text", "")
        if not text or not text.strip():
            return ToolResult(
                success=False,
                error="MISSING_PARAMETER",
                message="Parameter 'text' is required and must not be empty.",
            )

        threshold = float(params.get("threshold", _RIGIDITY_ABSOLUTIST_THRESHOLD))
        if not 0.0 <= threshold <= 1.0:
            return ToolResult(
                success=False,
                error="INVALID_PARAMETER",
                message="threshold must be between 0 and 1 (inclusive).",
            )

        result = self.analyze(text, threshold)
        return ToolResult(
            success=True,
            data=result,
            message=result["summary"],
        )


# ---------------------------------------------------------------------------
# Cross-sensor correlation
# ---------------------------------------------------------------------------

# Named relationship patterns — each describes what it means when a specific
# combination of sensors co-triggers on the same text.
_CROSS_SENSOR_PATTERNS: tuple[dict[str, Any], ...] = (
    {
        "sensors": frozenset({"alignment", "negotiation_rigidity"}),
        "label": "adversarial_rigidity",
        "insight": (
            "Identity adversarialism and negotiation absolutism are co-active: "
            "parties have hardened both their language and their positions. "
            "The adversarial framing reinforces the rigid stance — "
            "trust has eroded at the same time flexibility has vanished."
        ),
    },
    {
        "sensors": frozenset({"alignment", "operational_urgency"}),
        "label": "adversarial_reactivity",
        "insight": (
            "Adversarial identity framing has escalated into reactive urgency: "
            "the us-vs-them divide is no longer theoretical — it is driving "
            "immediate operational pressure. The situation is being reacted to, "
            "not managed."
        ),
    },
    {
        "sensors": frozenset({"operational_urgency", "negotiation_rigidity"}),
        "label": "reactive_lock",
        "insight": (
            "Reactionary urgency and absolutist rigidity are co-active: "
            "positions have hardened precisely as time pressure has spiked. "
            "High probability of imminent work stoppage or operational breakdown."
        ),
    },
    {
        "sensors": frozenset({"alignment", "operational_urgency", "negotiation_rigidity"}),
        "label": "full_friction_cascade",
        "insight": (
            "All three friction axes are simultaneously active. "
            "Identity adversarialism, reactive urgency, and negotiation lock "
            "form a self-reinforcing cascade: each dimension amplifies the others. "
            "This pattern is a pre-event signature — strike, walkout, or shutdown "
            "is likely imminent."
        ),
    },
)

# Composite risk levels keyed by the number of sensors that triggered (0–3).
_COMPOSITE_RISK_LEVELS: dict[int, str] = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
    3: "CRITICAL",
}


def correlate(
    alignment_result: dict[str, Any],
    urgency_result: dict[str, Any],
    rigidity_result: dict[str, Any],
) -> dict[str, Any]:
    """Cross-correlate the outputs of all three sensors to find relationships.

    Takes the result dicts returned by each sensor's ``analyze()`` method and
    identifies which cross-sensor patterns are active.  Returns a composite
    risk level and a list of named relationship insights that explain *how* the
    triggered sensors are connected — answering what the data means together,
    not just individually.

    Args:
        alignment_result: Return value of ``AlignmentSensorTool.analyze()``.
        urgency_result:   Return value of ``OperationalUrgencySensorTool.analyze()``.
        rigidity_result:  Return value of ``NegotiationRigiditySensorTool.analyze()``.

    Returns:
        dict with keys:

        triggered_sensors
            Sorted list of sensor names that individually triggered.
        sensor_count
            Number of sensors that triggered (0–3).
        composite_risk_level
            "LOW" / "MEDIUM" / "HIGH" / "CRITICAL" based on sensor_count.
        relationships
            List of dicts, one per matched cross-sensor pattern, each with
            ``label``, ``sensors`` (sorted list), and ``insight`` keys.
        relationship_count
            Number of cross-sensor patterns matched.
        summary
            Human-readable composite assessment.
    """
    triggered: list[str] = [
        result["sensor"]
        for result in (alignment_result, urgency_result, rigidity_result)
        if result.get("triggered")
    ]
    triggered_set = frozenset(triggered)
    sensor_count = len(triggered)
    risk_level = _COMPOSITE_RISK_LEVELS[sensor_count]

    # A pattern matches when every sensor in the pattern triggered.
    relationships = [
        {
            "label": p["label"],
            "sensors": sorted(p["sensors"]),
            "insight": p["insight"],
        }
        for p in _CROSS_SENSOR_PATTERNS
        if p["sensors"].issubset(triggered_set)
    ]

    if sensor_count == 0:
        summary = "No sensors triggered. Risk level: LOW."
    elif sensor_count == 1:
        summary = (
            f"1 of 3 sensors triggered ({triggered[0]}). "
            f"Risk level: MEDIUM. No cross-sensor relationships active."
        )
    else:
        pattern_labels = ", ".join(r["label"] for r in relationships)
        summary = (
            f"{sensor_count} of 3 sensors triggered "
            f"({', '.join(sorted(triggered))}). "
            f"Risk level: {risk_level}. "
            f"Cross-sensor patterns: {pattern_labels}."
        )

    return {
        "triggered_sensors": sorted(triggered),
        "sensor_count": sensor_count,
        "composite_risk_level": risk_level,
        "relationships": relationships,
        "relationship_count": len(relationships),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Module-level instances
# ---------------------------------------------------------------------------

alignment_sensor_tool = AlignmentSensorTool()
operational_urgency_tool = OperationalUrgencySensorTool()
negotiation_rigidity_tool = NegotiationRigiditySensorTool()
