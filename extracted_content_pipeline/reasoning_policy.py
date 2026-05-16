"""Reasoning-depth preset catalog for AI Content Ops.

This module is pure policy data. It does not construct providers, call LLMs,
read storage, or change generation behavior. Runtime seams can use it to
validate host-facing choices before later slices wire richer reasoning.

Source recommendations:
``docs/audits/content_ops_reasoning_policy_audit_2026-05-16.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


ReasoningPreset = Literal[
    "none", "context_only", "single_pass", "multi_pass_light",
    "multi_pass_structured", "multi_pass_strict",
]


@dataclass(frozen=True)
class ReasoningPresetDefinition:
    """Catalog metadata for a host-selectable reasoning preset.

    Most capability flags are delivered whenever the preset is requested.
    ``falsification`` means the preset can run falsification when the host also
    supplies explicit falsification rules; the catalog flag alone does not
    create an implicit default falsification policy.
    """

    id: ReasoningPreset
    label: str
    description: str
    generated_reasoning: bool = False
    multi_pass: bool = False
    narrative_planning: bool = False
    falsification: bool = False
    output_validation: bool = False
    blocking_validation: bool = False


@dataclass(frozen=True)
class OutputReasoningPolicy:
    output: str
    default_preset: ReasoningPreset
    supported_presets: tuple[ReasoningPreset, ...]

    def __post_init__(self) -> None:
        if not self.supported_presets:
            raise ValueError(f"{self.output}: supported_presets must not be empty")
        unknown = [p for p in self.supported_presets if p not in REASONING_PRESETS]
        if self.default_preset not in REASONING_PRESETS:
            unknown.append(self.default_preset)
        if unknown:
            raise ValueError(f"{self.output}: unknown reasoning preset(s): {unknown}")
        if self.default_preset not in self.supported_presets:
            raise ValueError(
                f"{self.output}: default_preset must be included in supported_presets"
            )

    def supports(self, preset: str) -> bool:
        """Return membership only; use resolve_reasoning_policy for validation."""
        return preset in self.supported_presets


REASONING_PRESETS: Mapping[str, ReasoningPresetDefinition] = MappingProxyType({
    "none": ReasoningPresetDefinition(
        id="none",
        label="None",
        description="No reasoning provider is used.",
    ),
    "context_only": ReasoningPresetDefinition(
        id="context_only",
        label="Context Only",
        description="Use host-provided reasoning context; do not generate new reasoning.",
    ),
    "single_pass": ReasoningPresetDefinition(
        id="single_pass",
        label="Single Pass",
        description="Generate lightweight reasoning with one LLM call.",
        generated_reasoning=True,
    ),
    "multi_pass_light": ReasoningPresetDefinition(
        id="multi_pass_light",
        label="Multi-Pass Light",
        description="Run bounded multi-pass reasoning without rich policy gates.",
        generated_reasoning=True,
        multi_pass=True,
    ),
    "multi_pass_structured": ReasoningPresetDefinition(
        id="multi_pass_structured",
        label="Multi-Pass Structured",
        description="Run multi-pass reasoning with planning and validation metadata.",
        generated_reasoning=True,
        multi_pass=True,
        narrative_planning=True,
        output_validation=True,
    ),
    "multi_pass_strict": ReasoningPresetDefinition(
        id="multi_pass_strict",
        label="Multi-Pass Strict",
        description="Run structured multi-pass reasoning with planning and validation gates.",
        generated_reasoning=True,
        multi_pass=True,
        narrative_planning=True,
        falsification=True,
        output_validation=True,
        blocking_validation=True,
    ),
})


_MULTI_PASS_OPTIONAL: tuple[ReasoningPreset, ...] = (
    "none", "context_only", "single_pass", "multi_pass_light",
)
_STRUCTURED_PRESETS: tuple[ReasoningPreset, ...] = (
    "none", "context_only", "single_pass", "multi_pass_light",
    "multi_pass_structured", "multi_pass_strict",
)
_LANDING_PAGE_PRESETS: tuple[ReasoningPreset, ...] = (
    "none", "context_only", "single_pass", "multi_pass_light",
    "multi_pass_structured",
)

PACKAGED_REASONING_RUNTIME_OUTPUTS: tuple[str, ...] = ("report", "sales_brief")
PACKAGED_REASONING_RUNTIME_PRESETS: tuple[ReasoningPreset, ...] = (
    "multi_pass_structured",
    "multi_pass_strict",
)
NOOP_REASONING_PRESETS: tuple[ReasoningPreset, ...] = tuple(
    preset
    for preset, definition in REASONING_PRESETS.items()
    if not definition.generated_reasoning
)


OUTPUT_REASONING_POLICIES: Mapping[str, OutputReasoningPolicy] = MappingProxyType({
    "signal_extraction": OutputReasoningPolicy(
        output="signal_extraction",
        default_preset="none",
        supported_presets=("none",),
    ),
    "email_campaign": OutputReasoningPolicy(
        output="email_campaign",
        default_preset="single_pass",
        supported_presets=_MULTI_PASS_OPTIONAL,
    ),
    "landing_page": OutputReasoningPolicy(
        output="landing_page",
        default_preset="single_pass",
        supported_presets=_LANDING_PAGE_PRESETS,
    ),
    "blog_post": OutputReasoningPolicy(
        output="blog_post",
        default_preset="multi_pass_light",
        supported_presets=_STRUCTURED_PRESETS,
    ),
    "report": OutputReasoningPolicy(
        output="report",
        default_preset="multi_pass_structured",
        supported_presets=_STRUCTURED_PRESETS,
    ),
    "sales_brief": OutputReasoningPolicy(
        output="sales_brief",
        default_preset="multi_pass_structured",
        supported_presets=_STRUCTURED_PRESETS,
    ),
})


def reasoning_preset_definition(preset: str) -> ReasoningPresetDefinition:
    try:
        return REASONING_PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"unknown reasoning preset: {preset}") from exc


def output_reasoning_policy(output: str) -> OutputReasoningPolicy:
    try:
        return OUTPUT_REASONING_POLICIES[output]
    except KeyError as exc:
        raise ValueError(f"unknown content output: {output}") from exc


def supported_reasoning_presets(output: str) -> tuple[ReasoningPreset, ...]:
    return output_reasoning_policy(output).supported_presets


def resolve_reasoning_policy(
    output: str,
    preset: str | None = None,
) -> tuple[OutputReasoningPolicy, ReasoningPresetDefinition]:
    """Resolve output policy and preset; None or blank preset uses the default."""
    policy = output_reasoning_policy(output)
    selected = preset if preset else policy.default_preset
    definition = reasoning_preset_definition(selected)
    if not policy.supports(definition.id):
        raise ValueError(
            f"reasoning preset {definition.id!r} is not supported for output {output!r}"
        )
    return policy, definition
