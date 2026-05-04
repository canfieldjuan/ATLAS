"""Reasoning pack registry (PR 5 from the reasoning boundary audit).

Packs are product-specific prompt/policy bundles that products register
with the shared reasoning core. The audit's `Core Versus Pack Decision
Rule` says a module belongs in a pack when it names a specific artifact
type (battle card, vendor briefing, campaign), encodes product-specific
narrative structure, or contains output schema for buyer-facing
surfaces. Examples called out in the audit:

  - battle card reasoning  (competitive intelligence)
  - cross-vendor battle    (competitive intelligence)
  - vendor classify        (atlas / competitive intelligence)
  - reasoning synthesis    (atlas)
  - content/campaign       (content pipeline)

This module provides the *registry* surface only. Concrete packs are
registered by their owning products in subsequent slices (PR-C3b onward
per the PR-C3 sub-sequence). Acceptance criteria from the audit:

  1. **Packs are explicit dependencies.** Products import a pack and
     register it; core never reaches into a product to find one.
  2. **Core can run without importing a product pack.** The registry
     starts empty and degrades gracefully (callers see ``None`` /
     ``[]`` for unregistered names).
  3. **Products select packs by name/version.** ``get_pack`` accepts
     an optional ``version`` filter; ``list_packs`` returns the
     full set so callers can pick the latest matching version.

Threading note: registration is process-wide and not thread-safe. The
expected pattern is at-import-time registration (a product imports its
pack module, which registers itself once). Callers that need runtime
mutation should serialise externally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Pack:
    """Immutable description of a reasoning pack.

    Fields:

      - ``name`` -- unique pack identifier (lowercase snake_case
        recommended; e.g. ``"battle_card_reasoning"``).
      - ``version`` -- semver string (``"1.0.0"``). Multiple versions
        of the same pack name can coexist in the registry; callers
        select via ``get_pack(name, version=...)``.
      - ``prompts`` -- mapping of prompt-id -> prompt text/template.
        Concrete shape is product-defined; the registry doesn't
        validate prompt content.
      - ``metadata`` -- free-form bag for output-schema references,
        policy flags, owner annotations, etc. Frozen at construction.
    """

    name: str
    version: str
    prompts: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


# Module-level registry. Keyed by ``(name, version)`` so multiple
# versions of the same pack can coexist. Reset via ``clear_packs()``
# (test-only entry point).
_PACKS: dict[tuple[str, str], Pack] = {}


def register_pack(pack: Pack) -> None:
    """Register a pack instance.

    Idempotent for identical re-registration (same name+version+content).
    Raises ``ValueError`` if a different ``Pack`` is already registered
    under the same ``(name, version)`` key -- conflicting packs are a
    bug in the registering product.
    """
    key = (pack.name, pack.version)
    existing = _PACKS.get(key)
    if existing is not None and existing != pack:
        raise ValueError(
            f"Pack {pack.name!r} version {pack.version!r} is already registered "
            f"with different content; refusing to overwrite."
        )
    _PACKS[key] = pack


def get_pack(name: str, *, version: str | None = None) -> Pack | None:
    """Return a registered pack by name (and optional version).

    When ``version`` is omitted, returns the highest-versioned pack
    with that name (lexicographic comparison on the version string;
    callers using non-semver schemes should pass ``version`` explicitly).

    Returns ``None`` when no matching pack is registered -- core never
    raises on unknown names so it can run without any pack registered.
    """
    if version is not None:
        return _PACKS.get((name, version))

    candidates = [pack for (pname, _), pack in _PACKS.items() if pname == name]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.version)


def list_packs() -> list[Pack]:
    """Return all registered packs, ordered by name then version.

    Used by products to discover what packs are available without
    knowing names in advance, and by tests to assert registry state.
    """
    return sorted(_PACKS.values(), key=lambda p: (p.name, p.version))


def clear_packs() -> None:
    """Reset the registry. Test-only.

    Production code should not call this -- packs are registered at
    import time and stay for the process lifetime. Tests that register
    packs should call ``clear_packs()`` in setup/teardown to avoid
    cross-test bleed.
    """
    _PACKS.clear()


__all__ = [
    "Pack",
    "clear_packs",
    "get_pack",
    "list_packs",
    "register_pack",
]
