"""Competitive Intelligence reasoning package.

Submodule imports are handled by Python's native import machinery from
the scaffold filesystem -- importing
``extracted_competitive_intelligence.reasoning.semantic_cache`` /
``...wedge_registry`` / ``...single_pass_prompts.cross_vendor_battle``
resolves directly to those files.

PR-D7a removed the Phase-1 ``__getattr__`` bridge that previously
lazy-delegated to ``atlas_brain.reasoning`` for non-submodule
attribute access. Nothing inside the codebase actually triggered that
hook (every consumer reaches into a concrete submodule), and the
post-PR-C4 reasoning core now provides everything reasoning-side,
so the bridge was dead code that also blocked the audit's
"no runtime atlas_brain.reasoning imports" acceptance criterion.
Submodule access is unaffected.
"""
from __future__ import annotations
