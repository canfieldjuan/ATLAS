"""Host wrapper for the B2B blog-post autonomous task.

Wires the per-blueprint fanout (PR #461,
``atlas_brain/_blog_blueprint_fanout.py``) into the autonomous
task's ``set_blueprint_published_hook`` injection seam, then
re-exports ``run`` so the task registry
(``atlas_brain/autonomous/tasks/__init__.py``) can register it
under the existing ``b2b_blog_post_generation`` task name.

This module is host-only -- it lives at
``atlas_brain/autonomous/tasks/`` and is NOT mirrored into
``extracted_content_pipeline/``. The autonomous task itself
(``b2b_blog_post_generation.py``) stays portable: it imports
no host-only fanout chain, only invokes the optional
``_blueprint_published_hook`` callback when one is installed.

Dual-resident discipline:

* ``b2b_blog_post_generation.py`` is mapped in
  ``extracted_content_pipeline/manifest.json``; both copies
  must stay byte-identical (the validate gauntlet enforces it).
* ``_b2b_blog_post_generation_host.py`` (this file) and
  ``_blog_blueprint_fanout.py`` are NOT mapped -- they live
  only in atlas_brain. The extracted package's standalone
  smoke runs without them and the autonomous task remains
  importable in the dependency-light extracted env.

Background: PR #461 originally placed the fanout calls
inline inside ``b2b_blog_post_generation.py`` and added a
``from ..._blog_blueprint_fanout import fanout_blueprint``
to that file. That broke standalone import because the
fanout module + its ``_blog_post_subscriptions`` dependency
exist only in atlas_brain. The validate gauntlet didn't catch
the drift on PR #461 because no PR between #461 and PR #463
touched ``extracted_content_pipeline/`` so the workflow's
``paths:`` filter never fired. PR #464 lifts the host-only
chain into this wrapper so the source-of-truth file is
extraction-safe again.
"""

from __future__ import annotations

from .b2b_blog_post_generation import (
    run,
    set_blueprint_published_hook,
)
from .._blog_blueprint_fanout import fanout_blueprint as _fanout_blog_blueprint


async def _publish_hook(pool, blueprint) -> None:
    """Per-blueprint fanout adapter wired into the autonomous task.

    The autonomous task swallows hook exceptions (``logger.warning``
    on any raise) so this adapter intentionally lets exceptions
    bubble up -- the task layer is the one that decides whether a
    fanout failure should be logged or escalated.
    """

    fanout_count = await _fanout_blog_blueprint(pool, blueprint)
    if fanout_count:
        # The task-side log already records "Blog blueprint published
        # hook" outcomes; surface the count here for observability
        # without duplicating the warning-on-exception path.
        import logging

        logging.getLogger(
            "atlas.autonomous.tasks._b2b_blog_post_generation_host"
        ).info(
            "Blog blueprint fanned out: slug=%s subscribers=%d",
            getattr(blueprint, "slug", "?"),
            fanout_count,
        )


# Install the hook at module import time. The autonomous tasks
# registry (``atlas_brain/autonomous/tasks/__init__.py``) imports
# this module, which side-effect-binds the hook before the
# registry calls ``register_builtin``. Idempotent: re-imports
# overwrite the same callable.
set_blueprint_published_hook(_publish_hook)


__all__ = ["run"]
