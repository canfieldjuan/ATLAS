# PR-Decouple-CompIntel-1: standalone shim for `services/brand_registry`

## Why this slice exists

`extracted_content_pipeline/autonomous/tasks/competitive_intelligence.py`
is a manifest-tracked byte-identical mirror of
`atlas_brain/autonomous/tasks/competitive_intelligence.py` (1,455 LOC, one
of three remaining monoliths in the extraction tracker). Today the mirror
fails to import under any code path that exercises the extracted package:

```
$ EXTRACTED_PIPELINE_STANDALONE=1 python3 -c \
    "from extracted_content_pipeline.autonomous.tasks import competitive_intelligence"
ModuleNotFoundError: No module named
  'extracted_content_pipeline.services.brand_registry'
```

The task imports `from ...services.brand_registry import
resolve_brand_name_cached, _ensure_cache as _ensure_brand_cache`. Under
the relative-import resolution, `...services.brand_registry` becomes
`extracted_content_pipeline.services.brand_registry` -- which does not
exist. Four of the task's five atlas_brain dependencies have already
been mirrored or shimmed (`config.py`, `storage/database.py`,
`storage/models.py`, `pipelines/llm.py`); `services/brand_registry.py`
is the missing fifth.

`scripts/audit_extracted_standalone.py` reports `0` Atlas runtime import
findings today, so the audit does not catch this -- it scans for
`from atlas_brain.*` imports, not for missing relative-import targets.

## Scope (this PR)

One new file: `extracted_content_pipeline/services/brand_registry.py`.

That file is a standalone substrate -- a minimal hand-rolled shim that
satisfies import time and provides no-op runtime behavior, matching the
existing precedent set by `extracted_content_pipeline/pipelines/llm.py`
and `extracted_content_pipeline/storage/database.py`.

Plus a regression test verifying
`extracted_content_pipeline.autonomous.tasks.competitive_intelligence`
imports cleanly under `EXTRACTED_PIPELINE_STANDALONE=1`.

## Standalone contract

The contract for the extracted package's standalone toggle is
**"compile and import,"** not **"run end-to-end without infrastructure."**
Three precedents:

- `extracted_content_pipeline/pipelines/llm.py` -- `call_llm_with_skill`
  returns `None` under standalone, real provider otherwise.
- `extracted_content_pipeline/storage/database.py` -- returns a fake
  `_StandalonePool` whose `is_initialized=True` is the only honest claim
  it makes; any `await pool.fetchrow(...)` would raise.
- `extracted_content_pipeline/storage/models.py` -- minimal
  `ScheduledTask` dataclass without the rich field set the atlas_brain
  original has.

This PR follows the same shape: `resolve_brand_name_cached(name)` returns
the input unchanged (identity passthrough -- no canonicalization);
`_ensure_cache()` is an async no-op. That makes
`competitive_intelligence.py` importable. Whether the task actually runs
end-to-end under standalone is a separate question -- it has 24 real
DB calls against 11 tables, which the fake `_StandalonePool` cannot
service. Closing that gap is out of scope here.

## File shape

```python
# extracted_content_pipeline/services/brand_registry.py
"""Standalone substrate for the brand registry service.

Real implementation lives in atlas_brain; this module exists only so
that mirrored modules importing ``brand_registry`` (e.g.,
``competitive_intelligence``) can be imported under
``EXTRACTED_PIPELINE_STANDALONE=1`` without touching atlas_brain.

Runtime behavior is no-op: ``resolve_brand_name_cached`` returns the
input unchanged; ``_ensure_cache`` is a no-op coroutine. Tasks that
need real brand canonicalization should run in atlas_brain mode.
"""
from __future__ import annotations

from typing import Any


def resolve_brand_name_cached(name: Any) -> Any:
    return name


async def _ensure_cache() -> None:
    return None
```

No env-var gate. Unlike `pipelines/llm.py`, there is no extracted-side
real implementation to delegate to (no `extracted_brand_registry`
package), so both branches would be identical stubs -- the gate would
be decorative. Keep the file flat.

## Intentional (looks wrong but is deliberate)

- **Always identity passthrough, no env-var gate.** `pipelines/llm.py`'s
  env-var gate exists because the non-standalone branch delegates to
  `extracted_llm_infrastructure`. There is no equivalent extracted
  package for brand registry, so the gate would have nothing to switch
  on. Atlas_brain consumes
  `atlas_brain.services.brand_registry` directly in production -- the
  mirror copy here is never imported by the host.
- **Not added to `manifest.json`.** The other untracked standalone
  shims (`pipelines/llm.py`, `storage/database.py`,
  `storage/models.py`, `config.py`) are also absent from the manifest.
  This PR matches that precedent rather than introducing a new
  bookkeeping shape mid-slice. Closing the methodology gap of
  untracked standalone shims is a separate concern.
- **Stub signature deliberately narrow.** Only the two symbols
  `competitive_intelligence.py` actually imports
  (`resolve_brand_name_cached`, `_ensure_cache`) are exported. The
  atlas_brain original exports more (e.g., cache mutation helpers).
  Adding those without a consumer would be plumbing for nothing.
- **`Any` typing on the passthrough.** atlas_brain's
  `resolve_brand_name_cached(name: str) -> str` is more strictly typed,
  but the standalone path has no canonicalization to perform; permitting
  whatever caller passes through avoids spurious narrowing errors and
  matches the "no-op" contract.

## Deferred (looks missing but is on purpose)

- **End-to-end standalone runnability for `competitive_intelligence`.**
  Requires mirroring the 11 tables it touches as
  `extracted_content_pipeline/storage/migrations/*.sql` (some already
  present, some not), plus an asyncpg pool factory, plus the
  ``digest/competitive_intelligence`` skill registration on the
  standalone side. Multi-PR lift, separate scope from "import passes."
- **Same fix for the other two monoliths
  (`b2b_vendor_briefing.py`, `b2b_campaign_generation.py`).** Both have
  their own dependency footprints; closing this slice first proves out
  the substrate-shim recipe on the smallest of the three.
- **Closing the manifest methodology gap.** Untracked standalone shims
  (`pipelines/llm.py` etc.) should arguably be on the manifest's
  ``owned`` list. Worth a separate PR; folding it into this slice
  would expand scope into a methodology change.

## Verification

- `EXTRACTED_PIPELINE_STANDALONE=1 python3 -c "from
  extracted_content_pipeline.autonomous.tasks import
  competitive_intelligence"` -> exits 0, no traceback.
- New regression test:
  `tests/test_extracted_competitive_intelligence_standalone.py`
  asserting the import succeeds and that `resolve_brand_name_cached`
  passes its input through unchanged under the standalone shim.
- `python3 scripts/audit_extracted_standalone.py --fail-on-debt` ->
  Atlas runtime import findings: 0 (unchanged).
- `bash scripts/check_ascii_python.sh` -> passed.

## Conflict check

- **No file overlap.** The in-flight `extracted_competitive_intelligence/
  autonomous/visibility.py` bridge promotion is in a different package
  (`extracted_competitive_intelligence`, not `extracted_content_pipeline`),
  different manifest, different migrations dir.
- **Pattern alignment.** The in-flight pioneers the bridge-to-manifest-
  synced-mirror pattern in `extracted_competitive_intelligence`. This
  PR follows the existing standalone-shim pattern in
  `extracted_content_pipeline`. Both packages already use both
  patterns; this slice does not introduce a new methodology.
- Single open PR (#418, Content Ops reasoning consumption summary)
  touches an unrelated surface.

## Diff size

1 source file (~25 LOC), 1 new test file (~30 LOC), 1 plan doc (this
file, ~140 LOC). Source-only ~25 LOC -- the smallest possible slice
that closes the failure.
