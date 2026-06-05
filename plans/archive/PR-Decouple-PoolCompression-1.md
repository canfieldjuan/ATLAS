# PR-Decouple-PoolCompression-1: standalone shim for `_b2b_witnesses`

## Why this slice exists

Follow-up to PR-Decouple-CompIntel-1 (#422). After that PR landed, a
sweep of all 114 manifest-tracked Python files across both extracted
packages found **exactly one** remaining standalone-import failure:

```
$ EXTRACTED_PIPELINE_STANDALONE=1 python3 -c \
    "from extracted_content_pipeline.autonomous.tasks import _b2b_pool_compression"
ModuleNotFoundError: No module named
  'extracted_content_pipeline.autonomous.tasks._b2b_witnesses'
```

`_b2b_pool_compression.py` (a manifest-tracked mirror) imports one
symbol from a sibling module that is not mirrored:

```python
from ._b2b_witnesses import build_vendor_witness_artifacts
```

`atlas_brain/autonomous/tasks/_b2b_witnesses.py` exists (1,325 LOC) but
is absent from `extracted_content_pipeline/`. Same shape as the
`brand_registry` failure closed by #422.

## Scope (this PR)

One new file: `extracted_content_pipeline/autonomous/tasks/_b2b_witnesses.py`.

Hand-rolled standalone substrate matching the precedent from #422.
Exports a single function with the matching signature, returns a
tuple `([], {})` -- the empty witness pack and section packets.

Plus a regression test pinning that
`_b2b_pool_compression` imports cleanly under
`EXTRACTED_PIPELINE_STANDALONE=1`.

## Standalone contract

Same as #422: import-only. The mirror copy is never imported by the
host -- atlas_brain consumes `atlas_brain.autonomous.tasks._b2b_witnesses`
directly in production. The substrate here exists so the mirror copy
of `_b2b_pool_compression` can be loaded in standalone mode without
reaching into atlas_brain.

`build_vendor_witness_artifacts` returning `([], {})` means downstream
callers get empty witness data; that is consistent with the rest of
the standalone surface (`pipelines/llm.py` returning `None`,
`storage/database.py` returning a fake pool that fails on first
real call).

## File shape

```python
# extracted_content_pipeline/autonomous/tasks/_b2b_witnesses.py
"""Standalone substrate for _b2b_witnesses ...

Real implementation lives in atlas_brain. This module exists so the
mirrored _b2b_pool_compression can be imported under
EXTRACTED_PIPELINE_STANDALONE=1 without reaching into atlas_brain.
"""
from __future__ import annotations
from typing import Any


def build_vendor_witness_artifacts(
    vendor_name: str,
    reviews: list[dict[str, Any]] | None,
    **_kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [], {}
```

`**_kwargs` absorbs the eight keyword args atlas_brain's real
implementation accepts (`max_witnesses`, `min_specificity_score`,
`fallback_min_witnesses`, `generic_patterns`, `concrete_patterns`,
`short_excerpt_chars`, `long_excerpt_chars`, `specificity_weights`)
without re-declaring them. The standalone body ignores all of them
because it never actually computes witnesses.

## Intentional (looks wrong but is deliberate)

- **`**_kwargs` instead of mirrored signature.** The real function's
  signature is wide (8 keyword params with defaults). Re-declaring
  them in the standalone shim duplicates surface area for no
  behavioral payoff -- under standalone they all get ignored. Match
  the narrow-export precedent from #422 (`brand_registry` exposed
  only the 2 symbols the consumer imports).
- **No env-var gate.** Same rationale as #422: there is no extracted-
  side real implementation to delegate to in the non-standalone
  branch, so a gate would be decorative.
- **Not added to `manifest.json`.** Matches the precedent of the
  other untracked standalone substrates (`pipelines/llm.py`,
  `storage/database.py`, `services/brand_registry.py` from #422).
  Closing the methodology gap (untracked shims belong on the `owned`
  list) is a separate slice.
- **`return [], {}` instead of raising or stubbing structured data.**
  `_b2b_pool_compression` calls
  `witness_pack, section_packets = build_vendor_witness_artifacts(...)`.
  Empty list and empty dict cause the downstream code to produce a
  result with no witness data; no AttributeError or KeyError. Standalone
  callers that go further than import will get a no-op result rather
  than a crash.

## Deferred (looks missing but is on purpose)

- **End-to-end standalone runnability for `_b2b_pool_compression`.**
  Same caveat as #422: the task's database calls remain unserviceable
  under the fake `_StandalonePool`.
- **Mirror `_b2b_grounding.py` and `_b2b_phrase_metadata.py`.** Atlas-
  brain's real `_b2b_witnesses.py` imports from both; the shim here
  doesn't, so they don't need to be mirrored to satisfy this slice.
  If a future module mirrored into `extracted_content_pipeline`
  imports them directly, that's the next slice -- not this one.
- **Closing the manifest methodology gap.** Same as #422.

## Verification

- New test:
  `tests/test_extracted_pool_compression_standalone.py` asserting
  the import succeeds under standalone and that the shim returns
  `([], {})` regardless of inputs.
- `EXTRACTED_PIPELINE_STANDALONE=1 python3 -c "from
  extracted_content_pipeline.autonomous.tasks import
  _b2b_pool_compression"` -> exits 0, no traceback.
- `python3 scripts/audit_extracted_standalone.py --fail-on-debt` ->
  Atlas runtime import findings: 0 (unchanged).
- `bash scripts/check_ascii_python.sh` -> passed.
- Repeat sweep across both packages (114 manifest-tracked Python
  files): 0 standalone-import failures.

## Conflict check

- No file overlap with any open PR. This slice touches a single new
  file in `extracted_content_pipeline/autonomous/tasks/`; the
  in-flight `extracted_competitive_intelligence/autonomous/visibility.py`
  bridge promotion is in a different package.

## Diff size

1 source file (~25 LOC), 1 new test file (~50 LOC), 1 plan doc (this
file, ~110 LOC). Source-only ~25 LOC -- the smallest possible slice
that closes the failure.

## After this lands

Sweep across 114 manifest-tracked Python files reports **0 standalone-
import failures**. The decoupling track for `extracted_content_pipeline`
and `extracted_competitive_intelligence` is fully green at the
standalone-toggle level. Whatever next: monolith decomposition,
manifest methodology cleanup, or product track.
