# Content Ops Reasoning Provider Status

## Why this slice exists

After the DB-backed Content Ops reasoning provider landed, the control-surface
catalog still reported `reasoning.configured` from route wiring only. Atlas
passes a chooser callback at mount time, so the catalog could report
configured even when the chooser resolves to `None` because DB/file providers
are not actually available. Operators need readiness to reflect the provider
that can currently be built.

## Scope (this PR)

1. Add an optional reasoning status provider to the extracted control-surface
   router.
2. Add an Atlas host status descriptor that reports DB, file, or none using
   the same DB > file > none priority as execute-time provider selection.
3. Wire the Atlas API mount to expose that status in
   `GET /content-ops/control-surfaces`.
4. Update frontend contract docs and coordination state.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `atlas_brain/_content_ops_reasoning.py`
- `atlas_brain/api/__init__.py`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_atlas_content_ops_reasoning.py`
- `docs/frontend/content_ops_frontend_contract.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The extracted router accepts a new optional `reasoning_status_provider`.
Catalog requests resolve it independently from execution. When it is absent,
the prior response shape is preserved:

```python
{"configured": reasoning_context_provider is not None}
```

Atlas wires `describe_content_ops_reasoning_context_provider()`, which probes
the same DB and file factories used by
`select_content_ops_reasoning_context_provider()` and returns:

```python
{"configured": True, "source": "db"}
{"configured": True, "source": "file"}
{"configured": False, "source": "none"}
```

The execute path is unchanged.

## Intentional

- The status provider is optional so standalone hosts that only pass
  `reasoning_context_provider` keep the existing API contract.
- The catalog status provider fails open to the old boolean fallback instead
  of returning a 503. Catalog readiness should degrade gracefully; execute-time
  failures still use the existing 503 path.
- `source` is catalog metadata only. The execution bundle still receives the
  provider object chosen per request.

## Deferred

- No frontend UI change in this slice. The contract doc now describes
  `reasoning.source`; the UI can render it in a follow-up if desired.
- No live DB smoke change. PR #471 already owns the live-DSN provider lookup
  check.

## Verification

- `pytest tests/test_extracted_content_control_surface_api.py tests/test_atlas_content_ops_reasoning.py` -> 43 passed
- `python -m py_compile extracted_content_pipeline/api/control_surfaces.py atlas_brain/_content_ops_reasoning.py atlas_brain/api/__init__.py tests/test_extracted_content_control_surface_api.py tests/test_atlas_content_ops_reasoning.py` -> passed
- `git diff --check` -> passed
- ASCII byte check on edited Python/test files -> passed
- `bash scripts/run_extracted_pipeline_checks.sh` -> 1449 passed, 1 existing torch/pynvml warning

## Estimated diff size

8 files, about +220 / -15 including this plan and docs.
