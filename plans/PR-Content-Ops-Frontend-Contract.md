# PR: rewrite Content Ops frontend contract against real code

## Why this slice exists

A prior frontend-contract draft (commit `61f1003` on
`claude/content-ops-frontend-contract`) declared the AI Content Ops
backend "hallucinated." That verdict was wrong: the verification
sweep ran against a stale `origin/main` snapshot (`332bb26`, 11 PRs
behind). The actual main has PRs **#389–#398** which shipped the
entire `/content-ops/*` surface:

- Routes: `GET /content-ops/control-surfaces`, `POST /content-ops/preview`,
  `POST /content-ops/plan`, `POST /content-ops/execute` —
  `extracted_content_pipeline/api/control_surfaces.py:212-272`
- Types: `OutputDefinition`, `ControlSurfacePreset`, `ContentOpsRequest`,
  `ControlSurfacePreview` — `extracted_content_pipeline/control_surfaces.py:16-90`
- Plan types: `GenerationPlan`, `GenerationPlanStep` —
  `extracted_content_pipeline/generation_plan.py:28-65`
- Execution types: `ContentOpsExecutionServices`, `ContentOpsStepExecution`,
  `ContentOpsExecutionResult` —
  `extracted_content_pipeline/content_ops_execution.py:16-101`
- Signal extraction: `SignalExtractionConfig`, `SignalExtractionResult`,
  `SignalExtractionService` —
  `extracted_content_pipeline/signal_extraction.py:19-81`

Building a frontend against the prior doc would produce a domain
model that's wrong end-to-end. This slice replaces the doc with one
that's grounded in the actual code, with file:line citations on every
type and field.

This is also a concrete failure-mode worth naming for AGENTS.md §4a
("Don't trust claims; reproduce them"): the prior doc trusted its own
local grep against a stale tree, then over-confidently declared the
spec hallucinated. Re-running the same grep against fresh `origin/main`
flips every "fabricated" finding to "verified."

## Scope (this PR)

Documentation only. Two files.

### Files touched

1. `docs/frontend/content_ops_frontend_contract.md` (rewritten in
   place — supersedes the prior version's content entirely).
2. `plans/PR-Content-Ops-Frontend-Contract.md` (this file).

## Mechanism

Read the real source modules at HEAD-of-main (`a4020c1`):

- `extracted_content_pipeline/control_surfaces.py` — pure-function
  preview engine; `OutputDefinition`, `ControlSurfacePreset`,
  `ContentOpsRequest`, `ControlSurfacePreview`, `OUTPUT_CATALOG` (6
  outputs), `PRESETS` (5 presets), `request_from_mapping`,
  `preview_control_surface`, `preview_from_mapping`.
- `extracted_content_pipeline/generation_plan.py` —
  `GenerationPlanStep`, `GenerationPlan`, per-output config builders,
  `build_generation_plan`, `build_generation_plan_from_mapping`.
- `extracted_content_pipeline/content_ops_execution.py` —
  `ContentOpsExecutionServices` (host-port bundle for 6 generators),
  `ContentOpsStepExecution`, `ContentOpsExecutionResult`,
  `execute_content_ops_request`, `execute_content_ops_from_mapping`.
- `extracted_content_pipeline/signal_extraction.py` —
  `SignalExtractionConfig`, `SignalExtractionResult`,
  `SignalExtractionService`.
- `extracted_content_pipeline/api/control_surfaces.py` —
  `ContentOpsControlSurfaceApiConfig`, `ContentOpsRequestModel`
  (pydantic body validator), `create_content_ops_control_surface_router`,
  `_compose_describe_response` (catalog response composer).

Map every type and route to its file:line, then express it as a
TypeScript-ish frontend domain model. The frontend adapter layer
becomes a thin pass-through (snake_case ↔ camelCase translation),
not a re-imagination of the backend.

## Intentional

- **Reverses the prior doc's stance.** The earlier doc called the
  proposal "hallucinated"; that was wrong. New doc opens with a
  direct retraction citing the verification failure mode (stale-tree
  grep) so the failure pattern is named, not buried.
- **Branched fresh from `origin/main` per AGENTS.md naming.** The
  prior `claude/content-ops-frontend-contract` branch is left
  untouched; the user can delete it. Trying to amend the wrong
  branch in place would muddle PR history.
- **TypeScript-shaped interfaces, not concrete TS code.** No build
  setup yet; the doc is a contract, not implementation.
- **Single doc, not split into per-screen / per-axis files.** The
  whole thing fits in ~500 LOC; splitting would add navigation
  overhead with no clarity gain at this size.
- **No frontend code in this slice.** The contract precedes the
  scaffold; mixing them defeats the point of a contract.

## Deferred

- Concrete TypeScript type generation (e.g. via `openapi-codegen`
  or hand-written `src/api/contentOps.ts`). Lands when the frontend
  repo is scaffolded.
- MVP screen implementations. Out of scope for the contract slice.
- A frontend tests-against-fixtures harness. Lands with code.
- Closing/deleting `claude/content-ops-frontend-contract` (the
  earlier wrong branch). User-facing decision; this slice doesn't
  touch other branches.
- Documenting the hallucination-retraction pattern in `AGENTS.md`
  itself. Could be a one-line addition under §4a; defer to a
  follow-up if the pattern recurs.

## Verification

- `git -C . log -1 origin/main --format=%H` → `a4020c1` (the HEAD
  this slice's citations are valid against).
- `grep -n "@router\." extracted_content_pipeline/api/control_surfaces.py`
  → 4 routes (`/control-surfaces`, `/preview`, `/plan`, `/execute`)
  at lines 212, 226, 232, 241.
- `grep -n "@dataclass" extracted_content_pipeline/control_surfaces.py
  extracted_content_pipeline/generation_plan.py
  extracted_content_pipeline/content_ops_execution.py
  extracted_content_pipeline/signal_extraction.py` → 9 frozen
  dataclasses; every one cited in the doc.
- Every `path:line` citation in `docs/frontend/content_ops_frontend_contract.md`
  resolves on HEAD-of-main (`a4020c1`).
- `bash scripts/check_ascii_python.sh` — clean (markdown not in scope).
- No Python touched; `bash scripts/validate_extracted_content_pipeline.sh`
  not required (no `extracted_content_pipeline/**.py` modified).
- `git diff main --stat` — 2 files, all additions.

## Estimated diff size

- `plans/PR-Content-Ops-Frontend-Contract.md`: ~140 LOC.
- `docs/frontend/content_ops_frontend_contract.md`: ~500 LOC
  (replaces equivalent-size doc on the prior wrong branch).

Total: ~640 LOC. **Over the soft 400-LOC budget.** Justification per
AGENTS.md §1d: the prior 550-LOC doc was wrong end-to-end and cannot
be salvaged with small edits — every claim in its "hallucinations"
table inverts when verified against current main. Splitting
"retraction" from "rewrite" leaves the wrong doc in place between
PRs, so the slice is indivisible. All documentation; no code.
