# PR-Content-Factory-Artifact-Store

## Why this slice exists

The Content Factory pipeline produces one JSON artifact per stage (brief,
evidence, draft, audit, manifest), whose shapes were contract-validated in #2116.
But nothing persists those artifacts as a durable, inspectable record -- the
Phase 1.4 end-to-end run wrote them with an ad-hoc scratchpad script that did no
contract validation. This slice adds the real persistence behavior: validate a
stage output against its content_factory contract and write it to a git-tracked
job folder, so every stage output is durable, auditable, and provably well-formed
before it lands.

### Problem-derived contract

A correct fix must:
- Validate a stage artifact against its content_factory contract BEFORE writing,
  so a malformed output is never persisted (fail closed).
- Persist the canonical form (the "schema" version key, not the attribute name)
  to a stable, per-job location.
- Guard the filesystem path against traversal: job_id and stage are caller-
  supplied strings that become path segments.
- Version each write with git so the job folder is an auditable trail.
- Live in atlas_brain (where the contracts are), not in an Open WebUI function
  (OWUI runs in a separate Python env and cannot import atlas_brain).
- Add no runtime wiring beyond the service and its tests.

## Scope (this PR)

Ownership lane: content-factory
Slice phase: vertical slice

Arc Phase 2.2a: the artifact-store service (behavior). The thin OWUI Action / API
caller that invokes it is a separate later slice (2.2b).

### Review Contract

- Acceptance criteria:
  - [ ] A valid stage artifact is validated against its contract, written to the
        per-job stage file in canonical form, and git-committed.
  - [ ] A contract-invalid artifact (e.g. blank evidence source_id) or one with
        no schema tag raises before any file or folder is created.
  - [ ] job_id and stage are path-traversal guarded: dot-dot, slash, leading
        dot, and empty are rejected on both; single dots and dashes are accepted.
  - [ ] Re-writing byte-identical content makes no empty commit; changed content
        makes a new commit.
- Reachability proof: N/A for a production surface -- the service has no runtime
  caller yet (the OWUI/API caller is slice 2.2b). Proof is the test suite against
  a real temp filesystem and real git.
- Affected surfaces: one new service module and its test file; no existing file
  modified; nothing imports the module yet.
- Risk areas: path traversal (the choke point is boundary-probed on both sides);
  git subprocess failure (wrapped in ArtifactStoreError); the empty-commit edge.
- Reviewer rules triggered: R2, R14.

### Files touched
- `atlas_brain/services/content_factory_store.py`
- `tests/test_content_factory_store.py`
- `plans/PR-Content-Factory-Artifact-Store.md`

Max files: 3

## Mechanism

write_artifact(job_id, stage, artifact, root=...) runs both path segments through
a single _safe_segment choke point (an anchored [A-Za-z0-9._-] pattern that also
rejects any dot-dot), validates the artifact with model_for + model_validate from
#2116 (raising before any filesystem use), dumps the canonical form (serialize_by
_alias emits the schema key), writes the per-job stage file, and commits it to the
job folder's own git repo -- skipping an empty commit on identical content. A
missing git binary and any git failure are wrapped in ArtifactStoreError. The root
defaults to a home-directory folder and is a parameter so tests use a temp dir.

## Intentional

- Behavior in atlas_brain, not an OWUI function: OWUI's snap Python env cannot
  import atlas_brain, so putting validation there would vendor and drift the
  contracts. The service is the source of truth; a thin caller comes later.
- root is a parameter (default home-dir/content-factory) so tests use a temp dir
  and callers can override; no os.environ read, no config.py surface this slice.
- Real git and real filesystem in tests (no mocks): the store's whole job is to
  persist and version real files, so mocking them would test nothing.

## Deferred

- The thin OWUI Action function / Atlas API endpoint that calls this service
  (arc Phase 2.2b) -- that is where the OWUI-env boundary is crossed.
- A typed config field for the root (add when a runtime caller needs it).
- read_artifact / manifest-assembly helpers (add when a consumer needs them).
- The #2120 contract edge-hardening is independent and tracked there.

## Verification

```
python -m pytest tests/test_content_factory_store.py -q
```
21 tests pass against a real temp filesystem and real git: valid write persists,
commits, and returns the sha; the canonical schema key is on disk; contract-invalid
and tagless artifacts raise before any write (no folder created); path-traversal
job_id and stage are rejected on both sides; valid dotted/dashed segments accepted;
a second stage adds a commit; an identical re-write makes no empty commit; a changed
re-write commits.

## Estimated diff size

| File | Lines |
|---|---|
| atlas_brain/services/content_factory_store.py | 163 |
| tests/test_content_factory_store.py | 167 |
| plans/PR-Content-Factory-Artifact-Store.md | 116 |
| **Total** | **446** |

Slightly over the 400 soft cap after the Codex-review hardening (stage/schema
map, reserved-key rejection, fullmatch guard, scoped commit) and their tests; the
overage is a cohesive service + its tests, indivisible without shipping the guard
without its proof. Carried by the `Diff-budget override:` line in the PR body.
