# PR-Audit-Extracted-Manifests

## Why this slice exists

Oversized PR #484 bundled three unrelated Tier-1 auditors and was
rejected as unreviewable. This split lands only the extracted-package
manifest consistency auditor because it directly protects product
extraction work from silent source/target drift.

## Scope (this PR)

1. Add `scripts/audit_extracted_manifests.py`.
2. Add focused fixture tests for the path-safety validator.
3. Add `tests/audit_helpers.py` as a small shared loader for audit-script
   tests.
4. Refresh the in-flight coordination row for this split.

### Files touched

- `plans/PR-Audit-Extracted-Manifests.md`
- `scripts/audit_extracted_manifests.py`
- `tests/audit_helpers.py`
- `tests/test_audit_extracted_manifests.py`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The auditor walks `extracted_*/manifest.json` files and checks:

- every `mappings[].source` exists under `atlas_brain/`
- every `mappings[].target` exists under the extracted package
- every `owned[].target` exists under the extracted package
- mapped source/target byte content is identical

Before touching the filesystem, `_validate_path()` rejects absolute
paths, parent traversal, and paths outside the expected tree. It checks
POSIX, Windows drive-letter, and UNC absolute paths so malformed
manifests cannot escape the repo subtree on non-POSIX platforms.

## Intentional

- The auditor is standalone and not wired into `scripts/pre_push_audit.sh`
  yet. Main currently has known extracted sync drift, so wrapper
  integration would make every pre-push run fail until that drift is
  reconciled.
- `tests/audit_helpers.py` raises on missing auditors instead of skipping.
  This split lands the script and tests together, so dormant skip tests are
  not needed.
- The tests target the path validator directly. Full live-repo execution is
  covered by running the script itself.

## Deferred

- Add the auditor to the pre-push wrapper after known sync drift is cleaned
  up or explicitly accepted.
- Split the remaining #484 auditors separately: MCP tool-name inventory and
  review-source count.
- Close or replace the oversized #484 after its useful pieces are harvested.

## Verification

```bash
python -m pytest tests/test_audit_extracted_manifests.py
python -m py_compile scripts/audit_extracted_manifests.py tests/audit_helpers.py tests/test_audit_extracted_manifests.py
python scripts/audit_plan_doc.py plans/PR-Audit-Extracted-Manifests.md
python scripts/audit_plan_doc_files_touched.py plans/PR-Audit-Extracted-Manifests.md origin/main
python scripts/audit_plan_doc_diff_size.py plans/PR-Audit-Extracted-Manifests.md origin/main
git diff --check
```

`python scripts/audit_extracted_manifests.py` is expected to return
non-zero on current main because #484 already surfaced preexisting sync
drift. That is the auditor doing its job, not a failure of this slice.

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `scripts/audit_extracted_manifests.py` | 123 |
| `tests/audit_helpers.py` | 27 |
| `tests/test_audit_extracted_manifests.py` | 63 |
| `plans/PR-Audit-Extracted-Manifests.md` | 76 |
| `docs/extraction/coordination/inflight.md` | 25 |
| **Total** | **~314** |
