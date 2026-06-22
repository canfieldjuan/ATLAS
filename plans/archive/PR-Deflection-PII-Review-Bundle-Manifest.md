# PR-Deflection-PII-Review-Bundle-Manifest

## Why this slice exists

#1742 now has the local operator handoff in two halves: #1769 creates a
sanitized review-bundle directory from an operator-labeled source, and #1770
scores that directory with predictable recall score artifacts. The remaining
pre-source gap is artifact/versioning prep: a future real-source bundle should
be self-describing without asking the reviewer to infer which fixed filenames,
schema versions, counts, and score status belong together.

Root cause: the bundle directory is deterministic by filename, but there is no
single sanitized manifest that records the bundle's artifact inventory and score
status. That makes the eventual "run locally, review, then version the
surrogate-only eval artifact" step depend on convention rather than an explicit
machine-readable contract.

This slice fixes that handoff root by adding a sanitized review-bundle manifest
that the builder writes and the scorer updates. It does not choose the real
source, persist raw labels/text, set thresholds, change scrubber behavior, or
promote the advisory score to a gate.

Diff budget note: the synced LOC total is over 400 because the manifest contract
must be proven on both halves of the bundle flow: builder valid/blocked output
and scorer success/failure updates. The review fixes add same-class regression
coverage for unreadable inputs, blocked manifests, and stale corpus inventory;
splitting those would leave the review-bundle self-description contract only
partly enforced.

Review-fix root cause: the first manifest pass treated builder provenance as a
mostly static record once scoring began, and it resolved bundle output paths only
after the source JSON was readable. That left three self-description gaps:
unreadable sources had no blocked manifest, scorer rewrites dropped blocked-build
error codes, and score load failures could leave stale corpus-present inventory.
This fix moves bundle path resolution before source reads, preserves sanitized
build blocking codes during score updates, and clears stale corpus inventory when
the bundle corpus cannot be loaded.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Vertical slice
Max files: 5

1. Add a predictable manifest filename, exposed as
   `REVIEW_BUNDLE_MANIFEST_NAME`, to the bundle builder.
2. Have valid and blocked bundle builds write a sanitized manifest with relative
   artifact filenames, schema versions, status, and count metadata.
3. Have `score_deflection_pii_recall.py --review-bundle-dir` update that
   manifest with recall score JSON/Markdown entries after bundle scoring.
4. Add focused tests proving the manifest is written/updated, remains sanitized,
   and does not require the real operator source.
5. Review update: keep manifests self-describing for unreadable sources,
   blocked-source score attempts, and stale corpus inventory after score load
   failures.

### Review Contract

- Acceptance criteria:
  - [ ] A valid review-bundle build writes the file named by
        `REVIEW_BUNDLE_MANIFEST_NAME` beside the source summary, Markdown
        summary, and surrogate eval corpus.
  - [ ] A blocked/invalid source still writes a manifest that records the
        blocked status, sanitized summaries, and absent surrogate artifact.
  - [ ] Bundle scoring updates the same manifest with recall score artifact
        filenames, score schema/status, and headline counts.
  - [ ] Unreadable source input still writes a blocked manifest without
        pretending source summary files or surrogate corpora exist.
  - [ ] Scoring a blocked build manifest preserves the builder's sanitized
        `blocking_error_codes`.
  - [ ] Corpus load failures clear stale corpus-present inventory before the
        score manifest is rewritten.
  - [ ] Manifest paths are bundle-relative filenames only; no absolute paths,
        raw source text, raw label spans, or high-risk raw tokens are emitted.
  - [ ] Existing individual output flags and bundle filenames remain compatible.
- Affected surfaces:
  - `scripts/build_deflection_pii_surrogate_eval_corpus.py`
  - `scripts/score_deflection_pii_recall.py`
  - bundle/scorer tests
- Risk areas: raw PII/path echo in manifest metadata, stale or contradictory
  manifest state on invalid reruns, and scope creep into policy/gating.
- Reviewer rules triggered: R1, R2, R3, R10, R14; boundary-probe required
  because this writes raw-source-adjacent review metadata.

### Files touched

- `plans/PR-Deflection-PII-Review-Bundle-Manifest.md`
- `scripts/build_deflection_pii_surrogate_eval_corpus.py`
- `scripts/score_deflection_pii_recall.py`
- `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py`
- `tests/test_score_deflection_pii_recall.py`

## Mechanism

The builder gets manifest constants:

- `REVIEW_BUNDLE_MANIFEST_NAME`
- `REVIEW_BUNDLE_MANIFEST_SCHEMA_VERSION`

When `--review-bundle-dir` is used, the builder writes the normal bundle files
and then writes a manifest containing only sanitized metadata: relative
filenames, source-summary schema/status, surrogate-corpus schema/counts when
present, and a blocked/ok status. Invalid sources still remove stale surrogate
artifacts before the manifest is written, so the manifest does not claim a
present corpus when the source failed validation. Bundle paths are resolved
before the source read, so malformed or non-UTF-8 source files can still write a
blocked manifest whose source summary and surrogate corpus entries are marked
absent.

The scorer uses the same manifest filename/schema and, in `--review-bundle-dir`
mode, reads any existing manifest, updates the recall-score JSON/Markdown
entries, and rewrites the manifest. If scoring fails, the manifest records the
sanitized score status/error codes; if scoring passes, it records the headline
counts that reviewers need before versioning or threshold discussions. The
scorer preserves sanitized build `blocking_error_codes` from blocked manifests
and, on corpus load failures, clears stale corpus-present schema/count metadata
before rewriting the manifest.

## Intentional

- No source selection or real-source run. The operator source remains transient
  and out of repo scope.
- No hashes or timestamps in this first manifest; stable count/schema/status
  metadata is enough to make the handoff explicit without introducing
  reproducibility churn.
- No threshold recommendation and no advisory-to-gating flip.
- No scrubber, detector, scorer math, or NER/open-set-name behavior change.

## Deferred

- Operator selection of the real transient source, corpus size/archetype mix,
  labeling ownership, and labeling-quality review remain open #1742 decisions.
- Running the full bundle plus scoring flow against the real source and
  committing/versioning the surrogate-only eval artifact remains deferred until
  that source is supplied and reviewed.
- Threshold selection and advisory-to-gating promotion remain deferred.

Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py tests/test_score_deflection_pii_recall.py -q -- 62 passed.
- python -m py_compile scripts/build_deflection_pii_surrogate_eval_corpus.py scripts/score_deflection_pii_recall.py tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py tests/test_score_deflection_pii_recall.py -- passed.
- python scripts/score_deflection_pii_recall.py --json -- status ok; free_high_severity_gate_eligible_leak_count=0, deferred_open_set_name_leak_count=1.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 188 matching tests are enrolled.
- bash scripts/check_ascii_python.sh -- passed.
- git diff --check -- passed.
- python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**' -- ratchet gate passed.
- python scripts/maturity_sweep_file_lane.py scripts/build_deflection_pii_surrogate_eval_corpus.py scripts/score_deflection_pii_recall.py --tests-root tests --baseline tests/maturity_sweep/baseline_deflection_lane.json --min-score 8 --sensitive-glob 'scripts/**' -- ratchet gate passed.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-deflection-pii-review-bundle-manifest.md -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-PII-Review-Bundle-Manifest.md` | 161 |
| `scripts/build_deflection_pii_surrogate_eval_corpus.py` | 155 |
| `scripts/score_deflection_pii_recall.py` | 135 |
| `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` | 88 |
| `tests/test_score_deflection_pii_recall.py` | 169 |
| **Total** | **708** |
