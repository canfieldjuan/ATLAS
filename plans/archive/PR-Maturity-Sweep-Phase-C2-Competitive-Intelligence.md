# PR-Maturity-Sweep-Phase-C2-Competitive-Intelligence

## Why this slice exists

Issue #1689 is extending the maturity-sweep ratchet through Phase C after the
`atlas_brain` rollout. Phase C1 enrolled compact extracted reasoning, quality,
and evidence packages. This slice enrolls `extracted_competitive_intelligence`
by itself because its baseline is large enough that combining it with LLM
infrastructure would exceed the reviewable PR budget.

Root cause: `extracted_competitive_intelligence` contains B2B briefing,
vendor-registry, MCP, storage, and campaign surfaces, but it is not yet
included in the blocking maturity-sweep baseline workflow. A PR can add new
high-score files or new swallowed/bare exception debt there without the
ratchet firing. This fixes the root for this package by enrolling it directly
in the workflow and committing its current baseline.

Follow-up scope in this PR also makes the production product surface explicit.
The Python ratchet intentionally stays focused on the extracted package, while
a companion manifest guard tracks adjacent non-Python/runtime surfaces: Atlas
Intel B2B UI, Atlas Churn vendor/campaign UI, competitive storage migrations,
extracted campaign docs/examples, extracted campaign migrations, and the
portfolio campaign-review media surface.

The over-budget size is intentional and indivisible: the workflow, checker,
fixture tests, and full current manifest must land together so CI can both run
and fail closed on added or removed production-surface files. Splitting the
manifest from enforcement would create a temporary false-green gap.

This PR also carries a narrow CI unblock for `Security Guardrails`: the OSV
reusable workflow job requested `actions: read` but the caller job did not grant
it, causing startup failure before any job logs existed. The PR Gitleaks job is
also scoped to the PR head range (`origin/<base>..HEAD`) so this PR is not held
red by trusted-base findings that require a separate baseline-rotation PR.
Scheduled and main-push security scans still keep their full-history behavior.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add a blocking maturity-sweep ratchet gate for
   `extracted_competitive_intelligence`.
2. Add the committed baseline so current debt is tracked but new debt fails.
3. Add path triggers so PRs touching the package run the workflow.
4. Add a blocking competitive-intelligence product-surface manifest guard for
   adjacent UI, storage, docs/examples, and media files that are not scored by
   the Python maturity sweep.
5. Add focused fixture tests proving the product-surface checker passes on a
   matched manifest and fails for missing or untracked discovered files.
6. Add the missing read-only `actions: read` permission to the trusted OSV
   reusable workflow caller so the existing Security Guardrails workflow can
   start.
7. Scope the pull-request Gitleaks scan to PR commits from the PR head checkout
   while leaving scheduled and main-push full-history scans unchanged.

### Review Contract

Acceptance criteria:
- `pull_request` and `push` path filters include
  `extracted_competitive_intelligence/**`.
- CI has one blocking Phase C2 step that runs `scripts/maturity_sweep.py` for
  `extracted_competitive_intelligence` with the committed baseline and
  `--min-score 8`.
- The C2 step marks the whole package sensitive for new bare/swallowed
  exceptions.
- The baseline is generated from the current tree with `--update-baseline`.
- Existing maturity-sweep tests still pass.
- A scratch negative proof shows the shipped C2 command fails when new
  swallowed-exception debt is added under the enrolled package.
- A companion product-surface workflow runs
  `scripts/check_competitive_intelligence_product_surface_manifest.py` for the
  competitive-intelligence UI/storage/docs/media surface.
- The Atlas Intel B2B UI discovery glob is recursive so future subdirectories
  do not bypass the manifest guard.
- The product-surface manifest is generated from the current tree and must be
  updated intentionally when a discovered file is added or removed.
- Fixture tests cover the checker happy path, missing expected file, and newly
  discovered untracked file failure modes.
- Security Guardrails no longer fails at workflow startup on the OSV reusable
  workflow permission check.
- The pull-request Gitleaks scan checks PR commits from `origin/<base>..HEAD`
  instead of scanning trusted-base history already outside this PR.

Affected surfaces:
- `.github/workflows/maturity_sweep_advisory.yml`
- `.github/workflows/maturity_sweep_competitive_intelligence_surface.yml`
- `.github/workflows/security_guardrails.yml`
- `scripts/check_competitive_intelligence_product_surface_manifest.py`
- `tests/test_competitive_intelligence_product_surface_manifest.py`
- `tests/maturity_sweep/baseline_extracted_competitive_intelligence.json`
- `tests/maturity_sweep/competitive_intelligence_product_surface_manifest.json`
- `atlas-intel-ui/src/api/b2bClient.ts` and `atlas-intel-ui/src/pages/b2b/**`
- `atlas-churn-ui/src/components/Campaign*.tsx`
- `atlas-churn-ui/src/pages/Campaign*.tsx` and `atlas-churn-ui/src/pages/Vendor*.tsx`
- B2B/vendor/campaign/competitive storage migrations in `atlas_brain`,
  `extracted_competitive_intelligence`, and `extracted_content_pipeline`
- `extracted_content_pipeline` campaign docs/examples
- `portfolio-ui/public/media/gifs/campaign-review.gif`

Risk areas:
- CI enrollment/path-filter drift.
- False-green coverage for B2B/MCP/storage/campaign surfaces if new swallowed
  exceptions are not treated as sensitive.
- False-green coverage for product-facing UI/storage/media surfaces if they
  are not listed in the product-surface manifest.
- Baseline churn that could accept unrelated debt.
- Security workflow startup failure if a reusable workflow permission is not
  passed through by the caller job.
- Pull-request secret scans failing on trusted-base history instead of only the
  PR's commits.

- Reviewer rules triggered: R2, R10, R12, R14.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `.github/workflows/maturity_sweep_competitive_intelligence_surface.yml`
- `.github/workflows/security_guardrails.yml`
- `plans/PR-Maturity-Sweep-Phase-C2-Competitive-Intelligence.md`
- `scripts/check_competitive_intelligence_product_surface_manifest.py`
- `tests/test_competitive_intelligence_product_surface_manifest.py`
- `tests/maturity_sweep/baseline_extracted_competitive_intelligence.json`
- `tests/maturity_sweep/competitive_intelligence_product_surface_manifest.json`

## Mechanism

The advisory workflow adds the package directory to PR and main-push path
filters, then runs:

```bash
python scripts/maturity_sweep.py extracted_competitive_intelligence \
  --tests-root tests \
  --baseline tests/maturity_sweep/baseline_extracted_competitive_intelligence.json \
  --min-score 8 \
  --sensitive-glob 'extracted_competitive_intelligence/**'
```

The committed baseline snapshots current per-file counts. Future PRs fail if a
file score increases, a new file crosses the threshold, or a new bare/swallowed
exception appears anywhere in the package.

The companion product-surface workflow runs:

```bash
python scripts/check_competitive_intelligence_product_surface_manifest.py
```

That checker discovers files from the manifest's glob patterns and compares
them with the committed file list. New or removed UI/storage/docs/media files
fail the check until the manifest is updated intentionally.

The Security Guardrails OSV reusable workflow caller now passes through the
read-only `actions: read` permission requested by the reusable workflow. The PR
Gitleaks job checks out the PR head SHA, fetches the trusted base ref, and scans
`origin/<base>..HEAD`; the trusted full-history scan remains reserved for main,
scheduled, and manual trusted-ref runs.

## Intentional

- This does not fix existing competitive-intelligence brittleness; it ratchets
  the package so new brittleness cannot land silently.
- Full-package sensitivity is intentional because this package includes B2B
  data, MCP, campaign, and storage surfaces.
- The product-surface manifest is intentionally separate from the Python score
  gate because it includes TSX, SQL, JSON, Markdown, and media files.
- The product-surface workflow, checker, tests, and full manifest are kept in
  one PR so the new guard has no enforcement/baseline gap.
- The Security Guardrails permission change is read-only and limited to the
  trusted OSV reusable workflow caller.
- The PR Gitleaks range is intentionally limited to PR commits; trusted-base
  baseline growth still belongs in a labeled security-rotation PR.
- `extracted_llm_infrastructure` is deferred to keep this PR scoped to the
  competitive-intelligence lane.

## Deferred

- Phase C3: enroll `extracted_llm_infrastructure`.
- Phase C4: enroll `scripts/**` with a script-specific baseline.

Parked hardening: none.

## Verification

- `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"` - pass.
- `python -m pytest tests/test_maturity_sweep.py --noconftest -q` - pass, 14 passed.
- C2 ratchet command for `extracted_competitive_intelligence` - pass.
- Path-filter spot check for `extracted_competitive_intelligence/**` and retained `tests/**` - pass.
- Scratch negative proof in `extracted_competitive_intelligence/services/vendor_registry.py` - pass. A temporary swallowed-exception probe failed with `score increased (8 -> 13)` and `new sensitive-path SWALLOWED_EXCEPT (1 -> 2)`; the scratch code was removed and the clean C2 command reran successfully.
- `python scripts/check_competitive_intelligence_product_surface_manifest.py` - pass, 171 files.
- `python -m py_compile scripts/check_competitive_intelligence_product_surface_manifest.py tests/test_competitive_intelligence_product_surface_manifest.py` - pass.
- `python -m pytest tests/test_competitive_intelligence_product_surface_manifest.py -q` - pass, 3 passed.
- `python scripts/sync_pr_plan.py plans/PR-Maturity-Sweep-Phase-C2-Competitive-Intelligence.md --check` - pass.
- Security Guardrails prior startup failure pointed at `.github/workflows/security_guardrails.yml#L185`; the OSV caller now grants the requested `actions: read` permission.
- Security Guardrails PR secret scan previously failed on trusted-base commit `28623552cf9827cd9294c2fb109778d0ff102b76`; the PR job now checks out the PR head and scans `origin/<base>..HEAD`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 11 |
| `.github/workflows/maturity_sweep_competitive_intelligence_surface.yml` | 57 |
| `.github/workflows/security_guardrails.yml` | 6 |
| `plans/PR-Maturity-Sweep-Phase-C2-Competitive-Intelligence.md` | 216 |
| `scripts/check_competitive_intelligence_product_surface_manifest.py` | 74 |
| `tests/test_competitive_intelligence_product_surface_manifest.py` | 83 |
| `tests/maturity_sweep/baseline_extracted_competitive_intelligence.json` | 220 |
| `tests/maturity_sweep/competitive_intelligence_product_surface_manifest.json` | 196 |
| **Total** | **863** |
