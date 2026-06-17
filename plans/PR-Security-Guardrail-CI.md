# PR-Security-Guardrail-CI

## Why this slice exists

The operator requested a DIY security pass this week, ordered by
highest-severity-per-effort: full-history secret scanning, dependency CVE
scanning, SAST, DAST, and IaC/container checks. Atlas currently has many
product-specific CI workflows and runtime security modules, but no durable
repo-wide security guardrail workflow that continuously scans old commits,
dependency manifests, source patterns, Docker/compose config, and staging
runtime headers after the one-time pass. After the first PR run showed this
would add several minutes to every small slice, the guardrail is intentionally
scheduled for `main`, nightly, and manual runs rather than every pull request.
Reviewer follow-up also showed the first scanner run would launch red on known
Semgrep, Trivy, and dependency backlog, so this slice now lands the broad sweep
as advisory adoption telemetry while keeping Gitleaks blocking for unbaselined
new leaks. A second review pass clarified that secret scanning is the one
security check worth keeping on every PR because it is cheap and catches the
highest-blast-radius failure mode.

This is production hardening because it adds continuous repository guardrails
around existing Atlas products rather than changing product behavior.

The slice is over the 400 LOC soft cap because Checkov surfaced a repo-wide
GitHub Actions permissions issue while validating the new guardrail. Fixing it
correctly requires adding top-level read-only permissions to the existing
workflow fleet in the same PR that turns on the Checkov gate; otherwise the new
gate would ship permanently red or knowingly waived.

## Scope (this PR)

Ownership lane: security/workflow
Slice phase: Production hardening

1. Add CI workflows for repo-wide full-history Gitleaks, Python pip-audit,
   OSV dependency scans, Semgrep SAST, Trivy/Checkov config scans, and a ZAP
   baseline job for a configured staging URL, without adding the heavy scanner
   tax to every pull request.
2. Add Dependabot coverage for GitHub Actions, active npm lockfile projects,
   Python requirements directories, Dockerfiles, and compose files.
3. Document which checks are repo-wide versus target-specific so future
   sessions know what CI actually covers.
4. Remove current-tree hardcoded secret-shaped fallbacks surfaced by the
   adoption scan so the baseline covers history, not live branch-tip literals.

### Review Contract

Acceptance criteria:
- Full-history secret scanning checks out the complete branch history, uses the
  redacted adoption baseline for known historical findings, runs on every PR,
  push to `main`, manual run, and schedule, and still fails on new unbaselined
  leaks.
- Pull-request Gitleaks reads the adoption baseline from the trusted base branch
  after the baseline has landed, not from the checked-out PR head.
- A lightweight pull-request guard rejects Gitleaks baseline changes after the
  initial adoption baseline so future PRs cannot hide new secrets by adding
  their fingerprints to the baseline.
- Dependency scanning covers deterministic tracked Python requirements files and
  active npm lockfile projects, with OSV running recursively and pip-audit
  running advisory checks. The floating `requirements.asr.txt` input is parked
  until its VCS dependency is pinned.
- SAST scans the repository with Semgrep default, OWASP Top Ten, and Python
  rulesets and uploads advisory SARIF during adoption.
- DAST uses OWASP ZAP baseline against an operator-configured staging URL and
  does not pretend to be repo-wide source scanning.
- IaC/container guardrails cover Dockerfiles, compose files, GitHub Actions, and
  Terraform paths if Terraform is added later, with Trivy and Checkov advisory
  until the initial backlog is triaged.
- Dependabot is enabled for the package-manager ecosystems this repo already
  contains.
- New security workflow Actions are pinned to immutable commit SHAs, and the
  Gitleaks container is pinned by image digest.

Affected surfaces:
- GitHub Actions workflow configuration under `.github/workflows/`.
- Dependabot configuration under `.github/`.
- Security guardrail documentation under `docs/`.
- GraphRAG Supabase API route configuration fallbacks.
- Archived IndexNow script configuration.

Risk areas:
- Security: Gitleaks must fail closed on new leaks, and the noisy first-run
  scanner backlog must be visible without making `main` permanently red.
- CI stability: heavy scans and DAST must not slow every small PR.
- Supply chain: security workflow dependencies must not float on mutable action
  tags.
- Scope clarity: product-specific runtime DAST cannot be described as
  whole-repo coverage.
- Supply chain: third-party Actions and scanner installs are new CI
  dependencies.

Triggered reviewer rules:
- R1 Requirements match
- R2 Test evidence
- R3 Security/auth
- R8 Dependencies
- R11 CI/workflow
- R14 Codebase verification

### Files touched

- `.github/dependabot.yml`
- `.github/workflows/admin_costs_checks.yml`
- `.github/workflows/atlas_b2b_campaign_migration_checks.yml`
- `.github/workflows/atlas_blog_public_checks.yml`
- `.github/workflows/atlas_brand_voice_checks.yml`
- `.github/workflows/atlas_content_ops_auth_checks.yml`
- `.github/workflows/atlas_content_ops_claim_registry_checks.yml`
- `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml`
- `.github/workflows/atlas_content_ops_deflection_report_checks.yml`
- `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
- `.github/workflows/atlas_content_ops_generated_assets_checks.yml`
- `.github/workflows/atlas_content_ops_input_provider_checks.yml`
- `.github/workflows/atlas_content_ops_macro_writeback_checks.yml`
- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`
- `.github/workflows/atlas_deflection_migration_apply_checks.yml`
- `.github/workflows/atlas_intel_ui_checks.yml`
- `.github/workflows/atlas_invoicing_checks.yml`
- `.github/workflows/atlas_main_voice_startup_checks.yml`
- `.github/workflows/atlas_migrations_runner_checks.yml`
- `.github/workflows/claude.yml`
- `.github/workflows/extracted_competitive_intelligence_checks.yml`
- `.github/workflows/extracted_llm_infrastructure_checks.yml`
- `.github/workflows/extracted_pipeline_checks.yml`
- `.github/workflows/extracted_umbrella_checks.yml`
- `.github/workflows/marketing_content_check.yml`
- `.github/workflows/maturity_sweep_advisory.yml`
- `.github/workflows/portfolio_ui_checks.yml`
- `.github/workflows/security_dast_zap.yml`
- `.github/workflows/security_guardrails.yml`
- `.github/workflows/semantic_diff_advisor.yml`
- `Dockerfile`
- `Dockerfile.graphiti`
- `HARDENING.md`
- `_ARCHIVED_atlas-intel-next/scripts/indexnow.ts`
- `app/api/graphrag/delete/[id]/route.ts`
- `app/api/graphrag/documents/route.ts`
- `app/api/graphrag/process/[id]/route.ts`
- `app/api/graphrag/upload/route.ts`
- `atlas_video-processing/Dockerfile.vision`
- `atlas_video-processing/ingest/drone_client/Dockerfile`
- `atlas_video-processing/processing/video_stream_processor/Dockerfile`
- `docs/SECURITY_GUARDRAILS.md`
- `docs/security/gitleaks-baseline.json`
- `plans/PR-Security-Guardrail-CI.md`

## Mechanism

CI is split by guardrail type so each scanner has a clear blast radius:

- Secret scanning uses Gitleaks with `fetch-depth: 0`, so old commits are in
  scope rather than only the checked-out tree. The trusted-main adoption scan
  found 22 redacted historical findings; CI uses
  `docs/security/gitleaks-baseline.json` so the guardrail fails on new leaks
  while the exposed providers are rotated.
  On PRs, the scan uses the baseline from `origin/${{ github.base_ref }}` after
  the adoption baseline exists on the base branch; this prevents a PR from
  adding a secret and whitelisting its own fingerprint in the same change. A
  PR-only baseline guard allows this initial adoption baseline but rejects later
  baseline changes unless they are isolated into a dedicated security rotation
  workflow.
- SCA uses two layers: advisory pip-audit for deterministic Python requirements
  files where Atlas has explicit dependency manifests, and OSV Scanner's
  reusable workflows for recursive ecosystem coverage and GitHub code-scanning
  output.
- SAST installs the open-source Semgrep CLI in CI and runs `semgrep scan`
  against `p/default`, `p/owasp-top-ten`, and `p/python` with SARIF upload but
  without `--error` until the first-run finding backlog is triaged.
- IaC/container config scanning runs Trivy config mode and Checkov across the
  repo, covering Dockerfiles, compose, GitHub Actions, and future Terraform.
  Both run advisory during adoption so their first findings do not keep `main`
  red before triage.
- DAST resolves `ATLAS_DAST_TARGET_URL` from a repository variable, then runs
  ZAP baseline only when a target exists.
- Dependabot runs weekly for GitHub Actions, active npm package-lock projects,
  Python requirements directories, Dockerfiles, and Docker Compose.
- Newly introduced security workflow Actions are pinned to the commit SHAs behind
  their selected tags, and the Gitleaks Docker image is pinned by digest. A
  fleet-wide pinning/OIDC review for existing workflows is parked separately.
- Current-tree secret-shaped fallbacks are removed instead of being added to the
  baseline: GraphRAG API routes now require Supabase env vars, and the archived
  IndexNow script requires `INDEXNOW_KEY`.

## Intentional

- The full security sweep is intentionally not a pull-request gate. Atlas PRs
  are small and frequent; findings from scheduled/main runs should become an
  immediate fix PR for exposed secrets/exploitable risk or `HARDENING.md` debt
  for non-blocking dependency/config/SAST issues.
- Semgrep, Trivy, Checkov, pip-audit, and OSV are advisory for this adoption
  slice. The first run found existing backlog; launching blocking gates before
  triage would make `main` permanently red and slow unrelated small slices.
- ZAP is target-specific, not repo-wide. This PR wires the recurring scanner
  and skips cleanly until `ATLAS_DAST_TARGET_URL` is configured because there
  is no staging URL in the repository.
- Gitleaks uses a redacted baseline for existing historical findings. Without
  it, every run would stay red until provider rotation plus git-history cleanup
  are complete; with it, CI still scans the full history and blocks new leaks.
  PR scans use the trusted base baseline after adoption, and baseline changes
  after adoption are blocked by a lightweight PR guard.
- Image build scanning is not forced in this slice. Atlas has multiple heavy
  GPU/ASR/video Dockerfiles; this slice scans Dockerfile/compose configuration
  continuously and leaves per-image build/publish scans for the deployment
  workflow that owns those images.
- Archived npm lockfiles are documented but not added to Dependabot's update
  loop because archived projects should not receive routine dependency churn.
- GraphRAG API routes return a 500 when Supabase env vars are missing. That is a
  deliberate fail-closed behavior; the previous placeholder fallback could mask
  a deployment configuration error and tripped secret scanning.

## Deferred

1. Add per-image Trivy image scans to each image publishing workflow once the
   production image build/push paths are named.
2. Rotate/revoke provider credentials exposed in historical commit
   `d63a9b77b9727766e14e523626c22dd6c1c80da8`, then decide whether to rewrite
   history or keep the redacted Gitleaks baseline as the permanent boundary.
3. Add a controlled Gitleaks baseline rotation escape hatch for legitimate
   post-rotation baseline changes.
4. Rotate the archived IndexNow key that was removed from the branch tip but
   remains in git history.
5. Audit and pin remaining non-security workflow actions, and review
   `.github/workflows/claude.yml`'s `id-token: write` trigger posture.
6. Burn down the advisory Semgrep, Trivy, Checkov, pip-audit, and OSV backlog,
   then ratchet the scanners to blocking gates where the repo can keep them
   green.
7. Pin or retire the floating `NVIDIA/NeMo@main` dependency in
   `requirements.asr.txt`, then add that file back to pip-audit.
8. Configure repository variable `ATLAS_DAST_TARGET_URL` to the staging Atlas
   URL and add a ZAP rules file only after the first real ZAP run produces
   actionable baseline noise.
9. Decide whether `_ARCHIVED_atlas-intel-next/package-lock.json` should be
   deleted, reactivated, or added to a separate archived-dependency policy.

Parked hardening: `Rotate credentials exposed in historical .env`, `Add
controlled Gitleaks baseline rotation escape hatch`, `Rotate archived IndexNow
key`, `Audit remaining workflow action pins and Claude OIDC trigger`, `Burn down
advisory security scanner backlog`, and `Pin or retire floating ASR dependency
audit input` in `HARDENING.md`; these are required to ratchet the guardrails but
outside this repo-only CI wiring slice.

## Verification

- `python - <<'PY' ... yaml.safe_load(...)` across every GitHub Actions
  workflow YAML file plus `.github/dependabot.yml` - passed.
- `BASE_REF=main ... git cat-file -e "origin/${BASE_REF}:docs/security/gitleaks-baseline.json"` baseline-guard simulation - passed; `origin/main` has no adoption baseline yet, so this PR is allowed while future baseline changes will be blocked after merge.
- `docker buildx imagetools inspect ghcr.io/gitleaks/gitleaks:v8.30.1` - passed; resolved the image index digest pinned in CI.
- `git ls-remote` for `actions/checkout`, `actions/setup-python`, `github/codeql-action`, `google/osv-scanner-action`, `aquasecurity/trivy-action`, `bridgecrewio/checkov-action`, and `zaproxy/action-baseline` - passed; resolved the tag SHAs pinned in the new security workflows.
- `docker run --rm -v "$PWD:/repo" ghcr.io/gitleaks/gitleaks@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f git --redact --verbose --log-opts="origin/main" --report-format json --report-path /repo/tmp/gitleaks-origin-main.json /repo || true` - generated the 22-finding redacted trusted-main adoption baseline.
- `docker run --rm -v "$PWD:/repo" ghcr.io/gitleaks/gitleaks@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f git --redact --verbose --log-opts="HEAD" --baseline-path /repo/.gitleaks-baseline-ci.json /repo` - passed; 3,862 commits scanned, no unbaselined leaks.
- First CI/review pass found existing Trivy HIGH/CRITICAL, Semgrep, and dependency CVE backlog; this update changes those scanners to advisory adoption mode and parks the ratchet work in `HARDENING.md` rather than launching `main` red.
- `python /home/juan-canfield/.codex/plugins/cache/openai-curated/github/43313cc9/skills/gh-address-comments/scripts/fetch_comments.py > tmp/pr1623_review_comments.json && jq ...` - passed; confirmed the open Codex/reviewer threads addressed by this update.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/dependabot.yml` | 88 |
| `.github/workflows/admin_costs_checks.yml` | 3 |
| `.github/workflows/atlas_b2b_campaign_migration_checks.yml` | 3 |
| `.github/workflows/atlas_blog_public_checks.yml` | 3 |
| `.github/workflows/atlas_brand_voice_checks.yml` | 3 |
| `.github/workflows/atlas_content_ops_auth_checks.yml` | 3 |
| `.github/workflows/atlas_content_ops_claim_registry_checks.yml` | 3 |
| `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml` | 3 |
| `.github/workflows/atlas_content_ops_deflection_report_checks.yml` | 3 |
| `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml` | 3 |
| `.github/workflows/atlas_content_ops_generated_assets_checks.yml` | 3 |
| `.github/workflows/atlas_content_ops_input_provider_checks.yml` | 3 |
| `.github/workflows/atlas_content_ops_macro_writeback_checks.yml` | 3 |
| `.github/workflows/atlas_content_ops_review_workflow_checks.yml` | 3 |
| `.github/workflows/atlas_deflection_migration_apply_checks.yml` | 3 |
| `.github/workflows/atlas_intel_ui_checks.yml` | 3 |
| `.github/workflows/atlas_invoicing_checks.yml` | 3 |
| `.github/workflows/atlas_main_voice_startup_checks.yml` | 3 |
| `.github/workflows/atlas_migrations_runner_checks.yml` | 3 |
| `.github/workflows/claude.yml` | 3 |
| `.github/workflows/extracted_competitive_intelligence_checks.yml` | 3 |
| `.github/workflows/extracted_llm_infrastructure_checks.yml` | 3 |
| `.github/workflows/extracted_pipeline_checks.yml` | 3 |
| `.github/workflows/extracted_umbrella_checks.yml` | 3 |
| `.github/workflows/marketing_content_check.yml` | 3 |
| `.github/workflows/maturity_sweep_advisory.yml` | 3 |
| `.github/workflows/portfolio_ui_checks.yml` | 3 |
| `.github/workflows/security_dast_zap.yml` | 43 |
| `.github/workflows/security_guardrails.yml` | 286 |
| `.github/workflows/semantic_diff_advisor.yml` | 3 |
| `Dockerfile` | 14 |
| `Dockerfile.graphiti` | 17 |
| `HARDENING.md` | 56 |
| `_ARCHIVED_atlas-intel-next/scripts/indexnow.ts` | 6 |
| `app/api/graphrag/delete/[id]/route.ts` | 11 |
| `app/api/graphrag/documents/route.ts` | 11 |
| `app/api/graphrag/process/[id]/route.ts` | 11 |
| `app/api/graphrag/upload/route.ts` | 11 |
| `atlas_video-processing/Dockerfile.vision` | 7 |
| `atlas_video-processing/ingest/drone_client/Dockerfile` | 5 |
| `atlas_video-processing/processing/video_stream_processor/Dockerfile` | 5 |
| `docs/SECURITY_GUARDRAILS.md` | 121 |
| `docs/security/gitleaks-baseline.json` | 1 |
| `plans/PR-Security-Guardrail-CI.md` | 302 |
| **Total** | **1076** |
