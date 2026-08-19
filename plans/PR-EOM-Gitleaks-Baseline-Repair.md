# PR-EOM-Gitleaks-Baseline-Repair

## Why this slice exists

The Billing & Payments provider's `Security Guardrails` workflow is red on
`main` after the Manual Square queue landed. The real GitHub job and a local
scan with the workflow's pinned Gitleaks image both report two unbaselined
`generic-api-key` findings from historical commit
`beb840f14f7b3de4c003cd2ed625fe5a923a53fe`. The matched source is the
non-secret validation label `"Authenticated actor"`; there is no provider
credential to rotate or revoke. This is H-19 in the Billing & Payments
Hardening & Deferred issue (#2363), discovered while validating provider PR
#2434 and its downstream Website slice.

### Problem-derived contract

- Root cause: `docs/security/gitleaks-baseline.json` lacks the two exact
  historical fingerprints emitted for the already-merged Manual Square queue,
  so the full-history Gitleaks gate fails even though the scanner evidence
  contains no credential.
- Root-cause disposition: This change fixes the root cause of the observed
  full-history gate failure—the trusted baseline's missing exact-fingerprint
  membership—not a symptom. The `generic-api-key` phrase match is a
  third-party scanner heuristic; weakening that rule would expand the
  suppression boundary, and editing current source cannot erase the historic
  commit. The reviewed baseline record is therefore the most-upstream safe
  repository control for this already-historical false positive.
- Policy root cause: the canonical security runbook described only the
  provider-credential branch of a baseline rotation. It must document the
  stricter reviewed-false-positive branch this evidence supports, rather than
  requiring a fictitious credential rotation or relying on the broader label
  description alone.
- Correct fix must touch/change: add only the two scanner-generated,
  redacted finding records to the canonical Gitleaks baseline, preserving every
  existing fingerprint, and update `docs/SECURITY_GUARDRAILS.md` to require
  exact scanner evidence for a reviewed non-secret false positive. The PR must
  use the controlled `security-rotation` label and prove that the trusted
  baseline-growth guard accepts the labeled candidate and rejects the same
  candidate without that label.
- Must not change: Gitleaks rules, the security workflow, the historic Square
  product code, financial behavior, credential configuration, Git history,
  Gmail behavior, or the existing baseline entries.

## Scope (this PR)

Ownership lane: eom/billing-payments-security-hardening
Slice phase: Production hardening

Max files: 3

1. Add the two verified false-positive fingerprints to the full-history
   Gitleaks baseline and document the separate reviewed-false-positive path so
   the trusted main workflow can classify them as known historical findings
   without misrepresenting a credential rotation.
2. Prove the baseline is additive, its membership matches the scanner output,
   the documented path requires exact redacted evidence, and the existing
   controlled-rotation guard still fails closed without its required GitHub
   label.

### Review Contract

- Acceptance criteria:
   - [ ] `docs/security/gitleaks-baseline.json` preserves every
     `origin/main` fingerprint and adds only
     `beb840f14f7b3de4c003cd2ed625fe5a923a53fe:atlas_brain/services/commercial_billing_manual_square_invoices.py:generic-api-key:370`
     and `...:475`; settled by
     `python scripts/check_gitleaks_baseline_rotation.py --base-ref origin/main --head-ref HEAD --labels-json '["security-rotation"]'`.
   - [ ] The pinned Gitleaks full-history scan using the candidate baseline
     exits zero; settled by the same Docker invocation used by
     `.github/workflows/security_guardrails.yml:88-116`.
   - [ ] The exact candidate remains rejected without the controlled label;
     settled by the same baseline-rotation command with `--labels-json '[]'`
     returning exit code 1.
   - [ ] `docs/SECURITY_GUARDRAILS.md` retains provider rotation/revocation for
     real credentials and permits a no-credential path only for a reviewed
     non-secret false positive with exact scanner evidence, additive
     fingerprints, both pinned scan outcomes, and the controlled label.
   - [ ] The scanner/guard behavior remains regression-covered by
     `pytest -q tests/test_check_gitleaks_baseline_rotation.py`.
- Reachability proof: `.github/workflows/security_guardrails.yml:88-116`
   copies this canonical baseline on trusted `main` and the `Gitleaks full-history
   secret scan` job's successful result is the observable gate outcome. This
   PR adds no application runtime surface.
- Affected surfaces: `docs/security/gitleaks-baseline.json`; its trusted
   caller in `.github/workflows/security_guardrails.yml`; the documented policy
   in `docs/SECURITY_GUARDRAILS.md`; and the controlled baseline-growth policy
   in `scripts/check_gitleaks_baseline_rotation.py`.
- Risk areas: security scanner suppression scope, historical-fingerprint
   integrity, CI/release health, and false-positive classification.
- Reviewer rules triggered: R1, R2, R3, R10, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: N/A - this does not change Gitleaks rule or guard logic;
  it changes a reviewed, exact-match data input to the existing trusted scan.
- Replaced-path behaviors: N/A - unlisted fingerprints remain blocking.
- Guard-relevant fields: `Fingerprint` on each scanner-generated baseline
  record; all other fields remain scanner evidence, not application input.
- Caller x input shape: N/A - the workflow consumes one JSON array from the
  trusted ref; its changed set is declared below.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no runtime environment configuration
  changes. Trusted `main` copies the checked-in baseline before scanning.
- Explicit value probe: the candidate baseline passed to the pinned scanner.
- Absent value probe: the same scan without a baseline emits the two new
  historical fingerprints in addition to the existing baseline findings.
- Default-session/default-context probe: N/A - no request/session/config
  fallback exists.
- Side-effect ordering: N/A - the scan is read-only and performs no product or
  financial mutation.

### Closure Declaration

- Dependency: the `Fingerprint` member set in the Gitleaks baseline that
  decides whether a historical scanner finding is suppressed.
- Closed/open: CLOSED for the scanner output at the reviewed `HEAD`, pinned
  Gitleaks image digest, and `--log-opts=HEAD` history boundary. A future
  commit or scanner finding is outside this reviewed snapshot.
- Membership source: ENUMERATED from the real full-history scanner output, then
  individually vetted as non-secret. The baseline is deliberately not derived
  automatically because automatic suppression would be unsafe.
- Out-of-set behavior: any unlisted fingerprint remains a Gitleaks failure.
  Blocking is the safer direction because an unreviewed secret-shaped finding
  must stop CI rather than silently enter the trusted baseline.

### Files touched

- `docs/SECURITY_GUARDRAILS.md`
- `docs/security/gitleaks-baseline.json`
- `plans/PR-EOM-Gitleaks-Baseline-Repair.md`

## Mechanism

The trusted main workflow copies this JSON baseline before running the pinned
Gitleaks image across the full history. The baseline is an exact-fingerprint
exception list, so the two added records suppress only the two verified historic
false positives. The existing `Gitleaks baseline growth guard` independently
compares the candidate to `origin/main`, requires the `security-rotation`
label, rejects non-security paths, and rejects removal of prior fingerprints.
The security runbook now distinguishes that exact reviewed-false-positive path
from the mandatory credential rotation/revocation path for real secret
exposure; neither path permits a broad Gitleaks suppression.

## Intentional

- Do not modify the historical Manual Square source merely to evade a history
  scanner; that cannot remove the historic finding and would create unrelated
  product churn.
- Do not add a broad Gitleaks rule exclusion or `.gitleaksignore` entry; exact
  baseline records preserve the scanner's evidence and keep future matching
  findings blocking.
- Do not claim a credential rotation: the scanned text is the validation label
  `"Authenticated actor"`, not a provider credential. This documents the
  reviewed-false-positive evidence path in the canonical runbook while keeping
  rotation/revocation mandatory for real credential exposure.
- No application, financial, Gmail, credential, or deployment behavior changes.

## Deferred

- Provider credentials exposed by historical commit
  `d63a9b77b9727766e14e523626c22dd6c1c80da8` remain separately tracked in
  `HARDENING.md`; this false-positive repair does not weaken their required
  provider-side rotation/revocation.
- H-19 in #2363 is updated with the final PR and verification evidence.
- H-20 in #2363: after this baseline repair merges, update the unlabeled
  `scripts/check_gitleaks_baseline_rotation.py` guidance and its assertion to
  distinguish actual credential rotation/revocation from the documented
  reviewed-false-positive evidence path. It is deliberately a standalone
  follow-up because the trusted baseline-growth guard allows this PR's
  baseline/docs/plan paths only; changing that guard or its tests here would
  make the required fail-closed gate reject the current baseline change.

Parking predicate: This production-hardening slice parks any finding that does
not falsify the exact-fingerprint membership or full-history Gitleaks gate
contract—for example, unrelated dependency scanner backlogs, broad scanner
heuristic redesign, credential lifecycle work, and any product/runtime
behavior. A finding that adds, removes, or mismatches one of this baseline's
reviewed fingerprints, or weakens the existing fail-closed scan, remains in
scope.

Parked hardening: #2363 H-20 (standalone guard-message follow-up after this
baseline repair merges).

## Verification

- `pytest -q tests/test_check_gitleaks_baseline_rotation.py` - 16 passed.
- Pinned full-history Gitleaks scan without a baseline - exits 1 as expected;
  its two newly discovered records match the candidate baseline records exactly.
- Pinned full-history Gitleaks scan with
  `--baseline-path docs/security/gitleaks-baseline.json` - 0 findings and exit
  0.
- Base/candidate fingerprint comparison - 22 preserved, 24 total, exactly the
  two named Manual Square false-positive fingerprints added, none removed.
- `git diff --cached --check` - passed.
- `python scripts/check_gitleaks_baseline_rotation.py --base-ref origin/main
  --head-ref HEAD --labels-json '["security-rotation"]'` - passed; the
  security-only candidate is accepted only under the controlled label.
- The same baseline-growth command with `--labels-json '[]'` - rejected with
  exit 1; an otherwise identical candidate cannot grow the baseline without the
  label.
- `pytest -q tests/test_security_guardrails_workflow.py
  tests/test_security_policy_docs.py tests/test_check_gitleaks_baseline_rotation.py`
  - passed; the documentation and baseline policy tests remain aligned.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/SECURITY_GUARDRAILS.md` | 28 |
| `docs/security/gitleaks-baseline.json` | 2 |
| `plans/PR-EOM-Gitleaks-Baseline-Repair.md` | 221 |
| **Total** | **251** |
