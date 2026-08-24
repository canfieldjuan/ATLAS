# PR-EOM-Residential-Onboarding-Quality-Check-Copy

## Why this slice exists

The operator reported that point 3 of the EOM onboarding email tells a
customer to walk the space with a team lead at the start or end of a visit. That
instruction does not describe the intended post-cleaning quality-check flow.
This product-polish slice changes the source template for future onboarding
drafts only, under the operator's explicit copy direction in this session.

### Problem-derived contract

- Root cause: `ONBOARDING_TEMPLATE` presents an ambiguous in-visit walkthrough
  as the way to communicate priorities, even though the intended customer action
  is to inspect the cleaned space after service and report anything needing
  attention. The template's own module contract also forbids clock promises,
  so an immediate-remediation promise would be an invalid implementation of
  that action.
- Correct fix must touch/change: replace only point 3 in
  `atlas_brain/templates/email/onboarding_welcome.py`, and add a direct renderer
  assertion in `tests/test_eom_lead_conversion.py` for the approved,
  non-temporal post-cleaning wording and absence of the legacy team-lead
  instruction.
- Must not change: first-clean booking, draft enqueue/idempotency, approval or
  send behavior, recipient resolution, existing persisted draft bodies,
  business contact details, APIs, schemas, migrations, Website/Tracker code,
  and unrelated onboarding copy.

### Contract revision (review evidence)

- New evidence: the template documents a no-clock-promise rule at
  `atlas_brain/templates/email/onboarding_welcome.py:6-8`; the first-clean
  completion query admits EOM leads without a `customer_type` predicate at
  `atlas_brain/services/crm_provider.py:3766-3789` and always invokes this
  renderer at `atlas_brain/services/crm_provider.py:3908-3912`.
- Revised root cause: the old in-visit direction is wrong, and the first
  proposed replacement accidentally made a clock promise while the plan
  misdescribed the shared EOM first-clean template as residential-only.
- Revised required change surface: retain the same template and focused test;
  make the assurance non-temporal and state the existing shared EOM first-clean
  scope accurately in this plan and the PR body.
- Revised non-scope: do not add customer-type routing, a new template variant,
  or a lifecycle admission rule to a copy correction.
- Revised verification: the focused renderer test must require the non-temporal
  assurance and reject both the former team-lead direction and `right away`.

## Scope (this PR)

Ownership lane: eom/residential-onboarding-copy
Slice phase: Product polish
Max files: 3

1. Replace point 3 with a non-temporal post-cleaning quality-check invitation
   for newly rendered English EOM onboarding drafts.
2. Render and test the shared-template wording without adding customer-type
   routing or changing the first-clean lifecycle.

### Review Contract

- Acceptance criteria:
  1. `format_onboarding_welcome()` renders point 3 as a non-temporal,
     after-cleaning walkthrough that asks the customer to report anything needing attention
     and says the team will take care of it; the focused renderer test settles
     the exact customer-visible text.
  2. The rendered body no longer contains the legacy `team lead at the start or
     end` instruction or `right away`; the same focused test settles both
     removals.
  3. The established shared EOM first-clean draft enqueue path still obtains its
     subject and body through `format_onboarding_welcome()` at
     `atlas_brain/services/crm_provider.py:3960-3962`; the existing enqueue
     regression remains in the focused test file.
- Reachability proof: a first-clean booking reaches
  `DatabaseCRMProvider._enqueue_eom_onboarding_email_draft()`, which renders
  the template before inserting the approval-only draft; the direct renderer
  test verifies the observable body that this path snapshots.
- Affected surfaces: the shared English EOM onboarding-welcome template and
  its focused EOM lead-conversion test coverage.
- Risk areas: accidental reintroduction of the legacy wording; changing the
  queue/send lifecycle while editing customer-facing copy.
- Reviewer rules triggered: R1 requirements match, R2 test evidence, R5
  backward compatibility, R10 maintainability, R12 deployment safety, R14
  codebase verification.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: N/A - no boundary change.
- Replaced-path behaviors: N/A - no boundary change.
- Guard-relevant fields: N/A - no boundary change.
- Caller x input shape: N/A - no boundary change.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no guard/config boundary change.
- Explicit value probe: N/A - no guard/config boundary change.
- Absent value probe: N/A - no guard/config boundary change.
- Default-session/default-context probe: N/A - no guard/config boundary change.
- Side-effect ordering: N/A - no guard/config boundary change.

### Files touched

- `atlas_brain/templates/email/onboarding_welcome.py`
- `plans/PR-EOM-Residential-Onboarding-Quality-Check-Copy.md`
- `tests/test_eom_lead_conversion.py`

## Mechanism

Replace the third numbered paragraph in the existing shared English template
with non-temporal post-cleaning quality-check language. The renderer continues
to format the same subject, customer name fallback, and business contact values.
A focused test calls that renderer directly and asserts the approved wording is
present while the superseded team-lead and `right away` wording are absent.

## Intentional

- The email continues to describe only future draft bodies. Existing draft rows
  remain immutable snapshots for office review; this slice does not bulk-edit
  or resend customer communications.
- Current code uses this template for all EOM first-clean drafts; the corrected
  quality-check wording is intentionally neutral across that existing shared
  output. This slice does not introduce a residential/commercial routing rule.
- This is an English-template correction. No Spanish counterpart exists in the
  current template package, so this slice does not invent a parallel
  localization surface.

## Deferred

- None.

Parking predicate: this copy-only slice parks customer-type routing,
localization, delivery/lifecycle, and broader CRM/template hardening unless it
directly prevents future shared EOM first-clean drafts from conveying the
post-service quality-check instruction without a clock promise.

Parked hardening: none.

## Verification

- Completed locally: focused onboarding-template regression selection
  (`7 passed, 219 deselected`) and `py_compile` for the changed Python template.
- Pending before push: the repository's mandatory `scripts/push_pr.sh` review
  wrapper, including its full unit-gate mirror for this exact update; GitHub CI
  will independently run its required checks after the push.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/templates/email/onboarding_welcome.py` | 4 |
| `plans/PR-EOM-Residential-Onboarding-Quality-Check-Copy.md` | 153 |
| `tests/test_eom_lead_conversion.py` | 14 |
| **Total** | **171** |
