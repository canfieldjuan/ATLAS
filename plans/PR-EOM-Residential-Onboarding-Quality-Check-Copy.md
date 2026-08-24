# PR-EOM-Residential-Onboarding-Quality-Check-Copy

## Why this slice exists

The operator reported that point 3 of the residential onboarding email tells a
customer to walk the space with a team lead at the start or end of a visit. That
instruction does not describe the intended post-cleaning quality-check flow.
This product-polish slice changes the source template for future onboarding
drafts only, under the operator's explicit copy direction in this session.

### Problem-derived contract

- Root cause: `ONBOARDING_TEMPLATE` presents an ambiguous in-visit walkthrough
  as the way to communicate priorities, even though the intended customer action
  is to inspect the cleaned space after service and report anything needing
  attention.
- Correct fix must touch/change: replace only point 3 in
  `atlas_brain/templates/email/onboarding_welcome.py`, and add a direct renderer
  assertion in `tests/test_eom_lead_conversion.py` for the approved post-cleaning
  wording and absence of the legacy team-lead instruction.
- Must not change: first-clean booking, draft enqueue/idempotency, approval or
  send behavior, recipient resolution, existing persisted draft bodies,
  business contact details, APIs, schemas, migrations, Website/Tracker code,
  and unrelated onboarding copy.

## Scope (this PR)

Ownership lane: eom/residential-onboarding-copy
Slice phase: Product polish

1. Replace point 3 with a post-cleaning quality-check invitation for newly
   rendered English onboarding drafts.
2. Render and test the new wording as a post-cleaning quality-check invitation.

### Review Contract

- Acceptance criteria:
  1. `format_onboarding_welcome()` renders point 3 as an after-cleaning
     walkthrough that asks the customer to report anything needing attention
     and says the team will take care of it right away; the focused renderer test
     settles the exact customer-visible text.
  2. The rendered body no longer contains the legacy `team lead at the start or
     end` instruction; the same focused test settles the removal.
  3. The established first-clean draft enqueue path still obtains its subject
     and body through `format_onboarding_welcome()` at
     `atlas_brain/services/crm_provider.py:3960-3962`; the existing enqueue
     regression remains in the focused test file.
- Reachability proof: a first-clean booking reaches
  `DatabaseCRMProvider._enqueue_eom_onboarding_email_draft()`, which renders
  the template before inserting the approval-only draft; the direct renderer
  test verifies the observable body that this path snapshots.
- Affected surfaces: the English residential onboarding-welcome template and
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

Replace the third numbered paragraph in the existing English template with
plain post-cleaning quality-check language. The renderer continues to format
the same subject, customer name fallback, and business contact values. A focused
test calls that renderer directly and asserts the approved wording is present
and the superseded team-lead wording is absent.

## Intentional

- The email continues to describe only future draft bodies. Existing draft rows
  remain immutable snapshots for office review; this slice does not bulk-edit
  or resend customer communications.
- This is an English-template correction. No Spanish counterpart exists in the
  current template package, so this slice does not invent a parallel
  localization surface.

## Deferred

- None.

Parked hardening: none.

## Verification

- Completed locally: focused onboarding-template regression selection
  (`7 passed, 219 deselected`) and `py_compile` for the changed Python template.
- Pending before push: the repository's single pre-push mechanical review bundle
  through `scripts/push_pr.sh`; the full suite remains a GitHub CI responsibility.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/templates/email/onboarding_welcome.py` | 4 |
| `plans/PR-EOM-Residential-Onboarding-Quality-Check-Copy.md` | 125 |
| `tests/test_eom_lead_conversion.py` | 13 |
| **Total** | **142** |
