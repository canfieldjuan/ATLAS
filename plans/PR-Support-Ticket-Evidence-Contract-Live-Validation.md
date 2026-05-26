# PR: Support-Ticket Evidence Contract Live Validation

## Why this slice exists

PRs #977 and #979 moved the support-ticket blog/landing generation path away
from downstream detector-only fixes and into a source-truth contract:

- resolution evidence controls whether concrete FAQ answer steps are allowed
- dated-window evidence controls whether calendar-window/cadence claims are
  allowed
- measured-outcome evidence controls whether impact, retention, churn,
  support-volume, and time-savings claims are allowed

The remaining question is whether the real live generation path now produces
customer-facing drafts that obey that contract. Earlier live Haiku/Sonnet
validation showed unsupported outcome claims and invented procedural FAQ steps
when uploaded tickets only carried customer questions. This slice reruns the
live support-ticket landing-page and blog generation path after the evidence
contract landed and records what the generated content actually does.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider-live-validation
Slice phase: Functional validation

1. Run fresh live Haiku landing-page generation with the packaged
   support-ticket CSV and export the saved draft/result artifacts.
2. Run fresh live Haiku blog-post generation with the same support-ticket CSV
   and export the saved draft/result artifacts.
3. Evaluate both saved drafts with the deterministic support-ticket
   generated-content evaluator.
4. Manually audit the exported copy for the source-contract drift classes this
   lane has been closing:
   - unsupported outcome/support-volume/churn/time-savings claims
   - unsupported date-window or cadence claims
   - invented procedural answer steps when no resolution evidence exists
5. Write a validation note under `docs/extraction/validation/` that records the
   commands, artifacts, pass/fail result, and any remaining follow-up.
6. If the fresh output exposes a data-truthfulness blocker, fix it at the
   source in this PR and update the validation note. If it only exposes a
   non-blocking product polish issue, park it in `HARDENING.md` or
   `ATLAS-HARDENING.md` according to ownership.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Support-Ticket-Evidence-Contract-Live-Validation.md` | Plan doc for this validation slice. |
| `docs/extraction/validation/support_ticket_evidence_contract_live_validation_2026-05-26.md` | Fresh live-generation validation record. |
| `extracted_content_pipeline/support_ticket_generated_content_eval.py` | Close false-green evaluator gaps found in fresh live blog output. |
| `tests/test_evaluate_support_ticket_generated_content.py` | Negative fixtures for each new detected false-green class. |
| `atlas_brain/skills/digest/blog_post_generation.md` | Tighten support-ticket blog prompt guidance at the mapped source. |
| `extracted_content_pipeline/skills/digest/blog_post_generation.md` | Synced extracted prompt copy. |
| `extracted_content_pipeline/support_ticket_input_package.py` | Replace the default promissory support-ticket secondary keyword. |
| `tests/test_support_ticket_provider_landing_blog_execute.py` | Update support-ticket provider context expectation. |
| `tests/test_extracted_content_ops_live_execute_harness.py` | Update support-ticket blog metadata expectation. |

## Mechanism

Use the existing live smoke harness so the validation exercises the real host
wiring instead of a unit-only path:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --account-id <unique-account> \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft <artifact>.json \
  --output-result <result>.json \
  --evaluate-generated-content \
  --json
```

Run once for `landing_page` and once for `blog_post`. The `haiku.env` override
keeps the validation on the cheaper Claude Haiku family. Saved artifacts stay
under `tmp/`; the committed validation doc names them but does not commit
draft JSON.

The validation succeeds only if the live path either saves a draft that passes
the deterministic evaluator, or refuses to save a draft whose generated copy
would mislead users.

## Intentional

- This is not a new FAQ-generation feature slice; FAQ ownership stays with the
  other session.
- This adds detector coverage because fresh live output proved the current
  evaluator had false greens that allowed misleading generated copy.
- This does not switch the default model. Haiku is intentionally used for
  validation because it is cheaper and has been the stricter stress case.
- This does not commit live draft JSON artifacts; the docs record their local
  `tmp/` paths and summarized results.

## Deferred

- Parked hardening: none. Root `HARDENING.md` currently has only FAQ
  scale/backpressure work owned by the FAQ lane; `ATLAS-HARDENING.md` holds
  older Atlas deep-dive blog items outside this support-ticket provider lane.
- Production load/backpressure for very large hosted FAQ runs remains parked as
  `FAQSCALE-1` and is not part of this live content validation slice.
- Support-ticket blog prompt/blueprint restructuring is the next product slice:
  after this PR, unsafe blog drafts are blocked, but the current long-form blog
  task can still fail to save because Haiku keeps trying to fill missing
  outcome evidence. The next slice should make the source task more explicitly
  descriptive so it can produce a passing blog without copied guardrail text.
- Broader product acceptance testing across many customer CSV shapes remains a
  later robust-testing slice after this representative live validation is
  recorded.

## Verification

Completed / planned:

- Live Haiku landing-page smoke with support-ticket CSV and generated-content
  evaluation - passed.
- Live Haiku blog-post smoke with support-ticket CSV and generated-content
  evaluation - initial drafts exposed false greens; final run blocked unsafe
  output before save.
- Manual copy audit of both saved drafts / failed candidates for unsupported
  outcomes, unsupported date/cadence, and invented procedural steps.
- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q`
  - `40 passed`.
- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py tests/test_extracted_support_ticket_input_package.py tests/test_support_ticket_provider_landing_blog_execute.py tests/test_extracted_content_ops_live_execute_harness.py -q`
  - `81 passed`.
- `bash scripts/local_pr_review.sh`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~150 |
| Validation doc | ~170 |
| Evaluator, prompt, keyword fix, tests | ~430 |
| **Total** | **~750** |

This exceeds the soft 400 LOC target because the live validation exposed several
related false-green branches in the same safety gate. Splitting the detector
fixtures from the prompt/input fix would leave one PR knowingly accepting
misleading generated output.
