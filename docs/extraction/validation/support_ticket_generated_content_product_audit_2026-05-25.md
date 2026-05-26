# Support-Ticket Generated Content Product Audit - 2026-05-25

## Scope

This audit looks past the deterministic readiness checks and inspects the
actual customer-facing copy generated from uploaded support tickets. The source
artifacts are the saved Haiku exports already produced by the support-ticket
live smoke lane:

- Landing page export:
  `tmp/support_ticket_live_haiku_eval_20260525/landing-page-draft.json`
- Blog export:
  `tmp/support_ticket_live_blog_gate_20260525/blog-post-draft-cadence2.json`

Ownership lane: `content-ops/support-ticket-provider`

## What Passed

Both artifacts were grounded in the uploaded support-ticket context:

- 4 uploaded support-ticket rows.
- 2 direct customer questions.
- Top clusters:
  - `email and profile updates` - 2 tickets
  - `reporting friction` - 2 tickets
- Customer wording was preserved in visible copy.

The final blog export passed SEO/AEO readiness, GEO readiness, and the
deterministic generated-content evaluator after the earlier timeframe, cadence,
and percentage-claim fixes.

The landing export also passed SEO/AEO readiness, GEO readiness, and generated
content evaluation.

## Product Copy Gap

The content still drifted from "uploaded tickets show likely FAQ
opportunities" into guaranteed future outcomes. Uploaded tickets can show
repeated questions and customer wording, but they do not prove that future
support tickets will disappear, that volume will drop immediately, or that
customers will churn less.

Examples from the landing export:

- `Support tickets for those questions drop.`
- `These are your highest-impact FAQ opportunities and will reduce repeat
  support tickets fastest.`
- `Answering these two questions in your help center will reduce incoming
  support tickets immediately.`

Examples from the blog export:

- `Writing one clear FAQ entry now prevents future support tickets on the same
  topic.`
- `The 2 FAQ entries derived from your uploaded tickets will prevent future
  support interactions on the same topics.`
- `Customers who find answers in your FAQ stay longer and churn less.`
- `Over time, your FAQ will become the first place customers look, and your
  support queue will shrink accordingly.`

Those may be plausible business goals, but they are not directly proven by the
uploaded CSV. The safer copy standard is:

- `can help reduce repeat questions`
- `gives your team the first FAQ topics to fix`
- `track whether ticket volume drops after publication`
- `can make answers easier for customers to find before they email support`

## Source Fix

This slice adds a deterministic evaluator check named
`support_ticket_outcome_claims_grounded`. It fails guaranteed outcome language
when support-ticket generated content claims that tickets will drop, future
support interactions will be prevented, the support queue will shrink,
customers will churn less, or support reduction will happen immediately.

The landing-page and blog prompts now tell the model to use cautious support
outcome language unless the source context includes explicit outcome metrics.

## Detector Proof

After adding the check, rerunning the evaluator against the saved artifacts
fails the exact issue this audit found.

Landing export:

```bash
python scripts/evaluate_support_ticket_generated_content.py \
  --output landing_page \
  tmp/support_ticket_live_haiku_eval_20260525/landing-page-draft.json \
  --pretty
```

Result: failed on `support_ticket_outcome_claims_grounded`.

Detected unsupported claims:

- `Support tickets for those questions drop.`
- `These are your highest-impact FAQ opportunities and will reduce repeat
  support tickets fastest.`
- `Answering these two questions in your help center will reduce incoming
  support tickets immediately.`

Blog export:

```bash
python scripts/evaluate_support_ticket_generated_content.py \
  --output blog_post \
  tmp/support_ticket_live_blog_gate_20260525/blog-post-draft-cadence2.json \
  --pretty
```

Result: failed on `support_ticket_outcome_claims_grounded`.

Detected unsupported claims:

- `Writing one clear FAQ entry now prevents future support tickets on the same
  topic.`
- `Over time, your FAQ will become more comprehensive and your support queue
  will shrink.`
- `Customers who find answers in your FAQ stay longer and churn less.`
- `The 2 FAQ entries derived from your uploaded tickets will prevent future
  support interactions on the same topics.`
- `Over time, your FAQ will become the first place customers look, and your
  support queue will shrink accordingly.`

## Remaining Work

The next validation slice should rerun live Haiku landing and blog generation
with the stricter prompt/evaluator contract and record whether the new drafts
pass without guaranteed outcome claims.
