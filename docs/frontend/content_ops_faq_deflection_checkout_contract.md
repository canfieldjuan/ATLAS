# Content Ops FAQ Deflection Checkout Contract

This is the frontend handoff for the one-time `$1,500` FAQ deflection report
unlock flow.

The portfolio owns Stripe Checkout creation and the results page UI. ATLAS owns
the free snapshot, paid report storage, Stripe webhook verification, and the
paid flag. The portfolio must not hold or synthesize the paid report before
ATLAS releases it.

## Flow

1. Run `faq_deflection_report` through the Content Ops execute route.
2. Render the returned free snapshot.
3. Create Stripe Checkout with the required metadata below.
4. Stripe sends `checkout.session.completed` to ATLAS.
5. ATLAS verifies the signed Stripe event and marks the report paid.
6. The portfolio hydrates the full report from ATLAS after the artifact route
   returns `200`.

## Execute Result

For `faq_deflection_report`, the gated execute result replaces the full report
step result with:

```ts
type GatedDeflectionExecuteResult = {
  request_id: string;
  snapshot: DeflectionSnapshot;
  full_report: {
    status: "locked";
    reason: "payment_required";
  };
};
```

Render `snapshot` on the free page. Do not derive answer text, evidence, source
IDs, or Markdown from this object.

## Portfolio Blob Submit Endpoint

The portfolio can hand ATLAS a support-ticket CSV blob and receive the same
gated execute response:

```http
POST /content-ops/deflection-reports/submit
```

Request:

```ts
type DeflectionReportSubmitRequest = {
  blob_url: string; // HTTPS Vercel Blob URL, public or signed
  support_platform: "zendesk" | "intercom" | "help_scout" | "other";
  company_name: string;
  contact_email: string;
  limit?: number; // default 1000, max 1000
};
```

ATLAS fetches the blob server-side, parses it as CSV, normalizes rows through
the support-ticket input package, and runs synchronous `faq_deflection_report`
generation. A `200` response means the report row exists for the authenticated
ATLAS account and `/snapshot` should be immediately readable.

Caps and failures:

- `blob_url` must be `https://`; private/signed URLs are preferred when the
  portfolio makes them available.
- ATLAS rejects localhost/private/link-local blob hosts after DNS resolution
  and does not follow redirects.
- Blob body cap: 50 MB. Larger responses return `413`.
- Sync source-material cap: first 1,000 parsed rows, or `limit` when lower.
- Empty CSVs or CSVs with no usable customer wording return `400`.
- Fetch and CSV parse failures return `400`.

## Free Snapshot Endpoint

```http
GET /content-ops/deflection-reports/{request_id}/snapshot
```

Responses:

- `200`: `DeflectionSnapshot`
- `404`: no report exists for this `request_id` in the authenticated account

The authenticated ATLAS scope supplies `account_id`; do not put `account_id` in
the path or query string.

## Paid Artifact Endpoint

```http
GET /content-ops/deflection-reports/{request_id}/artifact
```

Responses:

- `200`: `FAQDeflectionReportArtifact`
- `403`: report exists but `paid=false`; keep the unlock CTA visible
- `404`: no report exists for this `request_id` in the authenticated account

For now, this route is the paid-state read. Treat `200` as unlocked and `403` as
locked. The portfolio should not request or cache the full artifact before
payment.

## Stripe Checkout Metadata

The portfolio must create a one-time Stripe Checkout session with:

```ts
type DeflectionCheckoutMetadata = {
  source: "content_ops_deflection_report";
  account_id: string;
  request_id: string;
};
```

Required session properties:

- `mode: "payment"`
- `metadata.source: "content_ops_deflection_report"`
- `metadata.account_id`: the same ATLAS account/tenant id that owns the report
- `metadata.request_id`: the Content Ops request id returned by execute
- `amount_total >= 150000`
- `currency: "usd"`

ATLAS stores the Stripe Checkout session id as the report `payment_reference`
after the signed webhook marks the report paid.

## Trust Boundary

Use the Stripe webhook path:

```text
Stripe webhook -> ATLAS verifies -> ATLAS marks paid
```

The portfolio does not call an authed "mark paid" endpoint after Checkout. The
operator route exists for privileged internal recovery:

```http
POST /content-ops/deflection-reports/{request_id}/paid
```

Do not wire customer checkout completion to that route. It is not the customer
payment trust path.

## Failure Handling

- Missing or invalid metadata leaves the report locked.
- Wrong amount or currency leaves the report locked.
- A valid paid event with no matching report row returns non-2xx to Stripe so
  the event can retry or be reconciled.
- A repeated Stripe event is idempotent through the Stripe event id.

## Related Contracts

- [`content_ops_faq_report_contract.md`](./content_ops_faq_report_contract.md)
- [`content_ops_faq_deflection_snapshot_example.json`](./content_ops_faq_deflection_snapshot_example.json)
- [`content_ops_faq_deflection_report_example.json`](./content_ops_faq_deflection_report_example.json)
