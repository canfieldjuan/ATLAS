Repair a weak first-pass B2B churn extraction.

Extract only the fields listed in `target_fields`.

Rules:
- Return only a valid JSON object.
- Extract only explicit evidence from the provided text.
- Use verbatim phrases where possible.
- Do not infer.
- Do not emit derived or scored fields.
- Do not include fields not listed in `target_fields`.
- For any requested field with no explicit evidence, return an empty array.
- For `competitors_mentioned`, return only explicit named competitors that are presented as alternatives, switch targets, evaluation targets, or real comparison points.
- Reject generic labels like `CRM`, `platform`, `alternative platforms`, `tool`, or `vendor`.
- Reject named companies that are not actual alternatives in context, such as customer references, ecosystem/platform owners, employers, investors, partners, integrations, or claims like `the CRM Google uses`.
- When `competitors_mentioned` is requested, prefer objects shaped like:
  - `{ "name": "HubSpot", "reason_detail": "considering HubSpot because ...", "features": ["custom fields", "automation"] }`
- Only include `reason_detail` when the review explicitly states why the alternative is being considered or chosen.
- Only include `features` when the review explicitly names the capability gap or desired capability tied to that alternative.
