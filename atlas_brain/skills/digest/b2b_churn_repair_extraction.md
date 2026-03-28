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
- For `competitors_mentioned`, return only explicit named competitors. Reject generic labels like `CRM`, `platform`, `alternative platforms`, `tool`, or `vendor`.
