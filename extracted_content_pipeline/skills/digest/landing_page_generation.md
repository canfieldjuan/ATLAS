You generate ONE marketing landing page for a single marketing campaign.

The marketing campaign is JSON-encoded as `{campaign_json}`. Optional reasoning context (when present) lives inside the campaign payload under `reasoning_context` / `campaign_reasoning_context`. If the reasoning context carries a `narrative_plan` block under `canonical_reasoning`, use its `sections` array as the section spine; otherwise structure the page yourself from the campaign's persona and value_prop.

Output ONLY a single JSON object with this exact shape. No prose outside the JSON.

```json
{
  "title": "Page H1 / browser title (8-12 words)",
  "slug": "url-safe-slug-derived-from-campaign-name",
  "hero": {
    "headline": "Lead headline (max ~10 words)",
    "subheadline": "One-sentence subheadline that reinforces the value prop",
    "cta_label": "Book a demo",
    "cta_url": "/demo"
  },
  "sections": [
    {
      "id": "snake_case_section_id",
      "title": "Section heading",
      "body_markdown": "Section body in markdown",
      "metadata": {"order": 1}
    }
  ],
  "cta": {
    "label": "Primary CTA label",
    "url": "/demo",
    "variant": "primary"
  },
  "meta": {
    "title_tag": "<title> tag (50-60 chars)",
    "description": "<meta description> (120-160 chars)",
    "og_image_url": ""
  },
  "reference_ids": ["customer-logo-1", "case-study-2"]
}
```

Field rules:
- `title`: descriptive H1; reflects the campaign's value_prop; do NOT include date stamps.
- `slug`: lowercase, hyphenated, derived from `campaign.name`. URL-safe ASCII only.
- `hero.headline`: punchy, persona-aware. Subheadline reinforces the value prop in one sentence.
- `hero.cta_label` / `hero.cta_url`: the hero's primary CTA. Reuse for the page-level `cta` block unless the page uses a hero-CTA + sticky-footer-CTA pattern.
- `sections`: 3-6 ordered sections. Common shapes: problem / solution / social-proof / how-it-works / pricing / FAQ. Each section needs a non-empty `title` and `body_markdown`.
- `cta`: page-level primary CTA (label + url required). Optional `secondary_label` / `secondary_url`.
- `meta.description`: SEO meta description, 120-160 characters. Skip the title tag if it's identical to `title`.
- `reference_ids`: customer logos, case studies, testimonials, or other social-proof source ids the page cites. Empty when none.

When the reasoning context provides a `narrative_plan`, copy each plan section's `id`/`title` verbatim and write prose grounded in the plan's `claim_ids`. Persona-tailor the headline + subheadline for `campaign.persona`.

Source-row evidence policy:
- If campaign evidence or opportunity context has `source_type: "support_ticket"`, `source_type: "complaint"`, `source_type: "case"`, or `source_type: "conversation"`, treat it as service evidence, not buying-intent evidence.
- Use service framing such as "support evidence points to..." or "complaint narratives describe..." and do not claim the target account is evaluating, buying, switching, or considering a vendor unless account-specific CRM, call, or meeting evidence supports it.
- If source rows are reviews, use market-evidence framing rather than naming exact review sites.

Avoid: weasel words ("powerful", "robust", "leverage"), promises that can't be backed up, comparative claims about specific competitors unless the campaign provides them.
