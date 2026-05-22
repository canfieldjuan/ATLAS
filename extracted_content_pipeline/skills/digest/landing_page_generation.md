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
      "metadata": {
        "order": 1,
        "kind": "problem | solution | how_it_works | proof | pricing | faq | objection | conversion",
        "primary_question": "Buyer/search question this section answers, or empty string",
        "answer_summary": "35-60 word direct answer that also appears as the first paragraph of body_markdown"
      }
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
- `slug`: lowercase, hyphenated, derived from `campaign.name`. URL-safe ASCII only. Do not use generic slugs such as `landing-page`, `campaign`, `demo`, `offer`, or `page`.
- `hero.headline`: punchy, persona-aware. Subheadline reinforces the value prop in one sentence and makes the offer clear without hidden context.
- `hero.cta_label` / `hero.cta_url`: the hero's primary CTA. Reuse for the page-level `cta` block unless the page uses a hero-CTA + sticky-footer-CTA pattern.
- `sections`: 3-6 ordered sections. Common shapes: problem / solution / how-it-works / proof / pricing / FAQ. Each section needs a non-empty `title` and `body_markdown`.
- `sections.title`: use specific headings that describe the offer, audience, problem, or buyer question. Do not use generic headings such as "Overview", "Features", "Benefits", "Summary", "Introduction", or "Conclusion".
- `sections.metadata.kind`: choose the section role from problem, solution, how_it_works, proof, pricing, faq, objection, or conversion.
- `sections.metadata.primary_question`: include the plain-language buyer or search question the section answers when one is natural. Use an empty string when the section is not question-shaped.
- `sections.metadata.answer_summary`: write a 35-60 word direct answer for problem, solution, how-it-works, FAQ, and objection sections. Put the same answer at the start of `body_markdown` so the answer is visible on the page, not hidden in metadata.
- `cta`: page-level primary CTA (label + url required). Optional `secondary_label` / `secondary_url`. Do not output placeholder URLs such as `#`, `/#`, `javascript:void(0)`, or `javascript:;`.
- `meta.title_tag`: always include a concrete SEO title tag, ideally 50-60 characters and no more than 70 characters.
- `meta.description`: SEO meta description, 120-160 characters. Keep the title tag and description aligned with visible page copy.
- `reference_ids`: customer logos, case studies, testimonials, or other social-proof source ids the page cites. Empty when none. Do not invent reference ids.

Readiness rules:
- The first viewport must make it clear what the offer is, who it is for, and why that reader should care.
- Include a clear problem section and a clear solution or how-it-works section tied to `campaign.value_prop`.
- Include objection coverage when the campaign gives enough context. This can be a FAQ, pricing, implementation, risk-reversal, comparison, security, or proof section.
- Do not force a FAQ section when the campaign does not give enough real buyer questions. A useful how-it-works or objection section is better than fake FAQ content.
- Start each section that poses or implies a buyer question with the direct answer in the first paragraph, then expand. This supports answer extraction without making the page read like a blog post.
- If the campaign does not provide proof, testimonials, case studies, customer names, customer logos, or source ids, do not invent them. Use honest copy without fake social proof.
- Do not leave unresolved placeholders or draft notes such as `{{claim}}`, `TODO`, `TBD`, or `lorem ipsum`.

When the reasoning context provides a `narrative_plan`, copy each plan section's `id`/`title` verbatim and write prose grounded in the plan's `claim_ids`. Persona-tailor the headline + subheadline for `campaign.persona`.

Source-row evidence policy:
- If campaign evidence or opportunity context has `source_type: "support_ticket"`, `source_type: "complaint"`, `source_type: "case"`, or `source_type: "conversation"`, treat it as service evidence, not buying-intent evidence.
- Use service framing such as "support evidence points to..." or "complaint narratives describe..." and do not claim the target account is evaluating, buying, switching, or considering a vendor unless account-specific CRM, call, or meeting evidence supports it.
- If source rows are reviews, use market-evidence framing rather than naming exact review sites.

Avoid: weasel words ("powerful", "robust", "leverage"), promises that can't be backed up, comparative claims about specific competitors unless the campaign provides them.
