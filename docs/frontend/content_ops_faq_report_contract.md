# Content Ops FAQ Report Contract

This is the frontend/demo handoff for generated support-ticket FAQ reports.
The canonical producer is
`extracted_content_pipeline.ticket_faq_markdown.TicketFAQMarkdownResult.as_dict()`.

Use this contract for Content Ops `faq_markdown` execute results, persisted FAQ
detail hydration, and landing-page demos that render the full generated FAQ
artifact.

Do not use the compact search result row as the full report. Search returns a
ranked match preview; detail hydration returns the generated report.

## Generated Report

```ts
type TicketFAQMarkdownResult = {
  generated: number;
  markdown: string;
  items: TicketFAQItem[];
  source_count: number;
  ticket_source_count: number;
  output_checks: {
    uses_user_vocabulary: boolean;
    condensed: boolean;
    has_action_items: boolean;
  };
  warnings: Array<Record<string, unknown>>;
  saved_ids: string[];
};
```

## FAQ Item

```ts
type TicketFAQItem = {
  topic: string;
  question: string;
  question_source: "customer_wording" | "source_policy";
  summary: string;

  frequency: number;
  weighted_frequency: number;
  ticket_count: number;
  opportunity_score: number;
  failure_risk_score: number;
  failure_risk_signals: string[];

  answer: string;
  steps: string[];
  action_items: string[];
  answer_evidence_status: "resolution_evidence" | "draft_needs_review";
  resolution_source_count: number;
  when_to_contact_support: string;

  evidence_quotes: string[];
  source_ids: string[];
  source_labels: string[];
  source_type_counts: Record<string, number>;
  weighted_source_volume_by_type: Record<string, number>;

  term_mappings: FAQTermMapping[];

  evidence_count: number;
  displayed_evidence_count: number;
};

type FAQTermMapping = {
  customer_term: string;
  documentation_term: string;
  suggestion: string;
  source_id_count: number;
  zero_result_source_count: number;
  failure_risk_score: number;
  failure_risk_signals: string[];
  opportunity_score: number;
  first_source_id: string;
};
```

## Persisted Detail

The detail route returns the persisted draft wrapper. Its `items` and `markdown`
fields are the same generated report artifact.

```ts
type TicketFAQDraft = {
  account_id: string;
  id: string;
  status: string;
  target_id: string;
  target_mode: string;
  title: string;
  markdown: string;
  items: TicketFAQItem[];
  source_count: number;
  ticket_source_count: number;
  output_checks: Record<string, boolean>;
  warnings: Array<Record<string, unknown>>;
  metadata: Record<string, unknown>;
};
```

## Compact Search Result

Search is intentionally smaller. Use `faq_id` from a selected result to hydrate
the full report detail.

```ts
type TicketFAQSearchResponse = {
  query: string;
  count: number;
  results: Array<{
    account_id: string;
    corpus_id: string;
    faq_id: string;
    target_id: string;
    target_mode: string;
    status: string;
    rank: number;
    topic: string;
    question: string;
    answer_summary: string;
    source_ids: string[];
    ticket_count: number;
    score: number;
  }>;
};
```

## Rendering Guidance

- Use `items` for ranked cards or sections, and `markdown` for the full report.
- Show `source_count`, `ticket_source_count`, and `output_checks` as proof badges.
- Show `answer_evidence_status` near steps. `draft_needs_review` means the
  system found repeated customer wording but no uploaded resolution evidence, so
  the generated steps are review placeholders.
- Show `term_mappings` as vocabulary-gap suggestions. These are grounded in the
  supplied documentation terms/rules and customer wording.
- Treat `source_ids` and `evidence_quotes` as proof links/footnotes, not primary
  marketing copy.

See [`content_ops_faq_report_example.json`](./content_ops_faq_report_example.json)
for a current compact example.
