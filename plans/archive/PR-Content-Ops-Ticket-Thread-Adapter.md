# Content Ops Ticket Thread Adapter

## Why This Slice Exists

Support exports often arrive as ticket rows with nested comments or messages,
not as a single `message` or `description` string. AI Content Ops can now
accept support-ticket rows, but it still skips useful ticket threads when the
primary text lives inside an array.

## Scope

- Extend `extracted_content_pipeline/campaign_source_adapters.py` to build
  evidence text from nested ticket/conversation message arrays.
- Add focused source-adapter tests for comment/message arrays and scalar
  precedence.
- Refresh host-facing source-row docs and coordination state.

## Mechanism

- Add recognized thread keys such as `messages`, `comments`, `thread`, and
  `conversation`.
- Use scalar source text first; only fall back to thread arrays when scalar
  fields are missing.
- Convert message dictionaries into newline-separated text with optional
  speaker labels from `speaker`, `author`, `role`, or `name`.
- Use message-shaped keys before generic `body`/`content` keys inside nested
  thread items; row-level scalar source text keeps the existing document/review
  precedence.
- Keep the existing `max_text_chars` truncation in `_source_evidence`.

## Intentional

- No new CLI flags.
- No database, API, or generated-asset changes.
- No attachment parsing.
- No multi-message reasoning or summarization; this is deterministic
  source-to-evidence normalization only.

## Deferred

- Attachment extraction.
- Thread-aware source-quality scoring.
- LLM summarization over long conversations before opportunity generation.

## Verification

- Focused source-adapter tests.
- Compile check for touched Python files.
- Local PR review gate.

### Files Touched

- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `plans/PR-Content-Ops-Ticket-Thread-Adapter.md`
- `tests/test_extracted_campaign_source_adapters.py`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Source adapter | ~45 |
| Tests | ~45 |
| Docs and coordination | ~30 |
| Plan doc | ~55 |
| **Total** | ~175 |
