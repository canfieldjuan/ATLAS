# Content Ops Live Smoke Model Route

Issue #1299 requires a live end-to-end Content Ops smoke before more
generated-asset surface lands. That smoke must validate the model path the
product actually uses.

## Required Route

Content Ops generated content must run through the configured cloud LLM route:

- Provider path: OpenRouter via `PipelineLLMClient`.
- Current default model: `anthropic/claude-sonnet-4-5` unless the operator
  explicitly sets `EXTRACTED_CAMPAIGN_LLM_OPENROUTER_MODEL`.
- Credential source: the configured OpenRouter key.

Local Ollama/qwen is not an acceptable substitute for the live smoke. It is a
fallback path in the shared pipeline code, not the Content Ops validation
target.

## Fail-Closed Smoke Invocation

Run live validation with local fallback disabled:

```bash
EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false \
python scripts/smoke_content_ops_live_generation.py \
  --account-id <real-account-id> \
  --output blog_post \
  --support-ticket-csv \
  --export-saved-draft tmp/content_ops_live_blog_post_export.json \
  --evaluate-generated-content \
  --output-result tmp/content_ops_live_blog_post_result.json
```

If the cloud route is unavailable, the smoke should fail. Do not start Ollama
or substitute a local model to make the smoke pass.

## Deterministic Outputs

Deterministic outputs such as `quote_card` and `stat_card` still need real
Postgres validation and browser/export validation, but they are not evidence
that the LLM generation path works. The #1299 acceptance bar requires both:

- deterministic card path: real Postgres, tenant isolation, review/export, and
  real browser rendering for PNG;
- LLM path: real OpenRouter/Claude generation through `PipelineLLMClient`, with
  local fallback disabled.

## Reviewer Check

Before accepting a live-validation doc for #1299, confirm it names:

- provider/model actually used;
- local fallback disabled (`EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false`);
- no Ollama/qwen process used as substitute evidence;
- result artifacts for the real DB/model/browser run.
