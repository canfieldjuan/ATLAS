# Content Ops FAQ 1,000-Row Run - 2026-05-21

## Summary

The deterministic FAQ Markdown generator can process 1,000 local CFPB complaint
narrative rows without a performance failure. The run completed in about 0.5s
with roughly 23 MB maximum resident memory.

The initial run did surface output-quality failures. The first fail-closed CLI
run did not pass `--require-output-checks`; the generator failed `condensed` and
`uses_user_vocabulary`. A second run with a higher `max_items` cap passed
`condensed`, which proved the first condensed failure was caused by truncating
low-volume groups. `uses_user_vocabulary` still failed because several generated
questions used topic fallback wording.

After the fixes in this PR, the same 1,000-row source JSONL passed the
fail-closed command with `--max-items 12`. The fixed run completed in about
0.5s with roughly 23 MB maximum resident memory and all output checks true.

## Source Data

- Archive: `/home/juan-canfield/Downloads/archive (1)/rows.csv`
- Header: CFPB public complaint CSV fields including `Consumer complaint narrative`
  and `Complaint ID`
- Archive line count: `1,970,829` including header
- Extraction output: `tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl`
- Extracted source rows: `1,000`
- Archive rows scanned to get 1,000 narrative rows: `45,036`
- First extracted source ID: `cfpb:3189109`
- Last extracted source ID: `cfpb:3168734`

## Commands

Archive check:

```bash
head -1 '/home/juan-canfield/Downloads/archive (1)/rows.csv'
wc -l '/home/juan-canfield/Downloads/archive (1)/rows.csv'
```

Extraction used `cfpb_row_to_source_row` from
`scripts/export_content_ops_cfpb_sources.py` and wrote:

```text
tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl
tmp/content_ops_faq_1000/extract_summary.json
```

Fail-closed FAQ run:

```bash
/usr/bin/time -v python scripts/build_extracted_ticket_faq_markdown.py \
  tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl \
  --source-format jsonl \
  --title 'CFPB 1,000 Row FAQ Scale Run' \
  --max-items 12 \
  --max-evidence-per-item 5 \
  --max-text-chars 1200 \
  --require-output-checks \
  --output tmp/content_ops_faq_1000/cfpb_1000_faq.md
```

Result:

```text
FAQ output checks failed: condensed, uses_user_vocabulary
Elapsed wall time: 0:00.50
Maximum resident set size: 24036 KB
Exit status: 1
```

Inspection run with `max_items=12` wrote:

```text
tmp/content_ops_faq_1000/cfpb_1000_faq_unchecked.md
tmp/content_ops_faq_1000/cfpb_1000_faq_result.json
```

Result summary:

```json
{
  "source_count": 1000,
  "ticket_source_count": 1000,
  "generated": 12,
  "output_checks": {
    "condensed": false,
    "has_action_items": true,
    "uses_user_vocabulary": false
  },
  "ticket_counts": [633, 166, 113, 57, 9, 9, 3, 3, 1, 1, 1, 1]
}
```

Inspection run with `max_items=50` wrote:

```text
tmp/content_ops_faq_1000/cfpb_1000_faq_max50.md
tmp/content_ops_faq_1000/cfpb_1000_faq_max50_result.json
```

Result summary:

```json
{
  "source_count": 1000,
  "ticket_source_count": 1000,
  "generated": 15,
  "output_checks": {
    "condensed": true,
    "has_action_items": true,
    "uses_user_vocabulary": false
  },
  "question_source_counts": {
    "customer_wording": 7,
    "source_policy": 3,
    "topic_fallback": 5
  },
  "ticket_counts": [633, 166, 113, 57, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1]
}
```

Fixed fail-closed run after generator changes:

```bash
/usr/bin/time -v python scripts/build_extracted_ticket_faq_markdown.py \
  tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl \
  --source-format jsonl \
  --title 'CFPB 1,000 Row FAQ Scale Run' \
  --max-items 12 \
  --max-evidence-per-item 5 \
  --max-text-chars 1200 \
  --require-output-checks \
  --output tmp/content_ops_faq_1000/cfpb_1000_faq_fixed.md
```

Fixed result:

```text
Elapsed wall time: 0:00.53
Maximum resident set size: 23856 KB
Exit status: 0
```

Fixed result summary:

```json
{
  "source_count": 1000,
  "ticket_source_count": 1000,
  "generated": 12,
  "output_checks": {
    "uses_user_vocabulary": true,
    "condensed": true,
    "has_action_items": true
  },
  "question_source_counts": {
    "source_policy": 10,
    "customer_wording": 2
  },
  "ticket_counts": [633, 166, 108, 57, 9, 9, 5, 4, 4, 2, 1, 2]
}
```

## Issues Surfaced

### FAQ1000-1 - Default `max_items=12` can fail condensed coverage

The 1,000-row sample grouped into 15 topics. Running with `max_items=12`
generated only 12 FAQ groups and failed `condensed` because not all source IDs
were represented. Raising `max_items` to 50 generated 15 groups and passed
`condensed`.

Impact: a 1,000-row customer upload can process quickly, but the default item cap
can make the output gate fail when the row set has more distinct issue groups
than the cap.

Resolution: fixed in this PR. When issue groups exceed `max_items`, the
generator now keeps the highest-volume groups and folds the tail into one
`other support issues` item. The fixed 1,000-row run generated 12 FAQ items,
represented all 1,000 source rows, and passed `condensed`.

### FAQ1000-2 - `uses_user_vocabulary` fails on singleton topic-fallback groups

With `max_items=50`, five low-volume singleton groups used `topic_fallback`
questions:

- `advertising`
- `email and profile updates`
- `getting a credit card`
- `login and account access`
- `problem with a lender or other company charging your account`

Impact: the generator is correctly fail-closed by its current contract, but the
current behavior means a large batch with singleton tail issues can fail even
when the high-volume groups are usable.

Resolution: fixed in this PR. When customer wording is unavailable or rejected,
the generator now emits a source-policy question instead of a topic fallback
question. The fixed 1,000-row run produced 10 `source_policy` questions and 2
`customer_wording` questions, so `uses_user_vocabulary` passed.

### FAQ1000-3 - Some extracted customer-wording questions are malformed

Examples from the generated output:

- `What should I do if I paid the XX/XX/2019 credit card installment of {$100?`
- `I am not " allowed '' to speak to a human?`
- `What should I do if I received a letter dated XX/XX/XXXX, signed by Mr?`
- `What should we do if our company " XXXX XXXX '' was issued a money order from " XXXX XXXX '' / US Bank?`

Impact: the extraction can technically count as `customer_wording`, but the
human-facing question is not publish-ready. CFPB redactions, quoted fragments,
dates, names, and dollar amounts need stronger cleanup before this can be used as
a production proof point.

Resolution: fixed in this PR for the surfaced shapes. Customer-question
extraction now rejects redaction tokens, redacted dates, redacted money amounts,
quote artifacts, unbalanced quote marks, and trailing title fragments. Rejected
candidates fall back to source-policy questions.

### FAQ1000-4 - Some action steps are mismatched to CFPB banking topics

Rows grouped under `reporting friction` and `opening an account` received SaaS
report/export instructions:

```text
Open the reporting or analytics area and choose the date range you need.
Look for an Export or Download option...
```

Impact: the generic intent/action rules can misclassify CFPB financial complaint
language as SaaS reporting friction. That is a real correctness issue for the
public-support-ticket demo path.

Resolution: fixed in this PR for the surfaced rows. CFPB/banking action and
escalation rules now run before the generic SaaS reporting/profile guidance, and
the reporting export rule no longer matches the standalone word `report`. A
targeted grep of the fixed artifact found no `Export or Download`, `Open the
reporting or analytics`, or `export is missing` guidance.

### FAQ1000-5 - Narrative density matters for local archive extraction

The extractor scanned 45,036 archive rows to obtain 1,000 rows with usable
narratives.

Impact: generation itself is fast, but source preparation from the raw archive is
sparse. Any repeatable 1,000-row smoke should report both scanned rows and usable
rows so the run cannot hide source-density cost.

## Current Read

This run supports the claim that the deterministic generator can ingest, group,
and render 1,000 source rows locally with the fail-closed output checks enabled.
It does not prove arbitrary CFPB-derived batches are perfect; it proves the
specific scale failure modes surfaced by this run were acknowledged, fixed, and
rerun against the same 1,000-row source file.
