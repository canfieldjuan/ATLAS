# PR-Resolution-Audit-S6B-HTML-Line-Hygiene

## Why this slice exists

Issue #2043 (S6B), parent #1993. Rich helpdesk HTML stores message structure
(paragraph/break/list/table/heading/quote boundaries) in tags, but the single
extraction chokepoint flattens every tag boundary to a space, so line-based
hygiene downstream -- the S6C transcript state machine (#2044) and the S6E
junk/auto-reply gate (#2049, split from this slice per the launch-blocker
ledger scoping flag) -- can never work on HTML tickets. Three consequences
verified failing on `origin/main` before coding: signatures/questions flatten
into one run-on line; `blockquote` quoted-prior-message bodies are admitted as
customer text; and an unclosed `<script>` tag swallows the ENTIRE ticket body
(`support_ticket_plain_text` returns `""` -- silent data loss).

### Problem-derived contract

- Root cause: `_HTMLTextExtractor` emits a single space at every tag boundary
  and `_compact` collapses all whitespace, destroying the structure HTML
  encodes in tags; `blockquote` bodies are admitted as new customer text; the
  depth-based skip mechanism lets an unclosed `script`/`style` tag put the
  parser in CDATA mode and silently swallow all subsequent customer text.
- Correct fix must touch/change (all in
  `extracted_content_pipeline/support_ticket_clustering.py`, owned, the single
  extraction chokepoint, plus tests):
  1. Block-level tags -- members of the SAME `_HTML_TAG_NAMES_RE` families the
     HTML detector uses, so detection and extraction cannot drift -- emit
     newlines; inline tags keep emitting spaces.
  2. New public seam `support_ticket_plain_text_lines(value) -> str`:
     newline-preserving, per-line compacted, empty lines dropped; raw and
     escaped HTML route through it; plain text keeps its own newlines. This is
     the named input seam for S6C (#2044) and S6E (#2049).
  3. `blockquote` joins `script`/`style` as excluded content (the HTML-native
     quoted-prior-message marker).
  4. Robustness both directions: excluded bodies stay excluded when scopes
     close properly; an unclosed `script`/`style` scope recovers its buffered
     text tag-stripped at EOF instead of losing the ticket; an unclosed
     `blockquote` stays excluded (a quote to end-of-message is a quote, not
     data loss); self-closing skip tags cannot suppress later text.
  5. `support_ticket_plain_text` keeps its exact single-line compact output
     shape, now blockquote-excluded, with a non-empty fallback: if exclusion
     empties a previously non-empty body (all-quote ticket), keep the
     unexcluded extraction so no admitted row silently disappears.
  6. Tests: boundary families (p/br/div/li/h1-6/tr/section), escaped HTML,
     script/style/blockquote exclusion incl. nesting, all-quote fallback,
     unclosed + self-closing skip tags, near-miss guard (customer wording
     about out-of-office is untouched -- S6B strips nothing content-based),
     compact-API regression pins, and a drift-guard test asserting every
     block tag is inside the detector's tag families.
- Must not change:
  - `support_ticket_privacy.py` (S6A, merged), `support_ticket_input_package.py`
    admission/evidence logic, `support_ticket_zendesk_thread.py`.
  - The compact output SHAPE of `support_ticket_plain_text` for existing
    consumers (tokens, cluster keys, labels, titles).
  - Tokenization/clustering semantics beyond quoted-body exclusion.
  - No junk/auto-reply detection rules (S6E #2049); no scalar-history
    signature/quote state (S6C); no status/evidence-tier changes (S6D/M9).
  - Report/snapshot/email/PDF/landing/checkout/product shape; final-output
    scrubber grammar.
  - Goldens untouched (verified: full gauntlet green with no golden changes).

Round-1 review refinements (each verified failing first; two were blockers):
`script`/`style` join the detector tag families so script-only bodies are
excluded instead of passed through raw; EOF recovery re-extracts only the
markup after the first HTML signal in the buffer, so script/CSS machinery is
never admitted alongside rescued customer text (pure-machinery buffers recover
nothing); excluded-scope open/close boundaries follow the same block-tag rule
(an inline script inside a paragraph no longer splits the line); blockquote
bodies are no longer buffered (only recoverable script/style scopes are);
`support_ticket_plain_text_lines` exported via `__all__`.

Round-2 refinement (verified failing first): EOF recovery starts at the first
HTML signal OUTSIDE string literals and comments
(`_first_markup_outside_code_literals`), so script code embedding HTML
templates (`var t="<p>template</p>"`, backtick templates, commented markup,
escaped quotes) is never recovered as ticket text; unterminated literals run
to EOF and recover nothing.

Round-3 refinements (each verified failing first; one was a regression from
the round-1 detector change): lone `<script>`/`<style>` mentions in prose are
customer wording -- script/style detect only as a PAIRED open+close
(`_HTML_SCRIPT_STYLE_PAIRED_RE`), restoring "How do I add <script> to the
page?" while paired script-only bodies stay excluded; the code-literal
scanner also masks JS regex literals (expression-position slash, escapes,
division distinguished); recovery no longer invents a line boundary -- the
scope-open boundary and the recovered extraction's own boundaries are the
only ones emitted.

Round-4 refinements (each verified failing first): closed string/template
literals and closing braces are value positions, so division after them is
never misread as a regex open (no mask-to-EOF data loss); EOF recovery also
searches custom-element signals (`<x-ticket>`), matching `_looks_like_html`;
buffering/recovery key on the CDATA condition (top of skip stack is
script/style), so a malformed script nested inside a blockquote still
recovers the swallowed tail while quoted text before it stays excluded.

Round-5 refinements (each verified failing first; the scanner is now bounded
by JS lexical facts rather than heuristics): postfix `++`/`--` are value
positions (division); a slash inside a regex character class does not close
it; a regex candidate that reaches a newline was division all along (JS regex
literals cannot span lines), so a misread slash can never mask to EOF; a
slash after `<` is markup, never a regex; close tags swallowed as CDATA
unwind their scopes when the script closes (checked outside code literals),
so later real markup is not treated as still-quoted; `blockquote` joins the
paired-only detection family, so lone mentions in prose stay customer wording
while quote-to-EOF inside real HTML stays excluded; the drift-guard test
accepts either detector family.

Round-6 refinements (each verified failing first; the mention-vs-markup rule
is now POSITION-based, dissolving the paired-vs-unpaired contradiction across
rounds 3-6): an excluded tag at the START of the body is markup intent
(`<script>alert(1)` excludes, `<blockquote>quoted prior reply` is all-quote)
while mid-prose mentions -- lone OR paired -- are customer wording ("How do I
write <blockquote>hello</blockquote> in the editor?"); EOF recovery candidates
include excluded-tag positions so re-extraction opens the scope instead of
starting inside an excluded body; regex literals after expression keywords
(`return /.../;`) are masked via a bounded JS keyword set.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Vertical slice

Max files: 5

1. `extracted_content_pipeline/support_ticket_clustering.py` -- the extractor
   rewrite per the contract (owned file; manifest target-only).
2. `tests/test_support_ticket_plain_text_lines.py` -- the contract tests.
3. `scripts/run_extracted_pipeline_checks.sh` -- CI enrollment of the new test
   file in the same PR.
4. `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` -- the
   arc's standing rule: every remediation PR updates the tracker. Ticks S6A
   (#2046 merged), records the S6E/M9/S8a/S8b splits (#2049-#2052), resolves
   the ledger row-5 scoping flag.
5. This plan doc.

### Files touched

- `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
- `extracted_content_pipeline/support_ticket_clustering.py`
- `plans/PR-Resolution-Audit-S6B-HTML-Line-Hygiene.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_support_ticket_plain_text_lines.py`

### Review Contract

Acceptance criteria:
1. HTML variants in the detector's tag families reach line-based consumers
   with line breaks preserved (`support_ticket_plain_text_lines`).
2. Parser-excluded bodies (script/style/blockquote) stay excluded, including
   nested quotes; later valid customer text is never erased by self-closing
   or unclosed skip tags (the unclosed-`<script>` data-loss bug is fixed and
   pinned by test).
3. The compact `support_ticket_plain_text` output shape is unchanged for all
   existing pinned inputs; an all-quote body falls back instead of emptying a
   previously admitted row.
4. Customer wording about auto-reply/out-of-office features is untouched
   (S6B strips nothing content-based; detection rules are S6E #2049).
5. A drift-guard test binds every block tag to the detector's tag families.

Reachability proof: `support_ticket_plain_text` is the live chokepoint for
package/Zendesk text extraction -- blockquote exclusion and the unclosed-script
recovery change real admitted text on the existing paths today;
`support_ticket_plain_text_lines` is the named seam consumed by S6C/S6E next.

Affected surfaces: support-ticket text extraction feeding clustering,
evidence, and markdown generation.

Risk areas: over-exclusion (legit text in blockquote -- mitigated by the
all-quote fallback + tests), compact-shape regression (pinned), goldens
(gauntlet green, none changed).

Reviewer rules triggered: R1 requirements match (extracted package change), R2 test evidence, R10 guard/checker behavior, R14
sanitizer/parser admission change (boundary probed both directions).

## Mechanism

One extractor, two output shapes. `_HTMLTextExtractor` gains a block-tag set
(`_BLOCK_TAG_NAMES`, drift-bound to `_HTML_TAG_NAMES_RE`), an excluded-content
set (`_EXCLUDED_CONTENT_TAGS`), a skip STACK with a buffered `_pending` list
(instead of a bare depth counter), and a `finalize()` that recovers buffered
text tag-stripped when a script/style scope never closed. Block tags emit
`\n`, inline tags emit a space. `support_ticket_plain_text` compacts the
joined parts exactly as before (newlines collapse -- shape identical) and
falls back to unexcluded extraction when exclusion empties a non-empty body.
`support_ticket_plain_text_lines` splits on the emitted boundaries, compacts
each line, drops empties.

## Intentional

- Blockquote exclusion lands at the existing compact chokepoint TODAY (real
  behavioral change: quoted reply chains stop polluting clustering/evidence),
  not just in the new seam.
- The lines seam has NO all-quote fallback: an all-quote body is genuinely
  empty of new customer text for hygiene purposes.
- Unclosed-blockquote content stays excluded while unclosed-script content is
  recovered: the first is semantically a quote to EOF, the second is a parser
  artifact that was silently eating tickets.
- No junk-detection rules here: S6E (#2049) owns F2, sequenced right after.

## Deferred

- S6E (#2049): junk/auto-reply/OOO detection rules on the new line seam.
- S6C (#2044): scalar-history signature/quote state machine consuming the seam.
- S6D (#2045) evidence tier + M9 micro-slice (#2050).

## Verification

- `python -m pytest tests/test_support_ticket_plain_text_lines.py -q` (22 passed)
- Adjacent regression: `tests/test_extracted_support_ticket_input_package.py`,
  `tests/test_smoke_content_ops_support_ticket_package.py`,
  `tests/test_support_ticket_privacy.py`, `tests/test_support_ticket_privacy_sweep.py` (703 passed)
- scripts/run_extracted_pipeline_checks.sh full CI mirror (5772 passed, 21 skipped)
- Root-cause repro before coding: boundary flattening, blockquote admission,
  and the unclosed-script `""` data loss all reproduced on `origin/main`.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` | 14 |
| `extracted_content_pipeline/support_ticket_clustering.py` | 310 |
| `plans/PR-Resolution-Audit-S6B-HTML-Line-Hygiene.md` | 226 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_support_ticket_plain_text_lines.py` | 383 |
| **Total** | **934** |
