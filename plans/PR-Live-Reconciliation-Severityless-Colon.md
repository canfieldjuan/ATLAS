# PR-Live-Reconciliation-Severityless-Colon

## Why this slice exists

Owned Terms capability PR #2506 fixed and resolved its concrete Codex finding,
but the required trusted-base `live-reconciliation` check remains red. GitHub's
current review `bodyText` is a bounded title followed by `R2/R14: detail`.
`_COMPLETE_RULE_LABEL_RE` accepts dash-delimited and severity-qualified colon
evidence, but not this severity-less colon form, so the historical thread is
reported as permanently unparseable even when the PR body names its exact
decision. This CI-provider defect blocks the product slice and must be fixed at
the parser rather than bypassed in #2506.

### Problem-derived contract

- Root cause: the trusted rule-evidence grammar omits an exact rule reference
  followed directly by `:` and nonempty detail, although the trusted Codex
  connector now emits that observed form. A resolved finding therefore has no
  normalizable decision and cannot be honestly reconciled.
- Correct fix must touch/change: extend only the shared complete-label grammar
  to admit `R<n>(/R<n>)*: nonempty detail`; prove multiline and inline positive
  correlation; scan candidates in source order so inline evidence on an earlier
  line keeps precedence, while a complete adjacent evidence line preserves the
  whole current title before any inline-like text inside that title can truncate
  it; recognize malformed fragments at line start and mid-line without splitting
  bare multi-digit or complete chained rule mentions; recognize incomplete slash
  continuations with immediate or whitespace-separated delimiters; preserve
  existing dash and severity-qualified forms; and prove incomplete, malformed,
  short-title, label-only, and unrelated inputs still fail closed.
- Must not change: trusted bot identities, GitHub workflow triggers, open-thread
  blocking, current-head review freshness, PR-body disposition matching,
  docs-only handling, Terms capability code, Tracker code, or any customer,
  persistence, financial, authentication, or product behavior.

## Scope (this PR)

Ownership lane: workflow/live-reconciliation-severityless-colon
Slice phase: Workflow/process
Max files: 3

1. Add the observed severity-less colon-with-detail form to the trusted
   rule-label evidence grammar without accepting an empty or malformed label.
2. Pin both sides of the boundary in the focused live-reconciliation tests and
   rerun #2506's required trusted-base check after this provider merges.

### Review Contract

- Acceptance criteria:
  - The exact observed multiline producer shape
    `Exercise the deployed manifest entrypoint\nR2/R14: detail` yields only the
    bounded title and lets `evaluate()` correlate a matching `fixed-in`
    disposition for a resolved thread; settled by focused unit tests.
  - The same complete rule label works inline, while existing dash and
    severity-qualified colon forms retain their current roots; settled by
    parser boundary tests.
  - A following complete evidence line preserves the entire bounded title even
    when that title contains generated `R<n>(/R<n>)*: text` combinations, while
    a line that itself begins as a rule label or contains an incomplete
    rule-label fragment cannot become a title; settled by
    `test_adjacent_rule_evidence_property_preserves_rule_like_colon_text_inside_titles`,
    the label-only negative case,
    `test_adjacent_rule_evidence_rejects_malformed_fragments_at_start_or_midline`,
    and the existing incomplete-fragment matrix.
  - Bounded prefixes before rule-like colon text do not collapse distinct
    adjacent titles into one decision; settled by
    `test_adjacent_rule_evidence_keeps_bounded_prefix_titles_distinct`, including
    a negative one-disposition/two-title assertion.
  - Bare `R10` through `R14` mentions in adjacent titles remain ordinary title
    text rather than backtracking into malformed fragments; settled by
    `test_adjacent_rule_evidence_preserves_bare_multi_digit_rule_mentions`.
  - Bare complete chains such as `R4/R5` and `R10/R14` remain title text, while
    incomplete chains such as `R4/R`, `R4//R5`, `R4/Rfoo`, and `R4/R5/R` remain
    unparseable at line start and mid-line across immediate, space, tab, and
    mixed-whitespace delimiters; settled by the complete-chain test and generated
    malformed-fragment matrix.
  - Earlier complete inline evidence retains its decision even when later prose
    forms an otherwise valid adjacent title/evidence pair; settled by
    `test_earlier_inline_rule_evidence_precedes_later_adjacent_pairs`, including
    a negative assertion that the later prose cannot clear reconciliation.
  - `R2/R14:` with empty detail, `R2/R14   : detail` with pre-colon
    whitespace, `R2/R14foo: detail`, a short title, and text with no adjacent
    complete rule label yield no decision; settled by negative unit assertions.
  - The grammar remains the sole evidence choke point; no title allowlist,
    thread-state exception, or PR-number special case is added; settled by the
    cold diff.
- Reachability proof: `.github/workflows/ai_reconciliation_live.yml` continues
  to invoke `scripts/check_ai_reconciliation_live.py` from trusted base code;
  after merge, rerunning #2506 exposes the observable green/red gate result.
- Affected surfaces: the trusted live-reconciliation parser and its direct unit
  tests only.
- Risk areas: false reconciliation of malformed history, rejection of current
  trusted producer output, existing delimiter compatibility, and trusted-base
  deployment ordering.
- Reviewer rules triggered: R1, R2, R10, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: GitHub review `bodyText` ->
  `_evidenced_root_decision()` -> `_COMPLETE_RULE_LABEL_RE` ->
  `missing_thread_dispositions()` -> required gate result.
- Replaced-path behaviors: a bounded title adjacent to or sharing a line with
  `R<n>(/R<n>)*: nonempty detail` changes from unparseable to correlatable;
  source-order scanning preserves bounded inline evidence on an earlier line,
  while a complete following evidence line preserves the current whole title
  before rule-like colon text inside it can be truncated; complete multi-digit
  references cannot backtrack into shorter malformed fragments; every unmatched
  form retains the empty-decision fail-closed result.
- Guard-relevant fields: bounded title length/token floor, exact chained rule
  reference, atomic longest-reference matching, optional delimiter whitespace,
  colon/slash delimiter, and required nonempty detail token.
- Caller x input shape: resolved trusted-bot history with the observed complete
  colon form can match only its named structured disposition; untrusted author,
  empty detail, malformed rule reference, short title, mismatched disposition,
  and unresolved thread remain rejected by their existing checks.

### Capability-set closure declaration

- Producer prose is OPEN. The accepted evidence grammar is CLOSED and
  ENUMERATED in `_COMPLETE_RULE_LABEL_RE`; the inline and adjacent-line
  matchers derive their accepted forms from that one enumerated grammar.
- This slice adds one finite delimiter member: exact rule reference + colon +
  nonempty detail. It does not classify title vocabulary or accept generic
  prose as evidence.
- Outside-set inputs default to no decision, which keeps reconciliation red.

### Deployed-config probing

- Deployed/default config values: default trusted bot allowlist and workflow
  invocation are unchanged; this grammar has no environment toggle.
- Explicit value probe: the observed `R2/R14: detail` bodyText form returns the
  preceding bounded title.
- Absent value probe: missing or empty detail returns no decision.
- Default-session/default-context probe: direct `evaluate()` uses the default
  parser path with a resolved trusted-bot thread and matching PR-body ledger.
- Side-effect ordering: parser evaluation is pure; GitHub reads occur before
  evaluation exactly as before, and this diff adds no mutation.

### Files touched

- `plans/PR-Live-Reconciliation-Severityless-Colon.md`
- `scripts/check_ai_reconciliation_live.py`
- `tests/test_check_ai_reconciliation_live.py`

## Mechanism

Add the colon-with-detail alternative at the same complete-label grammar choke
point used by inline and adjacent-line parsing. Walk nonblank lines once in
source order: when the next line supplies complete evidence, preserve the
current whole bounded title unless it begins as a rule label or contains a
malformed fragment; otherwise return the current bounded inline root. Prevent
fragment matching from backtracking by atomically consuming the longest complete
reference, then treat any immediate or whitespace-separated leftover slash
continuation as malformed. The established bounded-title floor and root matching
remain downstream invariants, so recognizing the delimiter does not make
arbitrary or short prose authoritative.

## Intentional

- No PR-body workaround, thread deletion, bot-specific title allowlist, or
  special case for #2506. Those would hide the provider incompatibility rather
  than make the trusted producer and consumer contract agree.
- No acceptance of a colon without detail or of a rule-like prefix with trailing
  letters or an incomplete slash chain. The complete-evidence requirement
  remains fail closed.

## Deferred

None.

Parking predicate: producer delimiters not present in observed trusted review
`bodyText`, broader title grammar, bot-identity changes, and workflow-policy
changes remain outside this slice unless live evidence makes them a blocker.

Parked hardening: none.

## Verification

- `./ops test focused tests/test_check_ai_reconciliation_live.py -q` ->
  `88 passed in 0.63s` after the current-head review repairs.
- `/home/juan-canfield/miniconda3/bin/ruff check scripts/check_ai_reconciliation_live.py tests/test_check_ai_reconciliation_live.py`
  -> `All checks passed!`.
- `/home/juan-canfield/Desktop/Atlas/.venv/bin/python -m py_compile scripts/check_ai_reconciliation_live.py tests/test_check_ai_reconciliation_live.py`
  -> pass with no output.
- `/home/juan-canfield/Desktop/Atlas/.venv/bin/python scripts/check_guard_class_closure.py --base origin/main --strict`
  -> `OK: no guard-shaped change without a property test.`
- `/home/juan-canfield/Desktop/Atlas/.venv/bin/python scripts/sync_pr_plan.py --check plans/PR-Live-Reconciliation-Severityless-Colon.md origin/main`
  -> `plan already in sync` after the review repair.
- `git diff --check` -> pass with no output.
- `/home/juan-canfield/miniconda3/bin/ruff format --check scripts/check_ai_reconciliation_live.py tests/test_check_ai_reconciliation_live.py`
  reports existing whole-file formatter drift. Its diff was inspected and only
  branch-added assertion lines were aligned; unrelated baseline churn remains
  excluded.
- The repository's guarded local PR review passed on the first head. Rerun it
  through `scripts/push_pr.sh` after the review repair. The broad Unit Gate
  remains GitHub-owned.
- After provider merge, rerun and inspect #2506's trusted-base
  `live-reconciliation` check before resuming the Terms product slice.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Live-Reconciliation-Severityless-Colon.md` | 206 |
| `scripts/check_ai_reconciliation_live.py` | 31 |
| `tests/test_check_ai_reconciliation_live.py` | 228 |
| **Total** | **465** |
