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

This slice exceeds the 400-LOC soft cap because the parser change, the
fail-closed boundary matrix required for a review-admission guard, and this
repository-required plan form one indivisible safety change. Splitting the
recognizer from its negative proof would leave an unproved required gate;
splitting the plan would make either code PR inadmissible without reducing the
behavioral surface.

### Decision-Seam Analysis

- Shared decision: whether one trusted review-body line is admissible as the
  root decision when a following line contains complete rule evidence.
- Why it is wrong: the current admission predicate treats the absence of an
  enumerated malformed-suffix match as proof that the candidate is a title.
  Immediate suffixes are an open Unicode category, so an unlisted suffix can
  reach the admitting branch even though it is not complete rule evidence.
- Structural fix and default: recognize any immediate non-whitespace
  continuation after a complete rule reference, plus whitespace-separated
  rule-label operators, as potential evidence. Admit that candidate only when
  the existing complete-label grammar validates it; otherwise reject the
  adjacent pair. Ambiguity fails closed because a false rejection leaves the
  required check visibly red, while a false acceptance can clear unreconciled
  reviewer evidence.

### Problem-derived contract

- Root cause: the trusted complete rule-evidence grammar omitted an exact rule
  reference followed directly by `:` and nonempty detail, while adjacent-title
  admission also inferred validity from an enumerated malformed-suffix
  denylist. The producer form was therefore unparseable, and later repairs
  could still admit an unenumerated incomplete suffix.
- Correct fix must touch/change: extend only the shared complete-label grammar
  to admit `R<n>(/R<n>)*: nonempty detail`; prove multiline and inline positive
  correlation; scan candidates in source order so inline evidence on an earlier
  line keeps precedence, while a complete adjacent evidence line preserves the
  whole current title before any inline-like text inside that title can truncate
  it; recognize malformed fragments at line start and mid-line without splitting
  bare multi-digit or complete chained rule mentions; recognize incomplete slash
  continuations with immediate or whitespace-separated delimiters; classify
  immediate suffixes as an open non-whitespace category and evidence-gate them
  with the complete-label grammar; preserve existing dash and
  severity-qualified forms; and prove incomplete, malformed, Unicode-suffixed,
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
    a line that itself begins as a rule label or contains potential rule
    evidence that the complete grammar cannot validate cannot become a title;
    settled by
    `test_adjacent_rule_evidence_property_preserves_rule_like_colon_text_inside_titles`,
    the label-only negative case,
    `test_adjacent_rule_evidence_rejects_malformed_fragments_at_start_or_midline`,
    the existing incomplete-fragment matrix, the evidence-keyed product over
    rule tokens, title containers, and label families, and the generated
    all-Unicode immediate-suffix property.
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
  references cannot backtrack into shorter malformed fragments; any immediate
  non-whitespace continuation is structurally potential evidence and must pass
  the complete grammar; every unmatched form retains the empty-decision
  fail-closed result.
- Guard-relevant fields: bounded title length/token floor, exact chained rule
  reference, atomic longest-reference matching, the open immediate
  non-whitespace suffix category, optional delimiter whitespace, colon/slash
  delimiter, and required nonempty detail token.
- Caller x input shape: resolved trusted-bot history with the observed complete
  colon form can match only its named structured disposition; untrusted author,
  empty detail, malformed rule reference, short title, mismatched disposition,
  and unresolved thread remain rejected by their existing checks.

### Capability-set closure declaration

- Producer prose and malformed suffixes are OPEN. The accepted evidence grammar
  is CLOSED and ENUMERATED in `_COMPLETE_RULE_LABEL_RE`; the inline and
  adjacent-line matchers derive accepted forms from that grammar, while the
  potential-evidence recognizer treats every immediate non-whitespace suffix as
  requiring validation rather than enumerating rejected suffixes.
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
current whole bounded title unless it begins as a rule label or contains
potential rule evidence that the complete-label grammar cannot validate;
otherwise return the current bounded inline root. Prevent reference matching
from backtracking by atomically consuming the longest complete reference, then
treat every immediate non-whitespace continuation and every
whitespace-separated rule-label operator as potential evidence. The established
bounded-title floor and root matching remain downstream invariants, so
recognizing the delimiter does not make arbitrary or short prose authoritative.

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

- `python -m pytest tests/test_check_ai_reconciliation_live.py -q` ->
  `90 passed in 1.20s` after the structural decision-seam repair.
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
| `plans/PR-Live-Reconciliation-Severityless-Colon.md` | 241 |
| `scripts/check_ai_reconciliation_live.py` | 33 |
| `tests/test_check_ai_reconciliation_live.py` | 268 |
| **Total** | **542** |
