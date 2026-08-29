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
- Evidence ordering and lexical boundary: an existing severity/dash inline
  label is direct evidence on the current line and must win before a following
  label is considered. The new severity-less colon form is admitted only as a
  line-start adjacent label because inline `R<n>: text` is indistinguishable
  from rule-like title prose. Potential evidence starts at line start or after
  any Unicode non-alphanumeric boundary; alphanumeric-embedded text remains an
  ordinary word. Ambiguous severity-less inline input takes the fail-closed
  path.
- Leading continuation rule: when the first lexical token of a candidate line
  is a rule reference and nonblank whitespace-delimited text follows, the
  complete-label grammar must validate the line. Leading Unicode punctuation
  does not turn that ambiguous evidence into a title; genuinely embedded bare
  references after title text remain ordinary content.
- Canonical identity rule: complete rule IDs come from the repository's defined
  reviewer-rule set in `docs/REVIEWER_RULES.md` (currently `R1` through `R14`),
  without zero padding. The checker derives its complete-reference grammar from
  one explicit identity tuple, and a test keeps that tuple synchronized with the
  rule headings. The broader potential-evidence recognizer still detects
  ASCII out-of-set values and Unicode-decimal lookalikes, but they cannot satisfy
  the complete grammar. A complete-looking label whose rule token is wrapped in
  leading punctuation is likewise not canonical line-start evidence and cannot
  become a title.

### Problem-derived contract

- Root cause: the trusted complete rule-evidence grammar omitted an exact rule
  reference followed directly by `:` and nonempty detail, while adjacent-title
  admission also inferred validity from an enumerated malformed-suffix
  denylist. The producer form was therefore unparseable, and later repairs
  could still admit an unenumerated incomplete suffix.
- Correct fix must touch/change: extend the line-start complete-label grammar
  to admit `R<n>(/R<n>)*: nonempty detail`; prove adjacent positive correlation
  while keeping the severity-less inline form fail closed; scan candidates in
  source order so existing severity/dash inline evidence on the current line
  keeps precedence, while a complete adjacent evidence line preserves a title
  containing colon-like text that is not an admitted inline label; recognize
  malformed fragments at line start and after any Unicode non-alphanumeric
  lexical boundary without splitting
  bare multi-digit or complete chained rule mentions; recognize incomplete slash
  continuations with immediate or whitespace-separated delimiters; classify
  immediate suffixes as an open non-whitespace category and evidence-gate them
  with the complete-label grammar; require a leading rule reference with any
  nonblank whitespace continuation to pass that same grammar; preserve
  existing dash and severity-qualified forms; derive complete rule references
  from the defined reviewer-rule identities while retaining broad numeric
  malformed detection; prove the derived identity tuple agrees with the
  reviewer-rule headings; and prove zero, zero-padded, out-of-range,
  Unicode-suffixed, punctuation-wrapped, unknown-leading-continuation,
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
  - Severity-less colon evidence remains unambiguous by being line-start only;
    the same form inline yields no root, while existing dash and
    severity-qualified inline forms retain their current roots and take
    precedence over an immediately following complete label; settled by parser
    ordering and negative substring-disposition tests.
  - A following complete evidence line preserves the entire bounded title even
    when that title contains generated `R<n>(/R<n>)*: text` combinations, while
    a line that itself begins as a rule label or contains potential rule
    evidence that the complete grammar cannot validate cannot become a title;
    settled by
    `test_adjacent_rule_evidence_property_preserves_rule_like_colon_text_inside_titles`,
    the label-only negative case,
    `test_adjacent_rule_evidence_rejects_malformed_fragments_at_start_or_midline`,
    the existing incomplete-fragment matrix, the evidence-keyed product over
    rule tokens, title containers, and label families, the generated
    all-Unicode immediate-suffix property, and the generated Unicode lexical
    boundary oracle.
  - A rule reference that is the first lexical token cannot become a title by
    adding an unknown whitespace-delimited continuation, including after
    leading Unicode punctuation; complete dash/severity labels retain their
    grammar result, and embedded bare references remain title text. Settled by
    the leading-continuation end-to-end negative and generated Unicode
    continuation oracle.
  - Only reviewer-rule IDs defined by `docs/REVIEWER_RULES.md` are complete
    evidence. `R1`, `R14`, and defined chains remain accepted; zero,
    zero-padded, and out-of-range ASCII IDs plus every non-ASCII Unicode decimal
    used as a whole or mixed rule ID remain malformed across colon, dash, and
    severity label forms. A colon label after
    leading punctuation likewise remains a no-decision line even when the
    embedded token otherwise matches. Settled by the rule-registry synchronization
    assertion, defined-edge positives, generated ASCII outside-set negatives,
    all-Unicode suffix oracle, and punctuation-wrapped colon end-to-end negative.
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
  source-order scanning preserves existing direct inline evidence before an
  immediately following complete label, while a complete following evidence
  line preserves the current whole title when colon-like text is not an
  admitted inline form; severity-less inline evidence remains unparseable;
  complete multi-digit
  references cannot backtrack into shorter malformed fragments; any immediate
  non-whitespace continuation is structurally potential evidence and must pass
  the complete grammar; every unmatched form retains the empty-decision
  fail-closed result.
- Guard-relevant fields: bounded title length/token floor, exact chained rule
  reference from the defined `R1`-`R14` identity set, broader numeric potential
  reference, atomic longest-reference matching, Unicode non-alphanumeric token
  boundary, leading punctuation, the open immediate non-whitespace suffix category, optional
  delimiter whitespace, first-lexical-token continuation state, colon/slash
  delimiter, and required nonempty detail token.
- Caller x input shape: resolved trusted-bot history with the observed complete
  colon form can match only its named structured disposition; untrusted author,
  empty detail, malformed rule reference, short title, mismatched disposition,
  and unresolved thread remain rejected by their existing checks.

### Capability-set closure declaration

- Producer prose and malformed suffixes are OPEN. The accepted line-start
  evidence grammar is CLOSED and ENUMERATED in `_COMPLETE_RULE_LABEL_RE`; the
  inline matcher intentionally uses the pre-existing severity/dash subset so
  colon-like title prose cannot become ambiguous inline evidence. The
  potential-evidence recognizer treats every immediate non-whitespace suffix at
  a Unicode lexical boundary as requiring validation rather than enumerating
  rejected suffixes or boundary punctuation. It also treats every nonblank
  whitespace continuation after a leading rule reference as potential evidence;
  this finite structural position distinguishes it from embedded bare mentions.
- Canonical complete references are derived from the explicit reviewer-rule
  identity tuple, which is synchronized by test with `docs/REVIEWER_RULES.md`.
  Zero, zero-padded, out-of-range ASCII, and Unicode-decimal candidates are
  deliberately part of the broader potential-reference recognizer so they are
  rejected by evidence validation rather than ignored as ordinary title text.
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

Add the colon-with-detail alternative to the line-start complete-label grammar
while retaining the pre-existing severity/dash subset for inline extraction.
Walk nonblank lines once in source order: return direct legacy inline evidence
before considering the following line; otherwise, when that next line supplies
complete evidence, preserve the current whole bounded title unless it begins as
a rule label or contains potential rule evidence that the complete-label grammar
cannot validate. Atomically consume the longest complete reference and detect
it at line start or after any Unicode non-alphanumeric boundary, then treat every
immediate non-whitespace continuation and every whitespace-separated rule-label
operator as potential evidence. If a rule reference is the first lexical token,
any nonblank whitespace continuation also requires complete-grammar validation.
The leading-token check rejects any punctuation wrapper, and the complete
grammar is generated from the defined reviewer-rule identity tuple while the
potential recognizer remains Unicode-wide. A synchronization test makes rule-doc
growth fail visibly until that tuple and its admission proof are updated.
The established bounded-title floor and root matching remain downstream
invariants, so recognizing the delimiter does not make arbitrary or short prose
authoritative.

## Intentional

- No PR-body workaround, thread deletion, bot-specific title allowlist, or
  special case for #2506. Those would hide the provider incompatibility rather
  than make the trusted producer and consumer contract agree.
- No acceptance of a colon without detail or of a rule-like prefix with trailing
  letters or an incomplete slash chain. The complete-evidence requirement
  remains fail closed.
- No severity-less colon inline admission. That unobserved shape is ambiguous
  with title prose; only the observed line-start producer form is added.

## Deferred

None.

Parking predicate: producer delimiters not present in observed trusted review
`bodyText`, broader title grammar, bot-identity changes, and workflow-policy
changes remain outside this slice unless live evidence makes them a blocker.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_check_ai_reconciliation_live.py -q` ->
  `99 passed in 4.47s` after the defined-identity repair.
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
| `plans/PR-Live-Reconciliation-Severityless-Colon.md` | 310 |
| `scripts/check_ai_reconciliation_live.py` | 65 |
| `tests/test_check_ai_reconciliation_live.py` | 482 |
| **Total** | **857** |
