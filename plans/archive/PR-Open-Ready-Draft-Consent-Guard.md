# PR-Open-Ready-Draft-Consent-Guard

## Why this slice exists

The AGENTS mechanical-enforcement audit found that PRs are supposed to open
ready for review by default, but `open_pr.sh` still forwards `--draft` without
any explicit operator-consent signal. Draft PRs delay automated review and have
already cost this lane manual attention.

Diff budget: the slice exceeds the 400 LOC soft cap (622 actual) because five
Codex review rounds required class-closure work rather than spot fixes -- the
pflag-faithful argv-grammar walk, the admission-before-side-effects ordering,
a grammar-derived property suite with an independent oracle, and the plan's
closure/R11 declarations. The runtime product surface is one wrapper script;
over half the diff is tests and plan contract.

### Problem-derived contract

- Root cause: The PR mutation wrapper rejects target-changing args, but it does
  not classify draft mode as a consent-gated create option, so a builder can
  accidentally open a draft PR even though AGENTS says ready-for-review is the
  default.
- Correct fix must touch/change: `scripts/open_pr.sh` must reject every draft
  flag spelling `gh pr create` accepts -- `--draft`, `--draft=<value>`, `-d`,
  `-d=<value>`, and shorthand clusters that enable draft in any position
  (`-dw`, `-fd`, `-wd`, `-fd=true`) -- unless an explicit operator-consent
  environment flag is present. Value-taking shorthands whose attached value
  merely contains a `d` (for example `-tdraft-note`) must not be gated.
  `tests/test_open_pr_wrapper.py` must prove reject-by-default happens before
  any GitHub mutation for each spelling class and prove the explicit flag
  forwards draft mode when the operator intentionally allows it.
- Must not change: Do not alter PR body validation, target repo/base/head
  safety, local review ordering, existing PR edit behavior, branch protection,
  or any product/runtime code.

## Scope (this PR)

Ownership lane: dev-workflow/open-ready-draft-consent
Slice phase: Workflow/process

1. Add an explicit draft-consent gate to `open_pr.sh` create-argument parsing.
2. Add wrapper tests for rejected draft creation by default and allowed draft
   creation with the explicit consent flag.

### Review Contract

- Acceptance criteria:
  - `scripts/open_pr.sh` rejects `--draft`, `--draft=<value>`, `-d`,
    `-d=<value>`, and shorthand clusters containing `d` in any boolean
    position (`-dw`, `-fd`, `-wd`, `-fwd`, `-fd=true`) before GitHub mutation
    when `ATLAS_OPEN_PR_DRAFT_CONSENT` is not set to `1`.
  - `scripts/open_pr.sh` does not gate value-taking shorthands whose attached
    value contains `d` (`-tdraft-note` forwards without consent), nor
    draft-shaped tokens consumed as a separate option value (`--title --draft`
    and `-t -d` forward without consent, matching gh's grammar).
  - `scripts/open_pr.sh` rejects an unauthorized draft invocation before any
    side effect: with origin unreachable, `--draft` still fails on consent,
    not on fetch.
  - `scripts/open_pr.sh` forwards `--draft` and `--draft=true` to
    `gh pr create` when `ATLAS_OPEN_PR_DRAFT_CONSENT=1` is set.
  - `scripts/open_pr.sh` exports `GH_PROMPT_DISABLED=1` so gh's interactive
    create survey (which offers "Submit as draft" with no argv token) is
    unreachable through the wrapper.
  - `scripts/open_pr.sh` rejects the browser create route (`--web`,
    `--web=<value>`, and clusters whose walk reaches a boolean `w`) even when
    draft consent is set, because the web UI can submit a draft with no argv
    token and escapes post-mutation verification.
  - The consent decision matches an independent pflag oracle across a
    generated product of boolean-shorthand positions x value-taking
    terminators x attached/separate/`=` values
    (`test_open_pr_draft_admission_matches_gh_argv_grammar`).
  - Existing safe ready-for-review create and edit flows continue to pass.
- Reachability proof: `tests/test_open_pr_wrapper.py` invokes the real wrapper
  script against a fake `gh`; observable effects are process exit code, stderr,
  and captured `gh pr create` argv/stdin. Class closure of the argv grammar is
  proven by a grammar-derived property test whose expected verdicts come from
  an independent Python model of gh's pflag walk, not from the shell code
  under test.
- Affected surfaces: `scripts/open_pr.sh` create-argument boundary and
  `tests/test_open_pr_wrapper.py` fixtures.
- Risk areas: create-argument parsing, accidental draft mode, consent flag
  misuse, existing PR edit/create wrapper regression.
- Reviewer rules triggered: R1, R2, R6, R10, R11, R12, R13, R14.
- R11 configuration disposition: `ATLAS_OPEN_PR_DRAFT_CONSENT` is a
  per-invocation operator consent flag for a local dev-workflow wrapper, not a
  runtime `atlas_brain/config.py` setting. Default is unset, which refuses
  drafts. It is documented in the wrapper's usage text (the only operator
  surface for this script), and it must not be exported persistently in
  `.env`, shell profiles, or CI -- persisting it would convert a one-time
  exception into standing consent, defeating the gate.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/open_pr.sh` create-argument admission.
- Replaced-path behaviors: `--draft`, `--draft=<value>`, and every shorthand
  cluster whose pflag walk reaches a boolean `d` (`-d`, `-d=<value>`, `-dw`,
  `-fd`, `-wd`) no longer pass through by default; value-attached shorthands
  like `-tdraft-note` still do.

### Closure declaration

Set-valued dependencies in the admission decision, per
`docs/GUARD_CLASS_CLOSURE.md` (each answers closed/open, sourcing, and the
outside-set direction):

- Draft-flag spelling set (`--draft`, `--draft=<value>`, shorthand `d`):
  CLOSED with respect to the installed gh CLI (2.96.0), whose help documents
  exactly `-d, --draft` as the draft flag. Sourcing is ENUMERATED from
  `gh pr create --help` at authoring time; a bash wrapper has no runtime
  derivation point against gh's flag table, so a future gh alias for draft
  would not be auto-detected.
- Value-taking option inventory (long options with a separate value token:
  `--assignee --label --milestone --project --reviewer --title --template
  --recover`, plus the dedicated admission arms for `--base --head --repo
  --body --body-file`; shorthand letters: `a B b F H l m p r R t T`): CLOSED
  with respect to gh 2.96.0. Sourcing is ENUMERATED from
  `gh pr create --help` at authoring time; ENUMERATED is the honest answer
  because the wrapper cannot recompute gh's flag table per run and therefore
  cannot notice upstream drift.
- Caller x input shape: CLOSED. The only caller is a local builder running
  `bash scripts/open_pr.sh BODY_FILE [gh-pr-create-args...]`, and every
  scanned token is forwarded to exactly one consumer, `gh pr create`.

Outside-the-set direction (required even for CLOSED sets): every token the
inventory does not recognize flows to the consent-gated side. An unknown long
option is not treated as value-taking, so a `--draft`-shaped token after it is
still gated; an unknown shorthand letter scans on as a boolean, so a `d` after
it is still gated. Inventory incompleteness therefore produces over-rejection
(a ready invocation asked for consent it did not need -- cheap, because the
operator reruns with the flag or without the token) and never an unconsented
draft PR, which is the failure this slice exists to prevent. The one residual
drift direction -- a future gh release demoting a currently value-taking flag
to boolean, re-opening a cross-token spelling -- is accepted under ENUMERATED
sourcing and named in Deferred.
- Guard-relevant fields: wrapper argv and `ATLAS_OPEN_PR_DRAFT_CONSENT`.
- Caller x input shape: local builder running `bash scripts/open_pr.sh
  BODY_FILE [gh-pr-create-args...]`.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: `ATLAS_OPEN_PR_DRAFT_CONSENT` defaults absent.
- Explicit value probe: test sets `ATLAS_OPEN_PR_DRAFT_CONSENT=1` and proves
  draft mode is forwarded.
- Absent value probe: tests omit the flag for `--draft` and `-d` and prove the
  wrapper exits before fake `gh` is invoked.
- Default-session/default-context probe: existing ready create/edit wrapper
  tests continue to run without the flag.
- Side-effect ordering: draft rejection happens in argument admission before
  base refresh, local review, or GitHub mutation.

### Files touched

- `plans/PR-Open-Ready-Draft-Consent-Guard.md`
- `scripts/open_pr.sh`
- `tests/test_open_pr_wrapper.py`

## Mechanism

Argument admission (`--body`-family rejection plus `reject_target_overrides`)
runs before `refresh_base_ref` and the body audit, so a rejected invocation
produces no side effect -- no fetch, no ref update, no GitHub call.

`reject_target_overrides` treats every draft-flag spelling as a consent-gated
create argument by modeling gh's argv grammar. Long forms match
`--draft|--draft=*` directly. Long value-taking options with a separate value
token (`--assignee --label --milestone --project --reviewer --title
--template --recover`, plus the existing `--base` handling) consume their
next token, so `--title --draft` is a title value, not a draft flag. `--`
ends option parsing. Every other one-dash token goes through
`scan_shorthand_cluster`, which mirrors gh's pflag cluster walk: boolean
shorthands keep scanning (so `-fd` and `-wd` enable draft just like `-dw`), a
value-taking shorthand (`-a -B -b -F -H -l -m -p -r -R -t -T`) takes the
attached remainder as its value (`-tdraft-note` is a title) or, when nothing
is attached, consumes the next argv token (`-t -d` is a title value), and `=`
binds the remainder to the shorthand before it. Unknown letters scan on as
booleans, which fails closed. The gate is value-blind on purpose:
`--draft=false` and `-d=false` are also held behind consent rather than
parsing pflag's six truthy spellings, since ready-for-review is already the
default and fail-closed is simpler. Without `ATLAS_OPEN_PR_DRAFT_CONSENT=1`,
`require_draft_consent` prints a targeted error and exits before any GitHub
call. With the flag set, the wrapper leaves the argument in place so the
existing `gh pr create` call can intentionally create a draft PR.

Three structural guarantees close the remaining entry points: the wrapper
exports `GH_PROMPT_DISABLED=1`, so gh's interactive create survey -- whose
"Submit as draft" action carries no argv token -- is unreachable and draft
mode can only arrive through the gated argv path; `reject_web_create`
refuses the browser route (`--web`, `--web=<value>`, and clusters whose walk
reaches a boolean `w`) even under draft consent, because GitHub's create UI
can submit a draft with no argv token and the web flow also escapes this
wrapper's post-mutation head/body verification; and
`test_open_pr_draft_admission_matches_gh_argv_grammar` proves grammar closure
by generating the product of boolean-shorthand positions, value-taking
terminators, and attached/separate/`=` value bindings, then asserting the
wrapper's admission decision (consent-gate, web-reject, or pass) equals an
independent Python oracle modeling gh's pflag walk for every generated
sequence. Creating a new PR requires `--title` or a gh fill option by
pre-existing necessity, not as a change here: the wrapper binds stdin to the
body file, so gh has never been able to prompt for a title through it;
`GH_PROMPT_DISABLED` makes that non-interactive contract explicit and the
usage text now states it.

## Intentional

- The consent signal is an environment flag, not a new persistent session-state
  parser; it keeps this slice small and makes the exceptional draft request
  explicit at the command boundary.
- The wrapper still opens ready PRs by default; no extra flag is needed for the
  normal path.

## Deferred

- gh CLI flag-table drift: the value-taking inventory is enumerated from
  gh 2.96.0. A future gh release that changes an existing flag's arity
  (value-taking to boolean) would re-open a cross-token spelling and requires
  re-auditing the inventory; no runtime derivation point exists in a bash
  wrapper. Unlisted-member drift in the other direction only over-rejects
  (see Closure declaration).
- Non-leading cluster spellings of the pre-existing `-H`/`-R`/`-B`
  target-override gates (e.g. `-fHother`) predate this slice and belong to a
  separate target-override hardening follow-up; this slice's argv-grammar
  walk gates draft mode and the browser route only.

Parking predicate: this slice parks, by default, hardening of `gh pr create`
surfaces other than ready-for-review admission through this wrapper --
non-draft flag hygiene, target-override cluster spellings, and gh-version
drift tooling. Findings inside the draft/web admission path itself are owned
by this slice and were fixed in-review, not parked.

Parked hardening: none against that predicate; the two deferrals above are
named follow-ups, not silent parkings.

## Verification

- `python -m pytest tests/test_open_pr_wrapper.py` - 52 passed.
- `ATLAS_SESSION_STATE_FILE=SESSION_STATE.codex-open-ready-draft-consent.local.md bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-pr-body-open-ready-draft-consent.md` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Open-Ready-Draft-Consent-Guard.md` | 252 |
| `scripts/open_pr.sh` | 119 |
| `tests/test_open_pr_wrapper.py` | 251 |
| **Total** | **622** |
