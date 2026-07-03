# PR-Reddit-Fit-Contract

## Why this slice exists

Second slice of the approved Reddit Listening v2 arc (#1931 comment
4872154794), after the S1 evaluation harness merged as db4a10d3d. S1 shipped
the ruler; S2 ships the runtime half of the contract it measures: the
FitDecision object, the strict parser that will gate every model response,
the prompt builder, and the wire JSON schema. The load-bearing move is the
SWAP: the harness's S1-local shape checker now delegates to the real parser
core, and the entire S1 corpus (120 tests, adversarially hardened through
5 review waves) stays green through it -- mechanical proof that the ruler
measures the runtime contract rather than a lookalike. Zero model calls,
zero network, zero store/digest changes.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/fit-eval
Slice phase: Vertical slice

1. `atlas_reddit/fit.py`: `FitDecision` frozen dataclass;
   `fit_decision_problems` (the single shape-validation core, constants
   from `atlas_reddit/fit_rules.py`); `parse_fit_decision` (JSON string
   -- optionally code-fenced -- or decoded object -> FitDecision, or
   `FitParseError` whose `code` is always one of the closed
   PARSE_ERROR_CODES so it rides prediction envelopes untouched);
   `build_fit_prompt` (deterministic two-message prompt whose "never do
   this" bullets are rendered FROM the catalogue rule messages -- told,
   graded, and later blocked by the same rules); `FIT_OUTPUT_JSON_SCHEMA`
   (strict-mode-safe keywords only); `PROMPT_VERSION = "fit.v1"`.
2. `atlas_reddit/fit_eval.py`: `check_prediction_shape` retires its local
   twin and delegates to `fit_decision_problems`.
3. `tests/test_atlas_reddit_fit.py`: parser acceptance (canonical yes /
   no-with-null, whitespace canonicalization, fenced JSON) and every
   rejection class on both the string and dict paths; problem-code
   stability; the harness-is-the-parser identity probe; prompt boundary
   coverage (EVERY catalogue rule message asserted present), posture and
   contract lines, candidate rendering, no-body marker + low_context
   invitation, determinism, ASCII; wire-schema strict-safety walk.
4. Housekeeping (separate first commit): archive the merged S1 plan and
   regenerate `plans/INDEX.md`.

### Review Contract

- Acceptance criteria:
  - [ ] The S1 harness suite is green THROUGH the swapped-in parser core
        (the corpus contract test still fires exactly its declared
        checks/codes) -- no drift between ruler and runtime.
  - [ ] Parser rejects every malformed class with stable codes: string
        path (empty / prose / truncated JSON / non-object JSON) via
        FitParseError codes drawn from PARSE_ERROR_CODES; dict path via
        the same problem codes S1 graded.
  - [ ] `no` verdicts canonicalize angle to None; empty-string angles
        never survive parsing.
  - [ ] The prompt embeds EVERY rule message from the catalogue, the
        read-only/no-reply-drafting posture, the support-ticket product
        truth, the risk-flag vocabulary, and the exact output contract;
        it is deterministic and ASCII.
  - [ ] Body-less candidates render an explicit no-body marker and
        invite the low_context flag (pre-v5 stores have no body).
  - [ ] The wire schema uses only strict-mode-safe keywords; the parser
        remains the authoritative gate.
- Reachability proof (#1952): the real harness CLI
  `scripts/evaluate_atlas_reddit_fit.py` runs end-to-end with the real
  parser core in the grading loop -- pass corpus exits 0, fail corpus
  exits 1 with the summary artifact written (commands in Verification).
- Affected surfaces: one new pure module, one delegation swap inside the
  harness, one test file, plans housekeeping. No store, digest, poller,
  config, CLI-command, or credential surface is touched.
- Risk areas: contract drift between harness and parser (eliminated by
  delegation -- there is one implementation now); prompt/rule divergence
  (eliminated by rendering bullets from rule messages and asserting
  coverage per rule).
- Reviewer rules triggered: R1, R2 (parser probed both sides on both
  input paths), R11 (zero new dependencies; stdlib json/re only), R12
  (test auto-enrolls via the workflow glob), R14 (reachability named
  above).
- Test-adapter posture: nothing is faked -- S2 has no external boundary;
  real parser, real prompt builder, real catalogue, real harness.

### Files touched

- `atlas_reddit/fit.py`
- `atlas_reddit/fit_eval.py`
- `plans/INDEX.md`
- `plans/PR-Reddit-Fit-Contract.md`
- `plans/archive/PR-Reddit-Fit-Eval-Harness.md`
- `tests/test_atlas_reddit_fit.py`

## Mechanism

`fit_decision_problems` carries the exact validation semantics the S1
harness hardened through five adversarial review waves (strict four-key
shape, verdict enum, 280-char caps after whitespace collapse,
verdict-conditional angle with no-with-angle rejection, closed risk-flag
vocabulary with element type checks before dedup). `parse_fit_decision`
wraps it for raw model text: strip, unwrap an optional ```json fence,
json.loads, require an object, then shape-validate -- each failure mapped
to a closed PARSE_ERROR_CODES value so the S5 client can drop the code
straight into a prediction envelope. Successful parses canonicalize
(collapse whitespace; no/empty angle -> None) into a frozen FitDecision.
`build_fit_prompt` renders the system message from three fixed blocks --
role + read-only posture, support-ticket product truth, and one boundary
bullet per unique catalogue rule message -- plus the output contract with
the shared length caps, and a user message carrying the candidate fields
with an explicit no-body marker when the body is absent.

## Intentional

- **Delegation, not duplication**: the harness's checker body was
  DELETED, not kept in parallel -- one implementation means ruler/runtime
  drift is structurally impossible, which is what "the harness is the
  ruler" required.
- **FitParseError codes are the envelope codes**: the parser speaks the
  closed PARSE_ERROR_CODES taxonomy directly, so the S5 client will not
  translate (translation layers drift).
- **Fence-stripping is the only string normalization**: a deterministic
  unwrap of one common wrapper, not leniency about content; everything
  else malformed fails closed.
- **Length caps live in the parser, not the wire schema**: strict-mode
  backends reject unsupported keywords; the schema is best-effort, the
  parser is the gate.
- **Prompt bullets come from rule messages**: adding a rule in S3+ makes
  the prompt tell the model about it automatically, and the
  every-rule-present test forces the coverage.

## Deferred

- S3 runtime guard (imports the same catalogue; parity test) -> S4 store
  v5 + manual import -> S5 judge client -> S6 runner + digest, per the
  approved arc. The prompt builder's caller (S6 runner) and the schema's
  consumer (S5 client) land in their slices.

Parked hardening: none.

## Verification

- `.venv/bin/python -m pytest tests/test_atlas_reddit_fit.py -q`: 31
  passed (parser both paths both sides; harness-is-parser identity;
  prompt rule coverage; schema strict-safety).
- Full package suite `.venv/bin/python -m pytest
  tests/test_atlas_reddit_*.py -q`: 484 passed -- including the entire
  S1 harness corpus THROUGH the swapped parser core.
- Reachability: `python scripts/evaluate_atlas_reddit_fit.py --cases
  tests/fixtures/atlas_reddit_fit_eval/cases.jsonl --predictions
  tests/fixtures/atlas_reddit_fit_eval/predictions_pass.jsonl
  --fail-on-eval-fail` exits 0; the predictions_fail.jsonl variant exits
  1 -- the real CLI with the real parser in the loop.
- ASCII byte-scan on changed Python files: clean.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_reddit/fit.py` | 266 |
| `atlas_reddit/fit_eval.py` | 55 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reddit-Fit-Contract.md` | 150 |
| `plans/archive/PR-Reddit-Fit-Eval-Harness.md` | 0 |
| `tests/test_atlas_reddit_fit.py` | 249 |
| **Total** | **723** |
