# Guard Class-Closure Discipline

The specific form of the root-cause gate (`AGENTS.md` section 3k) for guards over an
**open input space**. It exists because "fix the reported input" is a symptom
fix on these surfaces, and the symptom fix is invisible per-round: every listed
input passes, so the diff looks correct, while the *class* the input belongs to
stays open and the next input in that class is reported next round.

**This document is the single source** for the guard-closure bar (the three
requirements below), the open-category evidence-gated exception, and the
asymmetric-safe default. **Every** doc that names them -- the reviewer rules,
`AGENTS.md` 3k, the session bootstrap prompt, and any future one -- carries a
POINTER here, never a copy, and must not restate them normatively. The pointer
set is stated as "every such doc," not an enumerated list, because an enumerated
list is itself a restatement that drifts (a copy gets missed in one doc -- which
is what happened across the review rounds on #2077). To change the bar, edit only
this file.

## When this applies (trigger)

**A. Guards over an open input space.** Any change to a guard / validator /
sanitizer / classifier / gate / denylist / parser-admission rule / privacy or
safety checker whose input space is **open**: free text, nested or recursive
structures, producer-supplied keys and values, user data, or anything where the
set of possible inputs is not a small closed enum. If the input space is
genuinely a closed enum (a fixed set of statuses, say), this does not apply --
enumerate it and move on.

**B. Set-valued dependencies (any change, guard-shaped or not).** Any change that
adds or edits a **set whose membership a decision depends on**: a literal
collection used in a branch, a pattern family (regex alternation, prefix list),
a list duplicating one that already exists elsewhere in the repo, or the set of
behaviors / callers / fields a change must cover to be complete.

Trigger B exists because the same failure arrives constantly in code that nobody
reads as "a guard." A mirror script's test-file list, a detector's grep patterns,
a refactor's replaced-path behaviors, and a browser snapshot predicate are all
one behavior -- **the set was enumerated at authoring time instead of bound to a
source of truth or given a stated default for its unlisted members** -- and each
member discovered later is a real, correct review finding, so the loop cannot be
blamed on the reviewer and does not converge until someone writes down what
*generates* membership rather than what is currently in the set.

Trigger B requires the closure declaration below. It requires the three
requirements only when the change is also a trigger-A guard.

## The failure mode this prevents

Observed concretely on the S6A structured-privacy guard: **9+ review rounds**,
each fixing exactly the string(s) the reviewer/bot reported, each shipping with
the rest of the class still open. Forensic check across round heads showed the
later-round bugs were **pre-existing and behavior-identical across rounds** --
not regressions churned in and out, but one latent class being closed one thin
slice per round. Two compounding symptoms:

1. **String-closure instead of class-closure.** The fix targets the literal
   reported input. `published: "not published"` gets caught; `published: "kept
   private"` / `"no longer public"` / `"withheld from public"` -- same class,
   different words -- still leak.
2. **Additive-branch growth instead of a choke point.** Each round adds a branch
   (object handling -> sequence handling -> content carve-out -> negation
   handling), and each new branch's *fall-through* is the next round's hole.
   Complexity and edge-count grow every round while the guarantee does not.

## Closure declaration (trigger B; also required of trigger-A guards)

Every set-valued dependency the change introduces or edits declares, in the plan,
**exactly one** disposition. Naming the set is the point: an undeclared set is
one whose unlisted members have no defined behavior, which is precisely the state
that generates a finding per round.

- **CLOSED** -- the set is genuinely finite and repo-owned. Cite where the
  canonical list lives. A fixed status enum, or a controlled vocabulary the
  system already owns, is legitimately closed; enumerate it and move on.
- **DERIVED** -- the set is computed at runtime from a source of truth, so it
  cannot drift. Parse the workflow rather than copying its file list; read the
  settings schema rather than grepping for one accessor idiom. Prefer this
  whenever a source of truth exists: it is the only disposition that stays
  correct without maintenance.
- **DEFAULTED** -- the set is open and any enumeration is a heuristic. State the
  default direction for members not in the list, make incompleteness err to the
  cheap-error side (see the asymmetric-cost section below), and say which side
  that is and why.

**Enumerating an open set with no declared default is the defect.** A list that
looks complete today is indistinguishable, in review, from one that is
deliberately partial with a safe default -- the declaration is what makes the
difference visible.

Duplication is a closure failure of its own: a list copied from somewhere else in
the repo is CLOSED only if something enforces the copy: otherwise it is a DERIVED
candidate that was not derived, and it will drift silently. Prefer inverting
ownership (have the one consumer call the canonical definition) over adding a
drift check on two copies.

Observed 2026-07-26, five instances in one day across the EOM funnel and
content-factory lanes: a browser attribution snapshot keyed on "has ad params"
rather than "is an acquisition entry"; an atomic resolver that shed the replaced
path's behaviors because they were never inventoried; a config-probing detector
matching `os.getenv` in a repo whose own law mandates Pydantic settings, so it
could not fire on the incident that motivated it; a CI mirror script hardcoding
the pytest list it was written to keep in sync; and a copy-verification predicate
enumerating 25 copulas, so `every issue requires review` failed to bind even
though `issue` is in the repo's own report-item vocabulary. Each was a real
finding; each cost at least one round; none required new information to prevent.

## The three requirements (all mandatory before merge)

These apply to trigger-A guards. A trigger-B change that is not a safety guard
satisfies this document with the closure declaration above.

### 1. Fail-closed choke point (allowlist-shaped, one decision)

The choke point governs the **safety decision** -- the verdict that decides
whether unsafe content is admitted. For that decision the guard admits **only on
affirmative recognition of a safe value**: every unrecognized, unresolved,
malformed, or novel-shape input reaches the safe verdict **by construction**,
through a single decision point -- never through a per-input branch whose
fall-through is the unsafe verdict.

On an open input space, "reject the known-bad list, admit everything else" is
banned for the safety decision: the bad list can never be complete, so its
complement always leaks. The safety decision must be "admit the known-good,
reject everything else," and a guard whose default (the path taken when nothing
matched) is the *unsafe* verdict is rejected at the plan stage, before code.

**Scope caveat -- this is not "reject all unknown text everywhere."** A guard
often has fields with a *documented* pass-through / data-column / neutral-admit
policy (for example the S6A guard admits neutral producer text under
`access`/`audience` and treats `type`/`kind`/publication-date columns as data
per issue #2060). Those policies are legitimate and stay: the choke point
constrains the *privacy/safety verdict*, not every field's text semantics. State
which families the safety decision governs and which have a documented
neutral-admit policy; the ban is on an *open unsafe default* in the safety
decision, not on any deliberate admit.

### 2. Class-closure, not string-closure

A fix for a reported input must close the **class** it belongs to -- its grammar,
vocabulary, shape family, and container variants -- not the literal input. The
plan names the class and the invariant. Concretely: if two inputs differ only by
a value drawn from the same vocabulary (`"not published"` vs `"kept private"`),
or by a container wrapper (`x` vs `[x]` vs `{"value": x}`), or by which key
family carries them, **one fix closes all of them**. A diff that adds a branch
per reported input, or a token per reported string, is a symptom fix by
definition here.

### 3. Generative property test derived from the grammar (the acceptance gate)

Acceptance requires a test that **generates** inputs from the vocabulary/grammar
and asserts the invariant across the product:

- tokens x modifiers (negation, casing, affixes) x
- container shapes (scalar, list, tuple, set, nested, wrapped) x
- key families.

It has **two independent layers, and representation parity alone is not
enough**:

1. **Representation parity** -- expected values derived from the scalar/base
   case (`verdict(K, [v]) == verdict(K, v) == verdict(K, {"value": v})`, blank
   placements are identity, etc.). This proves containers/wrappers do not change
   the verdict.
2. **Semantic oracle** -- an expected verdict derived from the **specification /
   issue contract, independent of the implementation**, for each anchor class:
   a recognized-private value rejects, a malformed-numeric value rejects, an
   affirmative-public value admits, a documented neutral/data-column value
   admits. Assert against that oracle, not against `verdict(K, v)` itself.

Parity alone is a trap: if the base-case verdict `verdict(K, v)` is *semantically
wrong*, every generated wrapper equals the same wrong verdict and a
parity-only gate stays green while the class is still broken (a guard that leaked
`{"published": "kept private"}` would pass a parity test that only checks
`[v] == v`). The semantic layer is what actually anchors correctness; parity
proves the choke point is representation-invariant on top of it. A fixture list
of the specific reported strings satisfies neither layer -- it proves the
strings, not the class. The suite must be able to fail the whole open class at
once, so a regression is caught by construction instead of by the next review
round.

## When the recognizer itself is open (evidence-gated closure)

Requirement 1 says "admit known-good, reject the rest." That assumes the
known-good set is *recognizable* -- that you can write the allowlist. Some guards
fail a level deeper: the decision requires classifying membership in an **open
semantic category** -- is this token a person name, a real sender, an intent, a
language, "is this junk." Then both lists fail. The denylist of non-members is
unbounded (reject-known-bad leaks), and the allowlist of members is *also*
unbounded (admit-known-good cannot be written either). No grammar over the
category closes it, because the category is open on both sides. A property test
over category members (requirement 3) still cannot converge, because the member
set it samples is itself infinite.

**Do not recognize the category. Gate on bounded, mechanical evidence, and
default the ambiguous case to the asymmetric-safe side.** Replace "is X a real
sender?" (NER, unbounded) with "does this line carry structural corroboration?"
(an email in the header position, a quote marker, a confirmed header block --
finite and checkable). Act only on positive evidence; when evidence is absent the
input is ordinary content and takes the safe default. This closes the expensive
class by construction: any member the recognizer would have argued about --
including ones no one enumerated -- lands on the safe default.

Worked example (Resolution Audit S6C, #2076): a transcript sanitizer tried to
decide "is `<X>` a sender?" to drop quoted replies. Nine rounds of denylist and
then allowlist edits chased an open set of non-person senders --
`I` -> `We` -> `It` -> `The System` -> `Our Team` -> `Support Team` / `Acme
Corp` / `Billing Department` -- each patch shipped, each exposed the next, and
every miss dropped a real customer question. The converging fix recognized NO
senders: it skipped a reply only with an email in the header, a `>` quote block,
or a `From:`+`Sent:` header, and defaulted every other line to content. A
brand-new Unicode sender the design never saw (`Jose Garcia Team`) then kept its
question by construction. Enumeration ran nine rounds and never closed;
evidence-gating closed in one.

### Asymmetric error cost: bias the default, close only the expensive side

When the two error directions cost differently -- dropping a customer question is
product-breaking; keeping an old signature is <= the status-quo leak -- the
default MUST fall to the cheap-error side, and only the **expensive-error class**
must be closed by construction. The cheap-error residual is accepted and stated
(bounded, <= status-quo), not chased. The acceptance bar is "<= status-quo in the
expensive direction," not "perfect in both." Trying to be precise in both
directions on an open category is what generates the endless rounds: every time
you tighten the cheap side you re-open the expensive one. Pick the safe default
from the cost asymmetry first, then gate the exceptions on evidence.

**Evidence-keyed, not membership-keyed (oracle refinement of requirement 3).**
When the class is an open category, the generated oracle must be keyed on the
*evidence signal* the guard acts on (`has-email x has-quote-marker x
has-header-block -> skip/keep`), NOT on a list of category members (sender
strings). An oracle that enumerates members and grows a row per reported member
is a fixture matrix wearing a `product()` costume -- it converges no better than
the denylist it tests. The evidence-keyed oracle is finite and independent
because the evidence signals are finite even when the category is not.

## Reviewer bar (enforced)

For a trigger-B PR, the reviewer states before LGTM: each set-valued dependency
the change adds or edits, and its declared disposition. An enumerated open set
with no declared default is "needs the closure declaration," even when every
listed member behaves correctly -- that is the round-generating state this
document exists to stop, and it is invisible per-round by construction.

For a trigger-A PR, the reviewer **blocks** until all three hold, and states
before LGTM: the choke-point location, the class and invariant it enforces, and
that the acceptance test is grammar-derived (not a fixture list). For an
**open-category** guard (see the section above), the three requirements hold in
their *evidence-gated* form: the choke point recognizes bounded structural
evidence with an asymmetric-safe default, and the acceptance test is the
evidence-keyed generative oracle. There the reviewer must NOT demand a
member-enumerating allowlist or a member-keyed property test -- the open category
cannot satisfy either, and requiring them re-creates the enumeration loop this
document exists to stop. A string-scoped
fix with string-scoped fixtures is an automatic "needs the class fix," even when
every listed input now passes.

This composes with the boundary-probe rule in `docs/REVIEWER_RULES.md` (probe
both error directions) and `AGENTS.md` section 3k (root cause, not symptom): here the
root cause is the open default, and closing it is the choke point.

An **advisory CI lint** (`scripts/check_guard_class_closure.py`, workflow
`Guard Class-Closure Lint`) surfaces a warning when a PR changes a Python guard-shaped
file over open input without a co-changed property/generative test. It is
heuristic and advisory-first (warns, never blocks); it does not replace the
reviewer bar above, it makes the omission visible on every PR. It covers
trigger A only: the closure declaration for trigger B is reviewer-enforced today,
with a mechanical trigger (a diff adding a literal collection or pattern
alternation inside a decision path) proposed in issue #2226. Opt a
false-positive path out in `scripts/guard_class_closure_config.json` (optional -- absent means no ignores; create it only when an opt-out is needed) or waive
inline with a `guard-class-closure: waived` marker in the PR body.

## Relationship to the review-round cap

The Codex/bot review-round cap is a circuit-breaker for **noise** -- a reviewer
demanding opposite verdicts on formally-identical shapes, where the PR's actual
contract was green from round 0-1. It is **not** satisfied by cap-and-waive when
the findings are a real open class in a money / auth / PII guard (the same set
`docs/OVERNIGHT_ARC_WORKFLOW.md` names as blocking past the cap): those block
regardless of round count. On these surfaces the convergence tool is this
discipline -- the choke point plus the grammar-derived property test close the
class in one pass -- not another round of spot patches and not waiving the
residual to reach green.
