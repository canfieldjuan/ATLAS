# PR-Resolution-Audit-S6A-Structured-Privacy-Semantics

## Why this slice exists

Issue #2060 reopens Resolution Audit launch-blocker 4 after a cold reconstruction
of current main proved that the S6A admission boundary merged in #2046 still
admits structured private markers and drops equivalent public metadata. A mixed
Zendesk-submit probe carried nested private prose into the stored paid artifact,
so this is a live privacy boundary defect, not speculative grammar hardening.

This slice fixes the canonical classifier and proves the public submit path. It
does not claim to close #2060: generic `/execute` can still restore raw caller
`source_material` over the provider-filtered package, and the serialized S6A.2
slice owns that separate upstream authority defect.

The diff exceeds the 400-line soft target because this privacy classifier is not
safe to land without both error-direction class coverage and the real
submit-to-persist proof. The production change remains one module; the rest is
the required plan, living-tracker correction, vocabulary-derived adversarial
matrix, and one route test. Splitting any of those would leave either an
unreviewable guard or a green helper with no proof that the paid artifact
boundary uses it.

### Problem-derived contract

- Root cause: marker parsing collapses raw scalar tokens and semantic object
  decisions into the same strings (`public`, `private`, `__ambiguous__`). Outer
  key-family policy then reinterprets those strings, losing recursive subfield
  polarity, losing whether a public-facing decision came from a boolean
  assertion, and treating a valid unknown object label differently from its
  scalar equivalent. Sequence content mode also bypasses scalar classification
  for strings, and malformed provenance is enforced inside individual family
  branches after other families have already accepted. The value grammar also
  recognizes positive family stems without recognizing bounded negation of
  those same stems, while sequence branches treat `ABSENT` as private or
  unresolved instead of the same identity value objects and scalars use.
  Public-audience vocabulary is also split between two sets, and conjunction
  tokens prevent otherwise explicit private phrases from resolving. Positive
  public-audience phrase promotion is applied without the outer key family's
  polarity, so `end user` can invert a `private`/`hidden` subfield. Sequence
  cardinality is also checked before scalar-`ABSENT` placeholders are removed,
  so one neutral scalar plus a blank takes the stricter multi-item path.
  Finally, public-flag policy admits every neutral string as if it were
  publication data, then tries to recover privacy by enumerating negation
  phrases; every unrecognized private phrasing therefore reopens the same
  fail-open. These are representation and invariant-placement defects plus a
  family-default defect, not independent spelling bugs.
  Round-10 exact-head probes exposed three residual invariant-placement defects:
  content-container provenance is granted from a body/text key's presence rather
  than from non-blank textual content; the recursive placeholder predicate skips
  empty sequences and therefore counts them as substantive; and the existing
  `malformed_numeric` bit intentionally covers both invalid non-finite numerics
  and valid finite numerics that are data only for publication flags. Adding the
  public-flag family to the shared malformed set would therefore close the leak
  by breaking the required finite-number data path. The missing root signal is
  distinct non-finite-numeric provenance that survives combination and dominates
  an otherwise-public sibling.
  Round-11 exact-head probes showed that the same masking defect still exists one
  class wider: an affirmative-public sibling can erase privacy-vocabulary text or
  an invalid date that fails as a scalar, and content-container provenance can
  erase a recognized private marker nested inside a body/text wrapper. The
  combiner records public/content evidence but has no uniform recognized-failing
  evidence that dominates either admission shortcut. Key admission also has two
  grammar holes before the combiner: compact `un<public-flag-stem>` keys and
  public-flag stems with `status`/`label` suffixes remain unclassified, so their
  values bypass the closed family entirely.
  Round-12 held-out probes show five residual provenance-order defects in that
  same bounded grammar: compact negated values are recognized only after their
  spelling has lost token boundaries; `un` key polarity is checked before, but
  not after, label/status suffix removal; numeric `0`/`1` returns before exponent
  provenance is attached; invalid-date detection excludes date-only timezone
  suffixes; and multi-item sequence admission omits otherwise-valid
  `publication_data` from its marker-shaped path. Each scalar policy is already
  correct; wrapper/alias ordering is discarding its evidence before combination.
  Round-13 review proves the deeper architecture is still not closed: family
  policy is distributed across scalar, object, sequence, and final-admission
  functions, while normalization repeats before and after lowercasing and
  compaction. There is no single canonical key meaning, semantic decision, or
  policy table against which wrapper equivalence can be proved. Consequently
  public-family failure dominance, qNaN, compact datetime offsets,
  strict-wrapper exponent provenance, and camelCase phrase boundaries each fall
  through a different ordering seam. These are not five new strings; they are
  evidence that production lacks a reference model and class-closure oracle.
  Round-14 re-review confirms that architecture closed every generated family
  except invalid publication dates: scalar invalid dates fail closed, but the
  semantic decision marks them `recognized_failing` only when a narrow
  date-shaped regex matches. Trailing junk and non-zero-padded calendar forms
  therefore become neutral evidence inside a wrapper and can be laundered by a
  public sibling. The root cause is a split publication-data decision -- valid
  data and invalid date candidates are not produced by one canonical scalar
  classifier -- plus an oracle domain that omitted those invalid-date shapes.
  Round-15 review exposes four remaining decision-boundary gaps rather than new
  vocabulary: placeholder identity recurses through sequences but not generic
  wrapper mappings; structured recognized-failing evidence is evaluated after
  the content exit; content presence uses raw non-whitespace rather than the
  package's rendered-text semantics; and a classified access/type subfield that
  its own family admits as neutral still exports raw failing provenance to a
  stricter outer family. These are normalization/precedence defects in the
  shared semantic model, not four marker strings.
  Round-16 review shows those predicates were repaired only at one structural
  level. Recognized-failing marker evidence is not derived recursively through
  nested content wrappers; absence identity does not use rendered-content
  semantics for body-bearing mappings inside sequences; admitted content is
  still also labeled unresolved and can conflict with an explicit public
  sibling; and a locally neutral classified subfield still exports affirmative
  strict/publication provenance. The root is four incomplete semantic folds at
  the shared choke point, not another value-vocabulary gap.
- Correct fix must touch/change: the package-owned
  `support_ticket_privacy.py` classifier must preserve distinct public, private,
  neutral-unknown, ambiguous, and absent decisions plus boolean-assertion origin
  through recursive object resolution, including neutral wrapper depth. Empty
  mapping markers must retain structured provenance; classified nested subfields
  must retain neutral rather than being promoted to public; and strict-public
  evidence across neutral subfields must aggregate without mapping-order loss.
  Neutral boolean wrappers must also retain their outer key family so categorical
  strict labels cannot inherit public polarity, and neutral metadata must remain
  non-conflicting beside an explicit public/private verdict. Multi-token public
  audience phrases must preserve `end user` as one semantic audience. Public-
  facing flags must honor explicit private text independently of boolean origin,
  while access/audience value labels must reject the complete recognized
  malformed-numeric class, including exponent notation and non-finite numeric
  values produced by JSON exponent overflow or represented as CSV strings,
  without rejecting valid producer-defined neutral text. Numeric provenance must
  survive recursive object resolution rather than being inferred later from a
  compacted token. Recursive marker resolution must also traverse non-empty
  sequence wrappers while accepting genuine row-level note/comment content only
  when the caller establishes that context and each body-bearing comment is
  independently public. Strict-label policy must consume malformed-numeric and
  strict-public-audience provenance in object and sequence forms. A classified
  strict-label boolean subfield must remain fail-closed instead of being demoted
  to neutral metadata beside public evidence, while generic categorical wrapper
  booleans must retain scalar/object/direct-sequence/nested-sequence parity for
  access/audience and type/kind. Strict sequences must preserve whether a boolean
  is raw assertion material or arrived through a generic wrapper: raw booleans
  remain ambiguous, while generic wrapper booleans remain neutral metadata.
  Key-family policy must consume that decision directly before any content
  carveout. Non-empty note/comment mappings without an actual body-bearing
  content field must retain structured provenance and fail closed, while real
  body-bearing mappings and free-text sequences retain the named content exit.
  Every singleton list/tuple/set/frozenset wrapper must reuse the scalar decision path;
  multi-item content sequences must classify each string first and treat only
  scalar-admissible neutral/public text as content. Malformed-numeric provenance
  must fail closed before any private/public/strict/value-label family can
  accept, while publication/facing and type/kind retain their documented data-
  column semantics. CI-facing tests must generate scalar/singleton list/tuple/
  generic-value-wrapper parity for marker families whose wrappers inherit the
  outer polarity. Note/comment content mappings remain structured markers and
  retain their separate fail-closed policy.
  Public-flag and strict-label families must admit only affirmative recognized
  public evidence; publication/facing flags additionally admit an explicit ISO
  date/time or finite-number data shape. Every other non-empty unrecognized text
  must fail closed without a negation-phrase enumeration. Access/audience and
  type/kind must retain #2060's separate neutral producer-data policy. `ABSENT`
  sequence elements (blank strings and `None`, including nested placeholder-only
  wrappers) must be removed before sequence cardinality is selected, so one
  remaining element reuses the scalar path in both content and marker modes,
  while recognized private, malformed, conflicting, and multiple unresolved
  non-absent siblings retain their existing fail-closed effect. Positive
  multi-token public-audience phrases must be recognized only where the outer
  public/strict family permits that polarity; private-family subfields must not
  be inverted, while exact explicit `public` labels retain their established
  meaning. CI-facing tests must derive public/private token combinations,
  unrecognized contexts, container shapes, and blank placement from the family
  vocabulary across every family in both error directions.
  Content-container provenance must require non-blank text under a recognized
  body/text field, including nested content wrappers; empty, whitespace, `None`,
  and placeholder-only content shapes must remain structured and fail closed.
  Empty nested sequences must be `ABSENT` identity before outer sequence
  cardinality without changing the established top-level empty-comment-list
  content behavior. Marker decisions must separately carry non-finite-numeric
  provenance through object and sequence combination; every publication/facing
  family must reject that provenance before affirmative public evidence, while
  calendar-valid dates and finite numbers continue to admit in scalar and
  public-labeled structured controls.
  Closed-family scalar decisions must additionally carry one
  `recognized_failing` invariant derived from privacy vocabulary, invalid
  date/non-finite data shapes, and explicit private/ambiguous decisions. Object
  and sequence combination must propagate it and fail closed before a public
  label can admit; genuinely neutral metadata without privacy vocabulary remains
  compatible beside explicit public evidence. Content admission must recursively
  apply the existing key-family classifier inside recognized body/text wrappers
  and deny the carveout when any nested marker resolves private, while explicit
  public controls and genuine text remain admitted. Key classification must
  recognize compact `un` plus an exact public/public-flag stem as the private
  side, and remove `status`/`label` only when the remaining exact stem is a
  public-flag family, preserving unrelated data-column keys.
  Compact negated closed-family values must resolve through the same bounded
  public/public-flag stem grammar before public siblings combine. Label/status
  stripping must then reapply that exact negated-stem grammar. Numeric parsing
  must retain exponent-shaped malformed provenance even when its numeric value
  is boolean-like `0` or `1`. Date-like detection must classify malformed
  calendar values with date-only `Z`/offset suffixes as failing evidence without
  widening arbitrary prose. Valid publication dates/numbers must remain
  marker-shaped in multi-item public-label sequences so wrapper policy matches
  scalar/object policy.
  The correct architectural fix must canonicalize keys into one typed family
  decision, canonicalize scalars/objects/sequences into one semantic decision,
  and apply exactly one immutable family-policy table. No family admission rule
  may remain in recursive parsing. The table must own boolean polarity,
  admitted verdicts, strict-public labels, neutral-data admission, publication
  data, content carveouts, and malformed/non-finite/failing precedence. A small
  independent test oracle must express the same product contract without
  importing runtime internals, generate family x semantic atom x key/value
  spelling x container x sibling-evidence compositions, and compare every case
  to the public predicates. New examples may extend a semantic atom generator,
  but must not add production admission branches.
  Publication data must likewise be one semantic classification with valid,
  invalid-date-candidate, and not-data outcomes. A date candidate that does not
  produce valid publication data must set the existing `recognized_failing`
  evidence before wrapper combination; arbitrary neutral prose must remain
  not-data. The generated oracle must include trailing-junk, non-zero-padded,
  and out-of-range date candidates across every family and supported wrapper/
  sibling composition so scalar and masked outcomes cannot diverge again.
  Placeholder normalization must recurse through non-empty recognized wrapper
  mappings whose values are all absent, without treating an empty structured
  marker as absent. Structured recognized-failing evidence must dominate the
  content carveout, while direct free-text content retains its established
  path. Content existence must use the same rendered plain-text semantics as
  package normalization so markup/entity-only placeholders cannot authorize
  content admission. A classified subfield must export the result of its own
  family policy to the outer combiner: provenance that the subfield family
  admits as neutral cannot be reinterpreted as failing by the outer family,
  while locally rejected provenance remains private.
  Recognized-failing dominance must recurse through every recognized content
  wrapper depth before any content carveout. In content-sequence context,
  absence must recurse through body/text wrapper mappings and use fully rendered
  text, with content requiring at least one alphanumeric Unicode character;
  admitted content is compatible evidence rather than an unresolved sibling.
  A classified subfield may export affirmative strict/publication provenance
  only when its own family decision is public; a locally neutral result exports
  no positive provenance to the outer family.
  The documented `has_access` data column must remain unclassified for every
  normalized spelling and value shape instead of inheriting access-marker
  policy from generic `has` prefix stripping. Public strict labels must share
  the public-audience vocabulary; conjunctions must be value structure.
  CI-facing classifier tests
  must cover both error directions and held-out same-class cases, and the real
  `/deflection-reports/submit` route must prove mixed public/private input keeps
  public evidence while no private sentinel reaches the snapshot or stored
  artifact. Package-level tests must prove recognized private/malformed
  structures cannot use row content-column carveouts. The living remediation
  tracker must reopen S6A and link #2060.
- Must not change: support-ticket package/Zendesk caller signatures, input-package
  precedence, the generic `/execute` merge, downstream campaign-source guard,
  report/snapshot/email/PDF/landing/pricing/checkout shape, scrubber grammar,
  clustering, junk, date, money, billing, delivery, database schemas, or any
  other lane.

## Scope (this PR)

Ownership lane: resolution-audit/privacy-admission
Slice phase: Vertical slice

1. Replace string-overloaded structured marker decisions with an internal typed
   decision consumed consistently by each existing key family.
2. Close the original post-merge finding classes and reviewed same-root
   regressions in both directions without changing the established content-
   column carveouts.
3. Prove the fix through the real mixed Zendesk full-thread submit route and
   persisted report artifact.
4. Prove package normalization rejects recognized private/malformed note and
   comment structures, including empty mapping markers and nested neutral metadata,
   while preserving genuine content containers.
5. Correct the local Resolution Audit tracker to show S6A reopened under #2060.

Max files: 6

### Review Contract

- Acceptance criteria:
  - [ ] Nested private/hidden assertions, object-wrapped false publication
        flags, and explicit private conjunction phrases classify private.
  - [ ] Neutral object labels and requester/client/end-user strict labels
        classify public/neutral consistently with scalar/key-form equivalents.
  - [ ] Empty, malformed, and contradictory structured markers still fail
        closed; free-text comment/note content columns still admit.
  - [ ] A note/comment body, text, message, content, or description key grants
        the content carveout only when its value contains non-blank text; empty,
        whitespace, `None`, empty-container, and placeholder-only wrappers reject
        at classifier and package boundaries.
  - [ ] Recognized privacy keys nested inside body/text/content wrappers are
        classified before content provenance is granted; private/failing markers
        reject through mapping/sequence nesting while explicit-public plus genuine
        text controls retain the content path.
  - [ ] Nested classified neutral metadata never becomes affirmative public
        evidence, and strict-public evidence is independent of mapping order.
  - [ ] Bare boolean wrappers under strict categorical labels fail closed;
        explicit public labels remain public beside neutral metadata.
  - [ ] End-user visibility phrases classify like equivalent end-user key forms.
  - [ ] Public-facing flags reject explicit private text but retain neutral date
        columns; access/audience reject malformed numerics, including scalar and
        object-wrapped exponent notation, JSON exponent overflow, and string
        non-finite forms, while valid neutral producer labels remain admitted per
        #2060.
  - [ ] Non-empty sequence wrappers preserve recognized marker decisions and
        reject structured private assertions. Genuine row-level string/body
        note and comment containers retain the carveout only when body-bearing
        mappings independently classify public; non-content keys remain closed.
  - [ ] Non-empty unknown and neutral-wrapper note/comment mappings fail closed
        across scalar mapping and sequence containers, while genuine body/text
        containers and the pinned public-comment compatibility path still admit.
  - [ ] Singleton list, tuple, set, and frozenset containers match scalar
        outcomes across every key family for private, public, neutral, boolean,
        and malformed-numeric values; content lists classify recognized scalar
        markers before using the free-text carveout.
  - [ ] Malformed-numeric provenance fails closed before public/private/strict/
        value-label acceptance in scalar, object, list, tuple, and generic-value
        wrappers; publication/facing and type/kind data semantics remain stable.
  - [ ] Publication/facing and strict families reject every non-empty
        unrecognized text by default, including but not limited to negated,
        withheld, and private-context phrases; recognized public evidence still
        admits, and publication/facing ISO dates and finite numbers retain their
        data-column outcome.
  - [ ] Every publication/facing key rejects non-finite numeric provenance beside
        a public label and through object/sequence wrappers; public-labeled valid
        dates and finite numbers remain admitted, so the repair does not widen the
        shared malformed-family set.
  - [ ] A public label cannot mask any recognized-failing closed-family value:
        privacy-vocabulary text, invalid date-like data, non-finite numerics, and
        explicit private/ambiguous decisions reject across public-flag/strict
        families and object/sequence/nested wrappers; genuinely neutral metadata
        plus explicit public evidence and valid publication data still admit.
  - [ ] Compact `un<public/public-flag-stem>` keys fail closed like existing
        `not_`/`non_` key forms, and public-flag `*_status`/`*_label` aliases reach
        the same family policy without classifying unrelated status columns.
  - [ ] Compact `un`/`not`/`non` public-flag values remain recognized failing
        beside public evidence, and suffixed negated keys such as
        `unpublished_status` reuse the same private polarity after suffix removal.
  - [ ] Exponent-shaped numeric `0`/`1` retains malformed provenance for
        access/audience and note/comment sequences; plain booleans and ordinary
        `0`/`1` controls retain their established outcomes.
  - [ ] Invalid calendar values with date-only `Z` or numeric-offset suffixes
        cannot be masked by public evidence, while calendar-valid date/time
        values remain publication data.
  - [ ] Multi-item public-label sequences preserve valid publication-data
        evidence and admit `public` plus a valid date/finite number without
        weakening failing siblings.
  - [ ] Key classification returns one typed canonical family decision; runtime
        code does not pass raw family strings or reclassify a key after value
        recursion begins.
  - [ ] One immutable family-policy table is the only place that decides boolean
        polarity, verdict admission, neutral-data behavior, publication data,
        content carveouts, and malformed/non-finite/recognized-failing precedence.
  - [ ] Recursive scalar/object/sequence parsing only produces and combines
        semantic decisions; it contains no family-specific row-admission exits.
  - [ ] An independent generated oracle covers every supported family across
        public/private/unknown/boolean/malformed/non-finite/publication atoms,
        separator/camel/compact spellings, scalar/object/sequence wrappers, and
        public/failing siblings, and matches both public predicates.
  - [ ] Public-family objects cannot launder failing public vocabulary; qNaN is
        non-finite; compact-offset datetimes remain publication data; strict
        wrappers retain exponent provenance; and camelCase private conjunctions
        retain token boundaries as consequences of canonicalization, not bespoke
        admission clauses.
  - [ ] Publication text produces one semantic valid/invalid/not-data decision;
        invalid date candidates, including trailing-junk, non-zero-padded, and
        out-of-range forms, reject identically as scalars and beside public
        siblings across every supported family and wrapper, while neutral prose
        and valid dates keep their established outcomes.
  - [ ] Non-empty generic marker wrappers containing only blank, `None`, empty
        sequence, or recursively placeholder-only values act as `ABSENT` identity
        before sequence cardinality; empty structured mappings remain fail-closed.
  - [ ] Recognized-failing marker subfields inside body-bearing note/comment
        mappings reject before the content carveout, while direct and nested
        genuine free text keeps the documented content path.
  - [ ] Markup/entity-only bodies that render to no ticket text do not grant the
        content carveout; rendered text uses the package normalization semantics.
  - [ ] Access/audience and type/kind subfields admitted as neutral by their own
        family policy do not leak raw failing provenance into an outer public/
        strict family; locally private subfields still dominate outer evidence.
  - [ ] Recognized-failing status/value/label/name markers reject through every
        supported body/text/content wrapper depth before content admission.
  - [ ] Content-sequence absence recursively removes rendered-empty body
        mappings, including decoded entity/punctuation-only values, while
        alphanumeric rendered content remains substantive.
  - [ ] Admitted comment content can compose with a standalone explicit-public
        marker without becoming ambiguous; private/failing siblings still win.
  - [ ] Locally neutral access/audience/type/kind subfields export neither
        strict-public nor publication-data provenance; an explicit outer public
        sibling is required for admission.
  - [ ] Blank-string and `None` sequence elements act as `ABSENT` identity in
        both content and marker paths: after removal, a single neutral/public
        survivor matches its scalar result, while multiple unresolved survivors
        and recognized private/malformed siblings retain fail-closed behavior.
  - [ ] Empty list/tuple/set/frozenset elements are likewise `ABSENT` when nested
        inside a marker sequence, while a top-level empty comments container keeps
        its established content-column behavior.
  - [ ] Multi-token public-audience phrases under `private`, `hidden`, and
        equivalent private-polarity subfields cannot become public evidence;
        public/strict visibility contexts still recognize those phrases and an
        exact explicit `public` value under a private key retains its control
        behavior.
  - [ ] A vocabulary-derived property matrix spans public/private tokens,
        unrecognized contexts, scalar/object/sequence containers, blank/`None`
        placement, and private/public/public-flag/strict/value-label/kind/content
        families; publication/strict default closed while access/audience and
        type/kind retain neutral producer text.
  - [ ] `has_access` remains a data column across normalized snake, camel,
        hyphen, space, and case spellings and cannot inherit malformed-numeric
        access-marker rejection from prefix stripping.
  - [ ] Classified strict-label boolean subfields remain fail-closed beside an
        explicit public sibling; generic neutral wrapper metadata remains neutral.
  - [ ] Strict labels reject malformed-numeric provenance even beside public
        evidence, and requester/client/end-user sequence wrappers retain the same
        public result as scalar and object forms.
  - [ ] Generic booleans under access/audience and type/kind retain their scalar
        family semantics through object, list/tuple, and nested sequence wrappers;
        classified strict boolean subfields do not become affirmative public.
  - [ ] Raw booleans in strict sequences remain ambiguous beside public audience
        evidence, while generic `value`-wrapped sequence booleans remain neutral.
  - [ ] Neutral wrapper recursion preserves inherited polarity at depths 1-3,
        and classified non-boolean subfields obey their own key-family policy.
  - [ ] Unknown-key subtrees remain opaque consistently at row, comment,
        marker-object, and body-bearing content levels; recognized privacy keys
        at those same levels still fail closed.
  - [ ] Package normalization rejects recognized private/malformed structures
        before they enter `source_material`.
  - [ ] At least 5-10 held-out same-class probes exercise recursive polarity,
        object/scalar parity, public audiences, and conjunctions.
  - [ ] A mixed real submit retains public evidence and excludes a private
        sentinel from response, snapshot, and stored artifact.
  - [ ] No product shape or non-privacy lane changes.
- Reachability proof: POST `/ops/deflection-reports/submit` through the real
  router, Zendesk full-thread importer, input package, report service, scrub/QA
  gate, and `InMemoryDeflectionReportArtifactStore`; inspect response, snapshot,
  and stored artifact state.
- Affected surfaces: support-ticket privacy classifier, public Resolution Audit
  submit admission, generated report persistence, remediation tracker.
- Risk areas: privacy leakage, public-ticket data loss, backward compatibility
  of producer marker shapes.
- Reviewer rules triggered: R1, R2, R3, R10, R12, R13, R14; guard-shaped
  boundary probe required.

### Files touched

- `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
- `extracted_content_pipeline/support_ticket_privacy.py`
- `plans/PR-Resolution-Audit-S6A-Structured-Privacy-Semantics.md`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_support_ticket_privacy_sweep.py`

## Mechanism

Introduce a private marker-decision value carrying one of `PUBLIC`, `PRIVATE`,
`NEUTRAL_UNKNOWN`, `AMBIGUOUS`, or `ABSENT`, plus whether the decision came from
a boolean assertion. Scalars and recursive objects resolve into this value;
objects combine all recognized subfield decisions, preserve inherited polarity
through neutral wrappers, apply classified subfields through their own family
policy without promoting neutral results, aggregate strict-public evidence
across all neutral subfields, retain the outer key family through neutral boolean
wrappers, ignore neutral metadata when combining an explicit public/private
verdict, and return ambiguous only on assertive contradiction or malformed
containers. Decisions retain malformed-numeric provenance across scalar and
object forms so value-label policy does not depend on the compacted spelling of
the numeric token, including non-finite numeric strings from CSV-shaped inputs.
The decision also retains explicit publication-data provenance for ISO
date/time and finite-number shapes, allowing public flags to distinguish real
data from arbitrary neutral text without interpreting open-ended prose.
It separately carries non-finite-numeric provenance, propagated with `any`
through object and sequence combination, so publication/facing policy can reject
invalid numeric evidence before an explicit public sibling without treating a
valid finite number as malformed publication data.
The same decision carries a single `recognized_failing` bit for every
closed-family scalar that contains privacy vocabulary, invalid date/non-finite
data, or an explicit private/ambiguous verdict. The shared combiner makes that
bit dominant before public-label admission, so adding another failing vocabulary
member or container cannot create a new laundering branch.
Publication-shaped text is classified once into valid data, an invalid date
candidate, or unrelated text. The classifier recognizes the stable ISO date
namespace rather than enumerating invalid suffix/month/day spellings, validates
the existing calendar/time grammar for the valid outcome, and derives both
`publication_data` and `recognized_failing` from that one result. Recursive
objects and sequences therefore cannot reinterpret a scalar-invalid date as
neutral merely because a public sibling is present.
Non-empty sequences recursively combine recognized marker decisions using the
same verdict combiner as objects. A singleton sequence delegates to `_marker`
exactly once so it cannot diverge from its scalar form. Scalar-`ABSENT`
placeholders and nested placeholder-only wrappers are removed before that
cardinality decision, so a sequence with one substantive survivor also
delegates to `_marker`; true multi-item content
sequences classify every string through that same path and use the content exit
only when the scalar family policy admits it. Deeper recursive `ABSENT`
decisions remain identity values during aggregation, matching object subfield
and scalar semantics without weakening any non-absent unresolved/private
decision. The caller explicitly marks genuine
note/comment columns; row inputs and the pinned public-comment compatibility
path treat strings and body-bearing mappings as content, and each mapping is
independently checked for privacy before the carveout applies. Strict-public
audience provenance participates in sequence
decisions, and strict-label policy rejects malformed-numeric provenance before
accepting public evidence. Classified strict-label boolean subfields remain
ambiguous/fail-closed; generic boolean wrapper keys retain the neutral categorical
semantics of scalar access/audience/type/kind values through direct and nested
sequence wrappers. Generic-wrapper origin is carried into nested sequences so
strict raw booleans remain ambiguous while `value`-wrapped booleans remain
neutral metadata. The decision also records whether an object actually
contained privacy structure, including an empty mapping marker, so only genuine
free-text/comment containers can use the row content carveout.

Placeholder identity also descends through non-empty generic marker mappings
whose recognized wrapper values are all recursively absent; a bare empty mapping
keeps structured ambiguous provenance. Classified subfields are reduced through
their own family policy before entering the outer combiner: locally admitted
neutral values export neutral evidence without raw failing flags, while locally
rejected values export `PRIVATE`. This keeps outer policy from overriding the
documented access/audience/type/kind carve-out.

For content sequences, the same absence fold recognizes content wrapper keys
and evaluates strings after package rendering; only rendered alphanumeric text
is substantive. Content admitted by family policy is recorded solely as content
evidence, not simultaneously as unresolved evidence. Before content admission,
recognized-failing semantics are derived recursively through recognized content
wrappers. Classified subfield reduction exports positive strict/publication
provenance only from a locally public decision, so neutral family carve-outs
cannot authorize a stricter outer family.

The decision separately records named content-container provenance. Raw text,
empty top-level comment lists, and body-bearing mappings receive that provenance only in
the established row/public-comment content context; unknown or neutral wrapper
mappings without a body-bearing field retain structured provenance and fail
closed. A body-bearing mapping qualifies only when a recognized content field
contains rendered plain text under the same HTML/entity normalization used by
the package; blank, markup-only, entity-only, `None`, and placeholder-only fields
cannot turn key presence into an admission signal. Structured recognized-failing
evidence is evaluated before the content exit, while direct free text retains
content provenance. Empty sequences nested inside a larger marker
sequence are removed as `ABSENT` before cardinality, without changing the
top-level empty-comments compatibility path. Exact normalized `has_access`
spellings exit key classification before generic prefix stripping, leaving
actual `access`, `access_label`, and flag-shaped access markers unchanged.
Before granting nested content provenance, the same key-family classifier walks
only recognized body/text/content wrappers and rejects any nested marker that
resolves private; unknown non-content subtrees remain opaque. Compact `un` key
forms are flipped only when their exact remainder is public/public-flag, and
label/status suffix removal is extended only to exact public-flag stems.
The same bounded negated-stem resolver is reused for compact closed-family
values and after label/status key suffix removal. Scalar numeric parsing records
exponent-shaped malformed provenance before returning boolean-like `0`/`1`, and
invalid-date detection accepts only ISO-shaped date/time or timezone suffixes.
Sequence marker-shape detection includes `publication_data`, preserving the
scalar/object admission decision for valid dates and finite numbers beside
public evidence.

Round-13 replaces distributed admission with three explicit layers.
`_classify_key` returns a typed canonical key decision. Recursive value parsing
returns only semantic evidence and never admits a row. `_decision_is_private`
then consults one `_FAMILY_POLICIES` table for every family behavior, including
content eligibility supplied by the caller. Scalar normalization preserves
original camel boundaries before case folding, assigns numeric/date/non-finite
evidence once, and passes the same decision through every container. The
test-side reference oracle independently defines semantic atoms and family
policies, generates supported normalization/container compositions, and asserts
the two public entrypoints match the oracle without importing any private
runtime helper or constant.

Existing key-family policy then consumes the decision instead of reinterpreting
the strings `public` and `private`:

- malformed provenance is rejected once, before private/public/strict/value-
  label family acceptance;
- private/public assertions retain fail-closed unknown behavior and row-mode
  content-column carveouts;
- publication/facing flags admit recognized public decisions and explicit ISO
  date/time or finite-number data shapes only, failing closed on every other
  non-empty neutral string;
- strict labels admit recognized public decisions only and likewise fail closed
  on unrecognized text;
- audience/access and type/kind reject private or ambiguous decisions but admit
  valid neutral unknown labels.

Public strict-label membership is derived from the existing public audience
tokens. Unrecognized public/private phrase contexts need no semantic scan: the
public-flag and strict family policies reject neutral text by default, while
the access/audience and type/kind policies retain their documented neutral-data
carve-out. `and`/`or` plus the separator in `end user` become structural only
inside recognized value phrases. Positive
public-audience phrase promotion is enabled only for public/public-flag/strict
outer families; the exact-label fast path remains unchanged, so explicit
`private: public` retains its established public override while
`private: end user` cannot invert private polarity. No caller or artifact
contract changes.

## Intentional

- This fixes the semantic representation root rather than adding the thirteen
  reproduced strings to denylists/allowlists.
- `InMemoryDeflectionReportArtifactStore` is the real local implementation of
  the report-store protocol and is appropriate for route-level observable-state
  proof; no pool or query-string fake is introduced.
- Top-level publication/facing date-valued columns remain admitted. When nested
  under a strict marker family, a neutral date is not promoted into affirmative
  public evidence. Finite numeric publication data likewise remains admitted,
  while boolean false asserts privacy. Arbitrary neutral text is not guessed at
  or scanned as natural language; it fails closed for publication/strict policy.
- Access/audience deliberately retain #2060's value-label semantics: known
  private labels and malformed numeric markers reject, while producer-defined
  neutral text remains admissible. Treating every unknown text label as private
  would reverse the issue's required public/neutral error direction.
- A blank or `None` beside one neutral access/audience label is representation
  noise and therefore reuses the scalar decision, including through a nested
  placeholder-only wrapper. Two substantive neutral labels remain a genuinely
  multi-valued unresolved marker and stay fail-closed.
- Review thread `PRRT_kwDOQ5Uhrs6P-Uf_` is not implemented as an admission
  widening: a valid date/number does not erase an unrecognized textual sibling
  under publication/strict policy. Admitting that object would contradict the
  operator's explicit fail-closed-on-any-unknown-text contract and would let the
  same class reopen beside data-shaped evidence. Recognized-public plus valid
  date/finite-number controls remain admitted.
- Multi-token public-audience phrases are contextual vocabulary, not universal
  overrides. Private-family fields preserve their negative polarity, while the
  exact scalar label `public` remains the existing explicit polarity override.
- `has_access` is a documented producer data column, so its normalized exact
  spellings are exempted before prefix stripping. The exemption does not apply
  to `access`, `access_label`, `has_access_flag`, or other marker-shaped keys.
- Unknown-key subtrees remain opaque at every level. Recursing into arbitrary
  producer metadata would replace the classifier's closed vocabulary with an
  unbounded structural scan and create inconsistent over-scrub risk; recognized
  nested privacy keys remain fail-closed. Review threads
  `PRRT_kwDOQ5Uhrs6P0VhX` and `PRRT_kwDOQ5Uhrs6P0Vha` are waived on this basis.
- The downstream campaign-source scalar guard remains defense-in-depth in this
  slice; broadening it across non-ticket campaign inputs would widen blast
  radius without fixing the generic provider-authority root.
- The maturity baseline is not widened. The attributable score increase was an
  unguarded first-token index, not the intentional fail-closed exception handler;
  aggregating semantic evidence removes that index. The unrelated seller-campaign
  INTERNAL_MOCK count remains unchanged at its pre-existing baseline.

## Deferred

- #2060 S6A.2: make normalized support-ticket `source_material` authoritative in
  `ContentOpsInputPackage` merge and prove generic `/execute` cannot restore raw
  rejected rows. This is required before #2060 closes.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_support_ticket_privacy_sweep.py
  tests/test_extracted_support_ticket_input_package.py -k
  'public_marker_composes_with_admitted_content or
  rendered_empty_content_wrappers_are_sequence_identity or
  rendered_alphanumeric_content_remains_substantive or
  recognized_failing_markers_dominate_at_every_content_depth or
  locally_neutral_subfields_export_no_positive_provenance or
  rejects_structured_private_content_columns or
  ignores_blank_marker_placeholders' -q` (`139 passed, 5101 deselected`).
- `python -m pytest tests/test_support_ticket_privacy.py
  tests/test_support_ticket_privacy_sweep.py
  tests/test_extracted_support_ticket_input_package.py
  tests/test_extracted_content_deflection_submit.py -q` (`5535 passed`).
- `python -m pytest tests/test_support_ticket_privacy_sweep.py -k
  'placeholder_only_marker_wrappers or
  content_carveout_requires_rendered_text or
  structured_failing_markers_dominate_content_carveout or
  neutral_subfield_policy_precedes_outer_family_policy' -q` (`79 passed,
  4865 deselected`).
- `python -m pytest tests/test_support_ticket_privacy_sweep.py -k
  generated_reference_model_matches_supported_cross_product -q` (`1 passed,
  4864 deselected`; failed before the runtime fix on masked
  `2026-02-30x`, then passed after the canonical publication-data decision).
- `python -m pytest tests/test_support_ticket_privacy_sweep.py
  tests/test_extracted_support_ticket_input_package.py -k
  'generated_reference_model or
  closed_families_reject_unrecognized_text_by_construction or
  data_families_retain_neutral_text_by_construction or
  closed_families_retain_recognized_public_values or
  public_flags_admit_explicit_date_and_number_data_shapes or
  public_flags_reject_invalid_data_shapes or
  public_flags_reject_nonfinite_data_beside_public_evidence or
  public_flags_keep_valid_data_beside_public_evidence or
  blank_is_identity_across_every_family_and_container or
  empty_nested_sequences_are_identity_across_every_family or
  content_carveout_requires_nonblank_text or
  content_carveout_keeps_nested_nonblank_text or
  public_audience_phrases_do_not_invert_private_polarity or
  ignores_blank_marker_placeholders or
  rejects_structured_private_content_columns or
  keeps_neutral_data_family_text or
  unprefixed_public_flag_keys or
  public_flag_label_status_aliases or
  public_evidence_cannot_mask_recognized_failing_values or
  public_evidence_keeps_genuinely_neutral_metadata or
  nested_content_privacy_markers_dominate_carveout or
  nested_content_public_markers_keep_genuine_text or
  boolish_exponent_numerics' -q`
  (`4181 passed, 923 deselected`).
- The generated reference-model test independently evaluates `12884`
  family/key/value/wrapper/sibling/content comparisons without importing
  runtime-private helpers or constants.
- `python -m pytest tests/test_extracted_support_ticket_input_package.py tests/test_smoke_content_ops_support_ticket_package.py tests/test_extracted_support_ticket_zendesk_export.py -k 'private or public or visibility or internal or placeholder or neutral_data_family' -q`
  (`176 passed, 123 deselected`).
- Bash `scripts/validate_extracted_content_pipeline.sh` (passed).
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  (clean).
- `python scripts/audit_extracted_standalone.py --fail-on-debt` (0 findings).
- Bash `scripts/check_ascii_python.sh` (passed).
- Bash `scripts/run_extracted_pipeline_checks.sh`
  (passed; one third-party `pynvml` deprecation warning).
- Python `scripts/maturity_sweep.py` with the workflow's exact
  extracted-content-pipeline baseline, threshold, and sensitive globs (passed;
  no baseline diff).
- `python scripts/sync_pr_plan.py
  plans/PR-Resolution-Audit-S6A-Structured-Privacy-Semantics.md origin/main --check`
  (passed).

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` | 29 |
| `extracted_content_pipeline/support_ticket_privacy.py` | 1126 |
| `plans/PR-Resolution-Audit-S6A-Structured-Privacy-Semantics.md` | 717 |
| `tests/test_extracted_content_deflection_submit.py` | 61 |
| `tests/test_extracted_support_ticket_input_package.py` | 276 |
| `tests/test_support_ticket_privacy_sweep.py` | 1665 |
| **Total** | **3874** |
