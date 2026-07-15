# PR-Twilio-SignalWire-Decouple

Ownership lane: comms-provider

## Why this slice exists

The `signalwire` Python SDK (`signalwire==2.0.4` in `requirements.txt`,
latest 2.1.1 on PyPI) hard-pins `twilio==6.54.0`, which transitively pins
`pyjwt==1.7.1` (and old boto3, etc.). This freezes the ASR/comms
dependency stack: Dependabot cannot bump twilio to 9 (#2092) and the
python-security-and-patches group (#2098) cannot install pyjwt 2.x -- the
`root-asr-constraints` resolution is unsatisfiable ("Because twilio==6.54.0
depends on pyjwt==1.7.1 ... your requirements are unsatisfiable"). No
signalwire release relaxes the pin (2.1.1, the latest, still requires
`twilio==6.54.0`), so the SDK is effectively abandoned.

The `signalwire` SDK is only a thin monkey-patch wrapper over `twilio.rest`
that overrides the Api-domain `base_url` to the SignalWire space;
SignalWire serves the twilio-compatible LaML REST API at
`https://<space>.signalwire.com/2010-04-01/...`. So the wrapper is
removable: the plain `twilio` SDK reaches SignalWire by pointing its
client at the space. Decoupling permanently removes the pin blocking
twilio 9 and the python security group.

### Problem-derived contract

- Root cause: `signalwire==2.0.4` hard-pins `twilio==6.54.0` (which pins
  `pyjwt==1.7.1`). The signalwire SDK is a wrapper that only rewrites the
  twilio client's `base_url` to the space; nothing in atlas_comms needs
  the wrapper itself.
- Correct fix must touch/change:
  1. `requirements.txt`: remove `signalwire==2.0.4`; promote `twilio`
     (currently a commented-out alternative) to a direct dependency at
     `>=9.10.9,<10`. twilio was only present transitively via signalwire,
     so removing signalwire without this drops twilio entirely.
  2. `atlas_comms/providers/signalwire.py` `connect()`: build a plain
     `twilio.rest.Client(project_id, api_token)` and set
     `client.api.base_url = f"https://{space}.signalwire.com"` instead of
     `signalwire.rest.Client(..., signalwire_space_url=...)`. The rest of
     the provider (`.messages.create`, `.calls.create`,
     `.calls(sid).update`, `.recordings.create`) is unchanged -- the API
     surface is identical.
  3. Tests: a regression test asserting the provider's client points at
     the space (`client.api.base_url` == space URL) and that
     messages/calls resolve to the space's `/2010-04-01/` LaML endpoint,
     plus the missing-credentials guard.
- Must not change:
  - `atlas_comms/providers/twilio.py` -- already uses `twilio.rest.Client`;
    it inherits twilio 9 with no code change (stable API surface).
  - Every other method in `signalwire.py` (make_call, send_sms, hangup,
    reject, transfer, start_recording, incoming-call/status/SMS webhook
    handlers, generate_stream_laml) -- identical twilio API, untouched.
  - pyjwt/boto3/other requirement pins -- those belong to the python
    security group (#2098), which this unblocks; not bumped here.
  - Config, atlas_brain comms code, the twilio MCP server, formatting.

Empirical pre-checks (run before coding): twilio 9.10.9 installed in a
scratch venv; `Client(project, token)` then `client.api.base_url = space`
resolves `client.messages` and `client.calls` to
`https://<space>.signalwire.com/2010-04-01/Accounts/<project>/{Messages,Calls}.json`
-- byte-for-byte what `signalwire.rest.Client` produced (its source sets
`self._api.base_url = space_url`). PyPI: signalwire 2.1.1 requires
`twilio==6.54.0`; twilio 9.10.9 requires `PyJWT>=2.0.0,<3.0.0`.

## Scope (this PR)

Slice phase: Vertical slice

Max files: 4

1. `requirements.txt` -- drop signalwire, add twilio 9.x as a direct dep.
2. `atlas_comms/providers/signalwire.py` -- twilio-client-pointed-at-space
   in `connect()`.
3. `tests/test_signalwire_provider_twilio_base_url.py` -- regression test
   for the base_url/URL wiring and the credentials guard.
4. This plan doc.

### Files touched

- `atlas_comms/providers/signalwire.py`
- `plans/PR-Twilio-SignalWire-Decouple.md`
- `requirements.txt`
- `tests/test_signalwire_provider_twilio_base_url.py`

### Review Contract

Acceptance criteria:
1. `requirements.txt` no longer contains `signalwire`; `twilio` is a direct
   dependency at `>=9.10.9,<10`.
2. `atlas_comms/providers/signalwire.py` imports `twilio.rest`, not
   `signalwire.rest`; `connect()` sets `client.api.base_url` to
   `https://<space>.signalwire.com`.
3. The provider's client resolves messages/calls to the space's
   `/2010-04-01/Accounts/<project>/...` LaML endpoint (proven by test).
4. No other provider method changed; `atlas_comms/providers/twilio.py` untouched.
5. The `twilio==6.54.0 -> pyjwt==1.7.1` constraint conflict is gone (CI
   `root-asr-constraints` no longer trips on it).

Affected surfaces: SignalWire provider client construction (SMS + voice);
requirements / ASR-constraints resolution.

Risk areas: SignalWire compat-API auth/paths could differ from the
wrapper's -- mitigated: the wrapper only set `base_url`, which we replicate
exactly, and the resulting URL construction was verified identical on
twilio 9. No live-credential send test in CI; the boundary is the URL
wiring, which is unit-tested.

Reviewer rules triggered: R1 (dependency/requirements change), R2 (test
evidence for changed behavior), R6 (outward-facing comms provider: SMS/
voice), R7 (third-party SDK boundary).

## Mechanism

`connect()` constructs `twilio.rest.Client(project_id, api_token)` (the
project id is the twilio "account_sid"/username; SignalWire uses the same
HTTP Basic auth) and overrides `client.api.base_url` to the space URL. The
twilio Api domain then builds `2010-04-01/Accounts/<project>/...` requests
against `https://<space>.signalwire.com`, which is SignalWire's
twilio-compatible LaML API. `signalwire.rest.Client` did exactly this
(`self._api.base_url = space_url`) plus cosmetic ANSI error-formatting
patches and a Fax domain that atlas_comms never uses.

## Intentional

- Decouple rather than bump signalwire: no signalwire release supports a
  modern twilio (2.1.1 still pins `twilio==6.54.0`), so bumping is
  impossible; the wrapper is removable because atlas_comms uses only the
  twilio-compatible surface (calls/messages/recordings).
- twilio promoted to a direct dep (was transitive via signalwire): without
  this, removing signalwire drops twilio and breaks BOTH providers.
- `>=9.10.9,<10` (not `==`): lets Dependabot keep twilio current within v9
  while `<10` guards a future major.
- Keep the `try/except ImportError` shape (message swapped to twilio) to
  match the sibling `atlas_comms/providers/twilio.py` provider; twilio is now a hard dep so the
  branch is defensive, not expected.

## Deferred

- The python-security-and-patches group (#2098) recreate + merge (pyjwt/
  boto3) is unblocked by this but lands separately -- its `boto3==1.43.47`
  is an independent Dependabot version glitch, not this PR's scope.
- Dependabot twilio PR #2092 is superseded once twilio is a direct dep at
  9.x (close/let Dependabot reconcile after merge).
- Twilio v9 native `region`/`edge` targeting is unused; the `base_url`
  override is the SignalWire-documented compat path. Parked hardening: none.

## Verification

- `python -m pytest tests/test_signalwire_provider_twilio_base_url.py -q`
- `python -c "import atlas_comms.providers.signalwire"` (no signalwire pkg)
- Scratch-venv proof (before coding): twilio 9.10.9 `base_url` override
  resolves messages/calls to the SignalWire `/2010-04-01/...` URL.
- CI `root-asr-constraints` resolves without the twilio/pyjwt conflict.

## Estimated diff size

| File | LOC |
|---|---:|
| requirements.txt | 3 |
| atlas_comms/providers/signalwire.py | 19 |
| tests/test_signalwire_provider_twilio_base_url.py | 49 |
| plans/PR-Twilio-SignalWire-Decouple.md | ~163 |
| **Total** | ~240 |
