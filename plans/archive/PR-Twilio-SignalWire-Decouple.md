# PR-Twilio-SignalWire-Decouple

Ownership lane: comms-provider

## Why this slice exists

The `signalwire` Python SDK (`signalwire==2.0.4`, latest 2.1.1 on PyPI)
hard-pins `twilio==6.54.0`, which pins `pyjwt==1.7.1` (and old boto3, etc.).
This freezes the ASR/comms dependency stack: Dependabot cannot bump twilio
to 9 (#2092) and the python-security-and-patches group (#2098) cannot
install pyjwt 2.x -- the `root-asr-constraints` resolution is unsatisfiable
("Because twilio==6.54.0 depends on pyjwt==1.7.1 ... unsatisfiable"). No
signalwire release relaxes the pin (2.1.1, the latest, still requires
`twilio==6.54.0`), so the SDK is effectively abandoned.

The `signalwire` SDK is only a thin monkey-patch wrapper over `twilio.rest`
that overrides the Api-domain `base_url` to the SignalWire space; SignalWire
serves the twilio-compatible LaML REST API at
`https://<space>.signalwire.com/api/laml/2010-04-01/...`. So the wrapper is
removable: the plain `twilio` SDK reaches SignalWire by pointing its client
at the space's `/api/laml` root. Decoupling permanently removes the pin
blocking twilio 9 and the python security group.

### Problem-derived contract

- Root cause: `signalwire==2.0.4` hard-pins `twilio==6.54.0` (which pins
  `pyjwt==1.7.1`). The signalwire SDK is a wrapper that only rewrites the
  twilio client's `base_url` to the space; nothing in atlas_comms needs the
  wrapper itself.
- Correct fix must touch/change:
  1. `requirements.txt`: remove `signalwire==2.0.4`; promote `twilio`
     (currently a commented-out alternative) to a direct dependency at
     `>=9.10.9,<10`. twilio was only present transitively via signalwire.
  2. `atlas_comms/providers/signalwire.py` `connect()`: build a plain
     `twilio.rest.Client(project_id, api_token)` and set
     `client.api.base_url = f"https://{space}.signalwire.com/api/laml"`
     (SignalWire's LaML compat root) instead of `signalwire.rest.Client`.
     The rest of the provider is unchanged -- the API surface is identical.
  3. `atlas_brain/mcp/twilio_server.py` `_get_client()`: the SAME
     `signalwire.rest.Client(signalwire_space_url=...)` branch must switch
     to `twilio.rest.Client` + `client.api.base_url = ".../api/laml"`, or
     the MCP SignalWire call/SMS/recording tools break once the package is
     removed.
  4. `constraints.root-asr.txt`: regenerate via
     `scripts/compile_root_asr_constraints.py` (pinned uv) so the compiled
     lock no longer pins `twilio==6.43.0`; otherwise `requirements.txt`'s
     `-c constraints.root-asr.txt` keeps the resolution unsatisfiable. The
     regen also refreshes the `constraints-root-asr-sha256` digest in
     `requirements.txt`.
  5. Tests: a regression test asserting the provider's client points at the
     space's `/api/laml` root and that messages/calls resolve to
     `/api/laml/2010-04-01/...`, plus the missing-credentials guard.
- Must not change:
  - Every other method in `atlas_comms/providers/signalwire.py` and the
    twilio branch of `atlas_brain/mcp/twilio_server.py` -- identical twilio
    API, untouched.
  - `atlas_comms/providers/twilio.py` -- inherits twilio 9, no code change.
  - pyjwt/boto3/other requirement pins beyond what the lock regen resolves
    -- the python security group (#2098) owns those.
  - Config, atlas_brain comms code, formatting.

Contract revision (post-Codex round 1): the original contract missed three
surfaces, all confirmed by Codex and fixed here -- (a) the compat root is
`/api/laml/2010-04-01`, not `/2010-04-01`; (b) `atlas_brain/mcp/twilio_server.py` has a
second `signalwire.rest.Client` site; (c) `constraints.root-asr.txt` itself
pins `twilio==6.43.0`, so requirements.txt alone left the resolution
unsatisfiable. Change surface expanded from 3 files to 5 (+ the regenerated
lock/digest).

Empirical pre-checks: twilio 9.10.9 `client.api.base_url =
"https://<space>.signalwire.com/api/laml"` resolves messages/calls to
`https://<space>.signalwire.com/api/laml/2010-04-01/Accounts/<project>/...`
(matches the in-tree recording helper at `atlas_brain/mcp/twilio_server.py`). PyPI:
signalwire 2.1.1 requires `twilio==6.54.0`; twilio 9.10.9 requires
`PyJWT>=2.0.0,<3.0.0`. The pinned-uv lock regen succeeded (proof the tree
is now satisfiable): `twilio==9.10.9`, `pyjwt==2.13.0`, `+aiohttp`,
signalwire removed.

## Scope (this PR)

Slice phase: Vertical slice

Max files: 6

1. `requirements.txt` -- drop signalwire, add twilio 9.x direct; refreshed
   constraints digest.
2. `atlas_comms/providers/signalwire.py` -- twilio client at the `/api/laml`
   space root in `connect()`.
3. `atlas_brain/mcp/twilio_server.py` -- same substitution in the MCP
   client factory's signalwire branch.
4. `constraints.root-asr.txt` -- regenerated lock (twilio 6.43.0 -> 9.10.9,
   +aiohttp, -signalwire).
5. `tests/test_signalwire_provider_twilio_base_url.py` -- regression test.
6. This plan doc.

### Files touched

- `atlas_brain/mcp/twilio_server.py`
- `atlas_comms/providers/signalwire.py`
- `constraints.root-asr.txt`
- `plans/PR-Twilio-SignalWire-Decouple.md`
- `requirements.txt`
- `tests/test_signalwire_provider_twilio_base_url.py`

### Review Contract

Acceptance criteria:
1. `requirements.txt` no longer contains `signalwire`; `twilio` is a direct
   dep at `>=9.10.9,<10`; the `constraints-root-asr-sha256` digest matches
   the regenerated lock.
2. `atlas_comms/providers/signalwire.py` and `atlas_brain/mcp/twilio_server.py`
   import `twilio.rest`, not `signalwire.rest`, and set `client.api.base_url`
   to `https://<space>.signalwire.com/api/laml`.
3. The provider's client resolves messages/calls to
   `/api/laml/2010-04-01/Accounts/<project>/...` (proven by test).
4. `constraints.root-asr.txt` pins `twilio==9.10.9` (not 6.43.0); the pinned
   uv regen is reproducible so CI `root-asr-constraints` (--check) passes.
5. No other provider method changed; `atlas_comms/providers/twilio.py` and
   the twilio branch of `atlas_brain/mcp/twilio_server.py` untouched.

Affected surfaces: SignalWire provider + MCP client construction (SMS +
voice + recording); requirements / ASR-constraints resolution.

Risk areas: SignalWire compat-API path (`/api/laml`) verified against the
in-tree recording helper + docs; no live-credential send test in CI (the
boundary is URL wiring, unit-tested).

Reviewer rules triggered: R1, R2, R5, R6, R11, R12 (dependency/requirements change; test evidence; outward-facing comms provider + MCP runtime; constraints/lock regen).

## Mechanism

`connect()` / `_get_client()` construct `twilio.rest.Client(id, token)` and
override `client.api.base_url` to `https://<space>.signalwire.com/api/laml`;
the twilio Api domain then builds `/api/laml/2010-04-01/Accounts/<project>/`
requests against SignalWire's twilio-compatible LaML API. The lock is
regenerated with the pinned uv so `constraints.root-asr.txt` resolves twilio
9 (+ its aiohttp/pyjwt deps) instead of the old `twilio==6.43.0` pin.

## Intentional

- Decouple rather than bump signalwire: no signalwire release supports a
  modern twilio (2.1.1 still pins `twilio==6.54.0`).
- twilio promoted to a direct dep (was transitive via signalwire).
- `/api/laml` root (not bare `/2010-04-01`): matches SignalWire's documented
  compat root and the existing in-tree recording helper.
- `>=9.10.9,<10` keeps Dependabot current within v9 while guarding v10.
- Lock regenerated (not hand-edited) so the CI `--check` digest matches.

## Deferred

- The python-security-and-patches group (#2098) recreate + merge (pyjwt/
  boto3) is unblocked by this but lands separately -- its `boto3==1.43.47`
  is an independent Dependabot glitch.
- Dependabot twilio #2092 is superseded once twilio is a direct dep at 9.x.
- Parked hardening: none.

## Verification

- `python -m pytest tests/test_signalwire_provider_twilio_base_url.py -q`
- `python -c "import atlas_comms.providers.signalwire"` (no signalwire pkg)
- `python scripts/compile_root_asr_constraints.py --uv <uv 0.10.10>` wrote
  the lock (twilio 9 resolves) -- CI `root-asr-constraints --check` matches.
- Scratch-venv proof: twilio 9.10.9 `/api/laml` base_url resolves to the
  SignalWire `/api/laml/2010-04-01/...` URL.

## Estimated diff size

| File | LOC |
|---|---:|
| requirements.txt | 5 |
| atlas_comms/providers/signalwire.py | 19 |
| atlas_brain/mcp/twilio_server.py | 14 |
| constraints.root-asr.txt | 4 |
| tests/test_signalwire_provider_twilio_base_url.py | 49 |
| plans/PR-Twilio-SignalWire-Decouple.md | ~170 |
| **Total** | ~261 |
