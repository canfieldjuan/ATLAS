# PR-Content-Factory-Runner

## Why this slice exists

The Content Factory has contracts (#2116) and an artifact store (#2121), but no
code that actually runs a worker and lands its output. The Phase 1.4 end-to-end
run did this with an ad-hoc scratchpad script that hit Open WebUI's chat API and
wrote files with no contract validation. This slice makes that a real, tested
primitive: call a worker via Open WebUI, robustly extract its JSON artifact, and
validate + persist it via the store. It is the bridge between the OWUI worker
wrappers and the atlas_brain pipeline, and the unit the future multi-stage
orchestrator calls once per stage.

### Problem-derived contract

A correct fix must:
- Call a worker by model id through Open WebUI's chat completions and return the
  assistant text, failing loudly (not silently) on transport/HTTP errors, a
  non-JSON body, or a response with no assistant message.
- Extract a single JSON object from the reply, tolerating code fences and
  surrounding prose (weak local models wrap output), and return None when none
  parses -- never a partial or non-dict value.
- Validate + persist through the store, so a malformed worker output is caught by
  the contract and never lands on disk.
- Live in atlas_brain (it uses the store + contracts); Open WebUI is the external
  boundary and is the only thing mocked in tests.
- Be model-agnostic: it addresses a worker by id and cares nothing about which
  model backs it.

## Scope (this PR)

Ownership lane: content-factory
Slice phase: vertical slice

Arc Phase 2.2b: the single-stage runner. The multi-stage orchestrator (sequence,
retry cap, manifest assembly, human-approval states) is a later slice.

### Review Contract

- Acceptance criteria:
  - [ ] call_worker POSTs one user turn to OWUI chat completions and returns the
        assistant text; a transport/HTTP error, non-JSON body, or missing
        assistant message raises WorkerError (no silent failure).
  - [ ] extract_json returns the embedded JSON object for clean, fenced, and
        prose-wrapped replies, and None for no-JSON, invalid-JSON, and non-dict
        (array) replies.
  - [ ] run_stage calls the worker, extracts, then validates + persists via the
        store; no-JSON raises WorkerError and a contract-invalid artifact raises
        ValidationError, and in both cases nothing is persisted.
- Reachability proof: N/A for a production surface -- no runtime caller yet (the
  orchestrator is a later slice). Proof is the test suite with OWUI mocked and a
  real store against a temp filesystem.
- Affected surfaces: one new runner module and its test file; no existing file
  modified; nothing imports the module yet.
- Risk areas: OWUI response shape handling (mocked both success and error paths);
  JSON extraction from noisy replies (boundary-probed); fail-closed persistence
  is delegated to the already-tested store.
- Reviewer rules triggered: R2.

### Files touched
- `atlas_brain/services/content_factory_runner.py`
- `tests/test_content_factory_runner.py`
- `plans/PR-Content-Factory-Runner.md`

Max files: 3

## Mechanism

call_worker builds a stream=false chat-completions POST to Open WebUI for a model
id (bearer api_key) and returns choices[0].message.content, wrapping URLError
(HTTPError included), a non-JSON body, and a missing message in WorkerError.
extract_json strips a leading/trailing code fence, takes the first-brace-to-last-
brace slice, json.loads it, and returns it only if it is a dict. run_stage chains
them and hands the artifact to the store's write_artifact, which does all contract
validation and path-guarded, git-tracked persistence. api_key and base_url are
parameters (no secret in code, no config surface this slice); root defaults to
the store's default and is overridable for tests.

## Intentional

- Open WebUI is the only mocked boundary; the store and contracts are real, so
  the extract + validate + persist path is exercised for real (real-adapters rule).
- No secret and no config.py surface: api_key/base_url/root are parameters.
- Single-stage only: sequencing, retries, and manifest assembly are the
  orchestrator's job (later), keeping this unit small and composable.

## Deferred

- The multi-stage orchestrator (stage order, retry cap, needs-human states,
  manifest assembly) -- a later slice.
- A typed config field for the OWUI base URL / key (add when a runtime caller
  needs it).
- An OWUI-side Action button that triggers a stage from the chat UI (optional UI).

## Verification

```
python -m pytest tests/test_content_factory_runner.py -q
```
16 tests pass with Open WebUI mocked and a real store on a temp filesystem:
extract_json handles clean/fenced/prose replies and rejects no-JSON, invalid, and
array replies; call_worker returns content and raises WorkerError on HTTP error
and a missing message; run_stage persists a valid artifact, extracts from fenced
prose, and on no-JSON or a contract-invalid artifact raises and persists nothing.

## Estimated diff size

| File | Lines |
|---|---|
| atlas_brain/services/content_factory_runner.py | 100 |
| tests/test_content_factory_runner.py | 142 |
| plans/PR-Content-Factory-Runner.md | 100 |
| **Total** | **342** |
