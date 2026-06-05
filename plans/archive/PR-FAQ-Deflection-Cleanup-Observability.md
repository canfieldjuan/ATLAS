# PR-FAQ-Deflection-Cleanup-Observability

## Why this slice exists

PR-FAQ-Deflection-Private-Blob-Cleanup closed the orphaned private Blob cleanup
gap and intentionally deferred production observability for cleanup failures
until the portfolio API had a shared logging/telemetry surface. Today the
cleanup failure path emits an inline `console.warn`, which is enough to avoid
silence but not enough to give later portfolio server routes a reusable,
sanitized event shape.

This slice adds the smallest shared server-side event helper for the FAQ
deflection portfolio API and routes cleanup failures through it. The cleanup
write remains secondary and best-effort.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Production hardening

1. Add a deflection-scoped server event helper for sanitized structured warning
   events.
2. Route private Blob cleanup failures through that helper with the safe
   validated Blob pathname and a fixed public error reason.
3. Prove event-field redaction and logger-failure isolation in the focused
   upload shell tests.
4. Keep browser responses, Blob cleanup timing, and ATLAS submit behavior
   unchanged.

### Files touched

- `plans/PR-FAQ-Deflection-Cleanup-Observability.md`
- `portfolio-ui/api/content-ops/deflection/events.js`
- `portfolio-ui/api/content-ops/deflection/submit.js`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

`events.js` exports `emitDeflectionServerEvent()`. It accepts an event name,
field object, and optional logger; drops secret-shaped field names, redacts
secret-shaped string values, bounds remaining strings, and catches logger
failures so observability cannot break a request path.

`cleanupPrivateCsvBlob()` keeps the existing delete catch, but emits:

```js
emitDeflectionServerEvent("faq_deflection_private_blob_cleanup_failed", {
  pathname: safePathname,
  error: "delete_failed",
});
```

The helper returns event metadata for tests only. Route responses still expose
only the existing submit success or failure payload.

## Intentional

- This is not a full telemetry backend. It creates a shared, sanitized event
  surface for the portfolio deflection API while preserving Vercel log
  compatibility.
- The cleanup event includes the validated Blob pathname and a fixed public
  error reason, not the Blob token, URL, ATLAS JWT, contact email, SDK error
  message, or CSV content.
- Logger failures are swallowed by the event helper because logging is a
  secondary write after the submit path has already produced a result.
- Cleanup still does not run for invalid, unavailable, or oversized Blob reads.

## Deferred

- Shipping these server events to a durable sink remains deferred until the
  portfolio deployment has an operator-selected telemetry backend.

Parked hardening: none.

## Verification

- `npm run test:deflection-upload-shell --prefix portfolio-ui` -- 18 checks passed.
- `npm run test:deflection-result --prefix portfolio-ui` -- 12 checks passed.
- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` -- 14 checks passed.
- `npm run build --prefix portfolio-ui` -- passed after hydrating local
  `portfolio-ui/node_modules` with `npm install --prefix portfolio-ui`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 93 |
| Event helper | 43 |
| Submit wiring | 11 |
| Focused tests | 137 |
| **Total** | **284** |

Current diff is 4 files, +242 / -42. Under the 400 LOC soft cap.
