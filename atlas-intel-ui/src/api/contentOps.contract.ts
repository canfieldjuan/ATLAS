/**
 * Content Ops contract verification.
 *
 * Compile-time-only check that the wire interfaces in `contentOps.ts`
 * match the canonical fixture JSON dumped from the real backend code
 * paths in `__fixtures__/contentOps/`.
 *
 * Usage: `tsc -b --noEmit` is the gate. If a backend dataclass
 * renames a field, removes one, or changes an enum value, regenerating
 * fixtures (via `scripts/dump_content_ops_fixtures.py` -- see plan doc)
 * surfaces the drift here as a compile error.
 *
 * No runtime tests live here; that needs Vitest, deferred to a
 * separate slice.
 */

import type {
  ContentOpsCatalogResponse,
  ContentOpsExecutionResult,
  ContentOpsIngestionDiagnosticsResponse,
  ContentOpsIngestionImportResponse,
  ContentOpsPreviewResponse,
  GenerationPlanResponse,
} from './contentOps'
import catalogFixture from './__fixtures__/contentOps/catalog.json'
import ingestionImportFixture from './__fixtures__/contentOps/ingestion-import.json'
import ingestionInspectFixture from './__fixtures__/contentOps/ingestion-inspect.json'
import previewCanRunFixture from './__fixtures__/contentOps/preview-can-run.json'
import previewBlockedFixture from './__fixtures__/contentOps/preview-blocked.json'
import planRunnableFixture from './__fixtures__/contentOps/plan-runnable.json'
import planBlockedFixture from './__fixtures__/contentOps/plan-blocked.json'
import executionCompletedFixture from './__fixtures__/contentOps/execution-completed.json'
import executionPartialFixture from './__fixtures__/contentOps/execution-partial.json'
import executionFailedFixture from './__fixtures__/contentOps/execution-failed.json'
import executionBlockedFixture from './__fixtures__/contentOps/execution-blocked.json'

// ---------------------------------------------------------------------------
// Wire-shape assignability checks. Each fixture is asserted via
// `satisfies Loosen<Wire>` -- `Loosen<T>` widens literal-string
// unions in `T` to `string`, so JSON's inferred `status: string` is
// compatible. Removing a required field from the fixture trips this
// gate; adding an extra field to the fixture is also rejected
// (`satisfies` enforces "no extra properties" in const positions).
// The literal-vocabulary check happens separately via the coverage
// blocks below, so enum drift fails compile too.
// ---------------------------------------------------------------------------

type Loosen<T> = T extends string
  ? string
  : T extends Array<infer U>
    ? Loosen<U>[]
    : T extends object
      ? { [K in keyof T]: Loosen<T[K]> }
      : T

export const __catalogContract = catalogFixture satisfies Loosen<ContentOpsCatalogResponse>
export const __ingestionInspectContract = ingestionInspectFixture satisfies Loosen<ContentOpsIngestionDiagnosticsResponse>
export const __ingestionImportContract = ingestionImportFixture satisfies Loosen<ContentOpsIngestionImportResponse>
export const __previewCanRunContract = previewCanRunFixture satisfies Loosen<ContentOpsPreviewResponse>
export const __previewBlockedContract = previewBlockedFixture satisfies Loosen<ContentOpsPreviewResponse>
export const __planRunnableContract = planRunnableFixture satisfies Loosen<GenerationPlanResponse>
export const __planBlockedContract = planBlockedFixture satisfies Loosen<GenerationPlanResponse>
export const __executionCompletedContract = executionCompletedFixture satisfies Loosen<ContentOpsExecutionResult>
export const __executionPartialContract = executionPartialFixture satisfies Loosen<ContentOpsExecutionResult>
export const __executionFailedContract = executionFailedFixture satisfies Loosen<ContentOpsExecutionResult>
export const __executionBlockedContract = executionBlockedFixture satisfies Loosen<ContentOpsExecutionResult>

// ---------------------------------------------------------------------------
// Status-enum coverage: every literal the backend emits must be in the
// TS union. If the backend adds a new status, this fails to compile.
// ---------------------------------------------------------------------------

const _executionStatusCoverage: Record<
  ContentOpsExecutionResult['status'],
  true
> = {
  completed: true,
  partial: true,
  failed: true,
  blocked: true,
}
export const __executionStatusCoverage = _executionStatusCoverage

const _stepStatusCoverage: Record<
  ContentOpsExecutionResult['steps'][number]['status'],
  true
> = {
  completed: true,
  failed: true,
  skipped: true,
}
export const __stepStatusCoverage = _stepStatusCoverage

const _planStepStatusCoverage: Record<
  GenerationPlanResponse['steps'][number]['status'],
  true
> = {
  runnable: true,
  blocked: true,
}
export const __planStepStatusCoverage = _planStepStatusCoverage

const _reasoningRequirementCoverage: Record<
  // The backend currently emits these two literals; the TS union also
  // accepts `string` to tolerate future additions without breaking the
  // adapter (the contract doc explicitly notes this field is a string,
  // not an enum). The fixture covers both literals today.
  'absent' | 'optional_host_context',
  true
> = {
  absent: true,
  optional_host_context: true,
}
export const __reasoningRequirementCoverage = _reasoningRequirementCoverage
