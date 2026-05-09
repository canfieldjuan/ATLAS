/**
 * Content Ops domain contract pin.
 *
 * Feeds each wire fixture (committed in `src/api/__fixtures__/`)
 * through the corresponding `fromWire*` mapper and asserts the
 * result satisfies the domain type. Same `Loosen<T>` + `satisfies`
 * gate pattern as `src/api/contentOps.contract.ts` from PR #404.
 *
 * If a wire field is renamed and the mapper isn't updated, the
 * mapper trips at compile time (its return statement won't match
 * the domain type). If a domain type is updated without the mapper
 * being updated, the satisfies check trips here.
 */

import type {
  ContentOpsCatalogResponse,
  ContentOpsExecutionResult as WireExecutionResult,
  ContentOpsPreviewResponse,
  GenerationPlanResponse,
} from '../../api/contentOps'
import catalogFixture from '../../api/__fixtures__/contentOps/catalog.json'
import executionBlockedFixture from '../../api/__fixtures__/contentOps/execution-blocked.json'
import executionCompletedFixture from '../../api/__fixtures__/contentOps/execution-completed.json'
import executionFailedFixture from '../../api/__fixtures__/contentOps/execution-failed.json'
import executionPartialFixture from '../../api/__fixtures__/contentOps/execution-partial.json'
import planBlockedFixture from '../../api/__fixtures__/contentOps/plan-blocked.json'
import planRunnableFixture from '../../api/__fixtures__/contentOps/plan-runnable.json'
import previewBlockedFixture from '../../api/__fixtures__/contentOps/preview-blocked.json'
import previewCanRunFixture from '../../api/__fixtures__/contentOps/preview-can-run.json'
import {
  fromWireCatalog,
  fromWireExecution,
  fromWirePlan,
  fromWirePreview,
} from './fromWire'
import type {
  ContentOpsCatalog,
  ContentOpsExecutionResult,
  ControlSurfacePreview,
  GenerationPlan,
} from './types'

// `Loosen<T>` widens literal-string unions in `T` to `string` so
// JSON's inferred `status: string` is compatible at the structure
// layer; literal vocabulary is gated separately by the `Record<>`
// blocks in `src/api/contentOps.contract.ts`.
type Loosen<T> = T extends string
  ? string
  : T extends Array<infer U>
    ? Loosen<U>[]
    : T extends object
      ? { [K in keyof T]: Loosen<T[K]> }
      : T

// Pin each fixture -> domain type via the wire-typed cast at the
// fixture import site. The fixture cast is safe because PR #404
// already pinned the fixtures to the wire shape.

export const __domainCatalogContract = fromWireCatalog(
  catalogFixture as ContentOpsCatalogResponse,
) satisfies Loosen<ContentOpsCatalog>

export const __domainPreviewCanRunContract = fromWirePreview(
  previewCanRunFixture as ContentOpsPreviewResponse,
) satisfies Loosen<ControlSurfacePreview>

export const __domainPreviewBlockedContract = fromWirePreview(
  previewBlockedFixture as ContentOpsPreviewResponse,
) satisfies Loosen<ControlSurfacePreview>

export const __domainPlanRunnableContract = fromWirePlan(
  planRunnableFixture as GenerationPlanResponse,
) satisfies Loosen<GenerationPlan>

export const __domainPlanBlockedContract = fromWirePlan(
  planBlockedFixture as GenerationPlanResponse,
) satisfies Loosen<GenerationPlan>

export const __domainExecutionCompletedContract = fromWireExecution(
  executionCompletedFixture as WireExecutionResult,
) satisfies Loosen<ContentOpsExecutionResult>

export const __domainExecutionPartialContract = fromWireExecution(
  executionPartialFixture as WireExecutionResult,
) satisfies Loosen<ContentOpsExecutionResult>

export const __domainExecutionFailedContract = fromWireExecution(
  executionFailedFixture as WireExecutionResult,
) satisfies Loosen<ContentOpsExecutionResult>

export const __domainExecutionBlockedContract = fromWireExecution(
  executionBlockedFixture as WireExecutionResult,
) satisfies Loosen<ContentOpsExecutionResult>
