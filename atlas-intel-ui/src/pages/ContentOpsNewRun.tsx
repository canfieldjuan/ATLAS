import { useMemo, useRef, useState } from 'react'
import { Loader2, Play, RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import {
  fetchContentOpsControlSurfaces,
  previewContentOpsRun,
} from '../api/contentOps'
import {
  fromWireCatalog,
  fromWirePreview,
  fromWireRequest,
  toWireRequest,
  type ContentOpsCatalog,
  type ContentOpsRequest,
  type ControlSurfacePreview,
} from '../domain/contentOps'
import useApiData from '../hooks/useApiData'
import { PageError } from '../components/ErrorBoundary'

type SubmitState =
  | { kind: 'idle' }
  | { kind: 'submitting' }
  | { kind: 'invalid_inputs_json'; message: string }
  | { kind: 'error'; message: string }
  | { kind: 'success'; preview: ControlSurfacePreview }

const DEFAULT_INPUTS_JSON = '{\n  \n}'

export default function ContentOpsNewRun() {
  const {
    data: wireCatalog,
    loading,
    error,
    refresh,
    refreshing,
  } = useApiData(() => fetchContentOpsControlSurfaces(), [])

  const catalog = useMemo<ContentOpsCatalog | null>(
    () => (wireCatalog ? fromWireCatalog(wireCatalog) : null),
    [wireCatalog],
  )

  const [request, setRequest] = useState<ContentOpsRequest>(() =>
    fromWireRequest({}),
  )
  const [inputsJson, setInputsJson] = useState<string>(DEFAULT_INPUTS_JSON)
  const [submitState, setSubmitState] = useState<SubmitState>({ kind: 'idle' })
  // Codex P2 fix: request-id ref so a stale in-flight preview response
  // can't overwrite a verdict the user has since invalidated by editing
  // the form. Mirrors the pattern in src/hooks/useApiData.ts.
  const submitRequestIdRef = useRef(0)

  if (error) {
    return <PageError error={error} onRetry={refresh} />
  }
  if (loading || !catalog) {
    return (
      <div className="flex items-center justify-center py-24 text-slate-400">
        <Loader2 className="mr-2 h-5 w-5 animate-spin" />
        Loading control surfaces…
      </div>
    )
  }

  // Codex P2 fix: any form mutation invalidates a stale preview verdict so
  // the user never sees a "Can run" badge that doesn't match the current
  // form state. Bumping the request id also drops any in-flight preview
  // response so it can't overwrite the verdict for a newer form state.
  const markStale = () => {
    submitRequestIdRef.current += 1
    setSubmitState((prev) => (prev.kind === 'idle' ? prev : { kind: 'idle' }))
  }

  const togglePreset = (presetId: string) => {
    const preset = catalog.presets.find((p) => p.id === presetId)
    if (!preset) return
    setRequest((prev) => ({
      ...prev,
      preset: prev.preset === presetId ? null : presetId,
      outputs: prev.preset === presetId ? prev.outputs : [...preset.outputs],
    }))
    markStale()
  }

  const toggleOutput = (outputId: string) => {
    setRequest((prev) => {
      const has = prev.outputs.includes(outputId)
      return {
        ...prev,
        outputs: has
          ? prev.outputs.filter((o) => o !== outputId)
          : [...prev.outputs, outputId],
      }
    })
    markStale()
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    // Bump the id so any prior in-flight request is ignored on resolve.
    const requestId = ++submitRequestIdRef.current
    setSubmitState({ kind: 'submitting' })

    let parsedInputs: Record<string, unknown>
    try {
      const trimmed = inputsJson.trim()
      parsedInputs = trimmed ? JSON.parse(trimmed) : {}
      if (typeof parsedInputs !== 'object' || Array.isArray(parsedInputs)) {
        throw new Error('inputs must be a JSON object')
      }
    } catch (err) {
      if (requestId !== submitRequestIdRef.current) return
      setSubmitState({
        kind: 'invalid_inputs_json',
        message: err instanceof Error ? err.message : String(err),
      })
      return
    }

    // Submit-time normalization: backend's pydantic validator declares
    // max_cost_usd with gt=0; collapse 0 / negative to null so we never
    // round-trip a request the API would reject.
    const normalizedMaxCost =
      request.maxCostUsd === null ||
      !Number.isFinite(request.maxCostUsd) ||
      request.maxCostUsd <= 0
        ? null
        : request.maxCostUsd
    const domainRequest: ContentOpsRequest = {
      ...request,
      inputs: parsedInputs,
      maxCostUsd: normalizedMaxCost,
    }
    try {
      const wirePreview = await previewContentOpsRun(toWireRequest(domainRequest))
      // Drop the response if a newer mutation / submission has happened.
      if (requestId !== submitRequestIdRef.current) return
      setSubmitState({ kind: 'success', preview: fromWirePreview(wirePreview) })
    } catch (err) {
      if (requestId !== submitRequestIdRef.current) return
      setSubmitState({
        kind: 'error',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-100">Content Ops · New Run</h1>
          <p className="mt-1 text-sm text-slate-400">
            Pick a preset or outputs, supply inputs, preview the run before
            you commit.
          </p>
        </div>
        <button
          type="button"
          onClick={refresh}
          disabled={refreshing}
          className="flex items-center gap-2 rounded-md border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800 disabled:opacity-50"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          Refresh catalog
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Preset picker */}
        <section>
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-slate-400">
            Preset
          </h2>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {catalog.presets.map((preset) => {
              const selected = request.preset === preset.id
              return (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => togglePreset(preset.id)}
                  className={clsx(
                    'rounded-lg border px-4 py-3 text-left transition',
                    selected
                      ? 'border-cyan-500 bg-cyan-500/10 text-slate-100'
                      : 'border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500',
                  )}
                >
                  <div className="text-sm font-medium">{preset.label}</div>
                  <div className="mt-0.5 text-xs text-slate-400">{preset.description}</div>
                  <div className="mt-2 text-xs text-slate-500">
                    {preset.outputs.join(' · ')}
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        {/* Output picker */}
        <section>
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-slate-400">
            Outputs
          </h2>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {catalog.outputs.map((output) => {
              const checked = request.outputs.includes(output.id)
              return (
                <label
                  key={output.id}
                  className={clsx(
                    'flex cursor-pointer items-start gap-3 rounded-lg border px-3 py-2 transition',
                    checked
                      ? 'border-cyan-500 bg-cyan-500/10'
                      : 'border-slate-700 bg-slate-900 hover:border-slate-500',
                    !output.implemented && 'opacity-60',
                  )}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleOutput(output.id)}
                    className="mt-1 h-4 w-4 rounded border-slate-600 bg-slate-800 accent-cyan-500"
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2 text-sm font-medium text-slate-100">
                      {output.label}
                      {!output.canExecute && output.implemented && (
                        <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-400">
                          host service not configured
                        </span>
                      )}
                      {!output.implemented && (
                        <span className="rounded bg-slate-700 px-1.5 py-0.5 text-[10px] text-slate-400">
                          coming soon
                        </span>
                      )}
                    </div>
                    <div className="mt-0.5 text-xs text-slate-400">{output.description}</div>
                    <div className="mt-1 text-[11px] text-slate-500">
                      requires:{' '}
                      {output.requiredInputs.length === 0
                        ? '(none)'
                        : output.requiredInputs.join(', ')}{' '}
                      · est ${output.estimatedRetryAdjustedUnitCostUsd.toFixed(2)}/item
                    </div>
                  </div>
                </label>
              )
            })}
          </div>
        </section>

        {/* Inputs (v0: raw JSON) */}
        <section>
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-slate-400">
            Inputs (JSON)
          </h2>
          <p className="mb-2 text-xs text-slate-500">
            Required keys depend on the selected outputs (see chip list
            above). Type a JSON object; the dynamic per-output form ships
            in a follow-up slice.
          </p>
          <textarea
            value={inputsJson}
            onChange={(e) => {
              setInputsJson(e.target.value)
              markStale()
            }}
            rows={6}
            spellCheck={false}
            className="w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-2 font-mono text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
            placeholder='{"target_account": "Acme", "offer": "Audit"}'
          />
        </section>

        {/* Options */}
        <section>
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-slate-400">
            Options
          </h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <label className="block text-sm">
              <span className="text-slate-300">Limit</span>
              <input
                type="number"
                min={1}
                max={1000}
                value={request.limit}
                onChange={(e) => {
                  setRequest((p) => ({
                    ...p,
                    limit: Math.max(1, Math.min(1000, Number(e.target.value) || 1)),
                  }))
                  markStale()
                }}
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-none"
              />
            </label>
            <label className="block text-sm">
              <span className="text-slate-300">Max cost (USD)</span>
              <input
                type="number"
                step={0.01}
                min={0}
                value={request.maxCostUsd ?? ''}
                onChange={(e) => {
                  // Codex P2 fix v2: keep 0 valid in form state during typing
                  // so the user can keep keying a sub-dollar value like "0.50"
                  // (the prior `<= 0 -> null` clobbered the input the moment
                  // they typed "0"). The submit handler normalizes 0 / negative
                  // values to null before sending, matching the backend's
                  // `gt=0` validator (control_surfaces.py:88).
                  const v = e.target.value
                  const parsed = v === '' ? null : Number(v)
                  setRequest((p) => ({
                    ...p,
                    maxCostUsd:
                      parsed === null || !Number.isFinite(parsed) ? null : parsed,
                  }))
                  markStale()
                }}
                placeholder="(no cap)"
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-none"
              />
            </label>
            <label className="block text-sm">
              <span className="text-slate-300">Ingestion profile</span>
              <select
                value={request.ingestionProfile}
                onChange={(e) => {
                  setRequest((p) => ({ ...p, ingestionProfile: e.target.value }))
                  markStale()
                }}
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-none"
              >
                {catalog.ingestionProfiles.map((profile) => (
                  <option key={profile} value={profile}>
                    {profile}
                  </option>
                ))}
              </select>
            </label>
            <div className="space-y-2 text-sm">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={request.requireQualityGates}
                  onChange={(e) => {
                    setRequest((p) => ({ ...p, requireQualityGates: e.target.checked }))
                    markStale()
                  }}
                  className="h-4 w-4 rounded border-slate-600 bg-slate-800 accent-cyan-500"
                />
                <span className="text-slate-300">Require quality gates</span>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={request.allowUnimplementedOutputs}
                  onChange={(e) => {
                    setRequest((p) => ({
                      ...p,
                      allowUnimplementedOutputs: e.target.checked,
                    }))
                    markStale()
                  }}
                  className="h-4 w-4 rounded border-slate-600 bg-slate-800 accent-cyan-500"
                />
                <span className="text-slate-300">Allow unimplemented outputs</span>
              </label>
            </div>
          </div>
        </section>

        {/* Submit */}
        <section className="flex items-center justify-between border-t border-slate-800 pt-4">
          <div className="text-xs text-slate-500">
            {request.outputs.length === 0
              ? 'Pick at least one output (or a preset).'
              : `${request.outputs.length} output(s) selected.`}
          </div>
          <button
            type="submit"
            disabled={
              submitState.kind === 'submitting' || request.outputs.length === 0
            }
            className="flex items-center gap-2 rounded-md bg-cyan-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-cyan-400 disabled:opacity-50"
          >
            {submitState.kind === 'submitting' ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            Preview
          </button>
        </section>
      </form>

      {/* Preview verdict */}
      {submitState.kind === 'invalid_inputs_json' && (
        <div className="mt-6 rounded-md border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          Invalid inputs JSON: {submitState.message}
        </div>
      )}
      {submitState.kind === 'error' && (
        <div className="mt-6 rounded-md border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          Preview failed: {submitState.message}
        </div>
      )}
      {submitState.kind === 'success' && (
        <PreviewVerdict preview={submitState.preview} />
      )}
    </div>
  )
}

function PreviewVerdict({ preview }: { preview: ControlSurfacePreview }) {
  return (
    <section className="mt-8 rounded-lg border border-slate-800 bg-slate-900/60 p-5">
      <div className="mb-3 flex items-center gap-3">
        <span
          className={clsx(
            'rounded-full px-3 py-0.5 text-xs font-medium',
            preview.canRun
              ? 'bg-emerald-500/20 text-emerald-300'
              : 'bg-amber-500/20 text-amber-200',
          )}
        >
          {preview.canRun ? 'Can run' : 'Blocked'}
        </span>
        <span className="text-sm text-slate-400">
          Estimated cost: ${preview.estimatedCostUsd.toFixed(2)}
        </span>
      </div>

      {preview.outputs.length > 0 && (
        <Section label="Outputs">
          <div className="text-sm text-slate-200">
            {preview.outputs.join(', ')}
          </div>
        </Section>
      )}

      {preview.missingInputs.length > 0 && (
        <Section label="Missing inputs">
          <ul className="ml-4 list-disc text-sm text-amber-200">
            {preview.missingInputs.map((name) => (
              <li key={name}>{name}</li>
            ))}
          </ul>
        </Section>
      )}

      {preview.blockedOutputs.length > 0 && (
        <Section label="Blocked outputs">
          <ul className="ml-4 list-disc text-sm text-amber-200">
            {preview.blockedOutputs.map((name) => (
              <li key={name}>{name}</li>
            ))}
          </ul>
        </Section>
      )}

      {preview.warnings.length > 0 && (
        <Section label="Warnings">
          <ul className="ml-4 list-disc text-sm text-slate-300">
            {preview.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </Section>
      )}

      {preview.normalizedRequest && (
        <Section label="Normalized request">
          <pre className="overflow-x-auto rounded-md bg-slate-950/80 p-3 font-mono text-xs text-slate-300">
            {JSON.stringify(preview.normalizedRequest, null, 2)}
          </pre>
        </Section>
      )}
    </section>
  )
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mt-3 first:mt-0">
      <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-slate-500">
        {label}
      </div>
      {children}
    </div>
  )
}
