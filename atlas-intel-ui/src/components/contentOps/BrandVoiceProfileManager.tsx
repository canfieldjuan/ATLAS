import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
} from 'react'
import {
  FileUp,
  Loader2,
  Palette,
  Pencil,
  Plus,
  RefreshCw,
  Save,
  Search,
  Trash2,
  X,
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  createContentOpsBrandVoiceProfile,
  deleteContentOpsBrandVoiceProfile,
  fetchContentOpsBrandVoiceSampleUrl,
  updateContentOpsBrandVoiceProfile,
  type ContentOpsBrandVoiceProfile,
} from '../../api/contentOps'
import {
  BRAND_VOICE_PROFILE_PRESETS,
  applyBrandVoiceProfileEditorPatch,
  blankBrandVoiceProfileEditorState,
  brandVoicePresetEditorPatch,
  brandVoiceProfileEditorRequest,
  brandVoiceProfileEditorStateFromProfile,
  canSaveBrandVoiceProfileEditor,
  deriveBrandVoiceProfileEditorPatch,
  type BrandVoiceProfileEditorState,
} from '../../domain/contentOps'

type BrandVoiceProfileMutationState =
  | { kind: 'idle' }
  | { kind: 'saving' }
  | { kind: 'archiving'; id: string }
  | { kind: 'success'; message: string }
  | { kind: 'error'; message: string }

type BrandVoiceSampleImportState =
  | { kind: 'idle' }
  | { kind: 'loading'; message: string }
  | { kind: 'loaded'; message: string }
  | { kind: 'applied'; message: string }
  | { kind: 'error'; message: string }

type BrandVoicePresetApplyState =
  | { kind: 'idle' }
  | { kind: 'applied'; message: string }
  | { kind: 'error'; message: string }

export function BrandVoiceProfileManager({
  profiles,
  selectedProfileId,
  loading,
  refreshing,
  error,
  onRefresh,
  onChange,
}: {
  profiles: ContentOpsBrandVoiceProfile[]
  selectedProfileId: string | null
  loading: boolean
  refreshing: boolean
  error: Error | null
  onRefresh: () => void
  onChange: (profileId: string | null) => void
}) {
  const selectedProfile = profiles.find((profile) => profile.id === selectedProfileId)
  const selectedProfileUnavailable = Boolean(selectedProfileId && !selectedProfile)
  const missingSelectedProfile = Boolean(
    selectedProfileUnavailable && !loading && !refreshing,
  )
  const [editor, setEditor] = useState<BrandVoiceProfileEditorState | null>(null)
  const [mutationState, setMutationState] =
    useState<BrandVoiceProfileMutationState>({ kind: 'idle' })
  const [sampleText, setSampleText] = useState('')
  const [sampleName, setSampleName] = useState('')
  const [sampleUrl, setSampleUrl] = useState('')
  const [selectedPresetId, setSelectedPresetId] = useState(
    BRAND_VOICE_PROFILE_PRESETS[0]?.id ?? '',
  )
  const [presetApplyState, setPresetApplyState] =
    useState<BrandVoicePresetApplyState>({ kind: 'idle' })
  const [sampleImportState, setSampleImportState] =
    useState<BrandVoiceSampleImportState>({ kind: 'idle' })
  const mutating =
    mutationState.kind === 'saving' || mutationState.kind === 'archiving'
  const sampleFetching = sampleImportState.kind === 'loading'
  const canSave = editor ? canSaveBrandVoiceProfileEditor(editor) : false
  const selectedProfileIdRef = useRef(selectedProfileId)
  const sampleFetchTokenRef = useRef(0)
  useEffect(() => {
    selectedProfileIdRef.current = selectedProfileId
  }, [selectedProfileId])

  const invalidateSampleFetch = () => {
    sampleFetchTokenRef.current += 1
  }

  const resetSampleImport = () => {
    invalidateSampleFetch()
    setSampleText('')
    setSampleName('')
    setSampleUrl('')
    setSampleImportState({ kind: 'idle' })
  }

  const startCreate = () => {
    resetSampleImport()
    setPresetApplyState({ kind: 'idle' })
    setEditor(blankBrandVoiceProfileEditorState())
    setMutationState({ kind: 'idle' })
  }

  const startEdit = () => {
    if (!selectedProfile) return
    resetSampleImport()
    setPresetApplyState({ kind: 'idle' })
    setEditor(brandVoiceProfileEditorStateFromProfile(selectedProfile))
    setMutationState({ kind: 'idle' })
  }

  const updateEditor = (
    field: keyof Omit<
      BrandVoiceProfileEditorState,
      'mode' | 'profileId' | 'metadata'
    >,
    value: string,
  ) => {
    setEditor((current) => (current ? { ...current, [field]: value } : current))
  }

  const handleSave = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!editor || !canSave || mutationState.kind === 'saving' || sampleFetching) {
      return
    }
    const mode = editor.mode
    setMutationState({ kind: 'saving' })
    try {
      const body = brandVoiceProfileEditorRequest(editor)
      const profile =
        mode === 'edit' && editor.profileId
          ? await updateContentOpsBrandVoiceProfile(editor.profileId, body)
          : await createContentOpsBrandVoiceProfile(body)
      resetSampleImport()
      onChange(profile.id)
      setEditor(null)
      setMutationState({
        kind: 'success',
        message:
          mode === 'edit'
            ? 'Brand voice profile updated.'
            : 'Brand voice profile saved.',
      })
      onRefresh()
    } catch (err) {
      setMutationState({
        kind: 'error',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const handleArchive = async () => {
    if (!selectedProfile || mutationState.kind === 'archiving') return
    const archiveProfileId = selectedProfile.id
    const archiveProfileName = selectedProfile.name
    if (
      !window.confirm(`Archive brand voice profile "${archiveProfileName}"?`)
    ) {
      return
    }
    setMutationState({ kind: 'archiving', id: archiveProfileId })
    try {
      await deleteContentOpsBrandVoiceProfile(archiveProfileId)
      if (selectedProfileIdRef.current === archiveProfileId) {
        onChange(null)
      }
      if (editor?.profileId === archiveProfileId) {
        resetSampleImport()
        setEditor(null)
      }
      setMutationState({
        kind: 'success',
        message: 'Brand voice profile archived.',
      })
      onRefresh()
    } catch (err) {
      setMutationState({
        kind: 'error',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const handleSampleFileChange = async (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    try {
      const text = await file.text()
      setSampleText(text)
      setSampleName(file.name)
      setSampleImportState({ kind: 'loaded', message: `Loaded ${file.name}.` })
    } catch (err) {
      setSampleImportState({
        kind: 'error',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const handleFetchSampleUrl = async () => {
    const url = sampleUrl.trim()
    if (!url) {
      setSampleImportState({
        kind: 'error',
        message: 'Add a URL before fetching.',
      })
      return
    }
    const requestToken = sampleFetchTokenRef.current + 1
    sampleFetchTokenRef.current = requestToken
    setSampleImportState({ kind: 'loading', message: 'Fetching URL.' })
    try {
      const sample = await fetchContentOpsBrandVoiceSampleUrl({ url })
      if (sampleFetchTokenRef.current !== requestToken) {
        return
      }
      const fallbackName =
        sample.title?.trim() || brandVoiceSampleFallbackName(sample.url)
      setSampleText(sample.text)
      setSampleName(fallbackName)
      setSampleUrl(sample.url)
      setSampleImportState({
        kind: 'loaded',
        message: `Loaded ${fallbackName}.`,
      })
    } catch (err) {
      if (sampleFetchTokenRef.current !== requestToken) {
        return
      }
      setSampleImportState({
        kind: 'error',
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const handleApplyPreset = () => {
    const patch = brandVoicePresetEditorPatch(selectedPresetId)
    if (!patch) {
      setPresetApplyState({
        kind: 'error',
        message: 'Select a preset before applying.',
      })
      return
    }
    setEditor((current) =>
      current ? applyBrandVoiceProfileEditorPatch(current, patch) : current,
    )
    setPresetApplyState({
      kind: 'applied',
      message: 'Preset applied to empty profile fields.',
    })
  }

  const handleApplySample = () => {
    if (!sampleText.trim()) {
      setSampleImportState({
        kind: 'error',
        message: 'Add sample copy before applying.',
      })
      return
    }
    const patch = deriveBrandVoiceProfileEditorPatch(sampleText, {
      fallbackName: sampleName,
    })
    setEditor((current) =>
      current ? applyBrandVoiceProfileEditorPatch(current, patch) : current,
    )
    setSampleImportState({
      kind: 'applied',
      message: 'Sample applied to empty profile fields.',
    })
  }

  return (
    <section className="mb-8 rounded-lg border border-slate-800 bg-slate-900/60 p-4">
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-medium text-slate-100">
            <Palette className="h-4 w-4 text-cyan-300" />
            Brand voice
          </h2>
          {selectedProfile && (
            <p className="mt-1 text-xs text-slate-500">
              {formatBrandVoiceProfileSummary(selectedProfile)}
            </p>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={startCreate}
            disabled={mutating}
            className="flex items-center justify-center gap-2 rounded-md border border-cyan-500/60 px-2.5 py-1 text-xs text-cyan-100 hover:bg-cyan-500/10 disabled:opacity-50"
          >
            <Plus className="h-3.5 w-3.5" />
            New
          </button>
          <button
            type="button"
            onClick={onRefresh}
            disabled={loading || refreshing || mutating}
            className="flex items-center justify-center gap-2 rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-800 disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-3.5 w-3.5', refreshing && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </div>

      {error && !loading && (
        <div className="mb-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
          Brand voice profiles unavailable: {error.message}
        </div>
      )}
      {missingSelectedProfile && (
        <div className="mb-3 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-100">
          Selected brand voice profile was not found for this tenant.
        </div>
      )}

      <label className="block text-sm">
        <span className="text-slate-300">Saved profile</span>
        <select
          value={selectedProfileId ?? ''}
          disabled={loading || mutating}
          onChange={(event) => onChange(event.target.value || null)}
          className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-hidden disabled:opacity-50"
        >
          <option value="">
            {loading ? 'Loading profiles...' : 'No saved brand voice'}
          </option>
          {selectedProfileUnavailable && (
            <option value={selectedProfileId ?? ''}>
              Selected profile unavailable
            </option>
          )}
          {profiles.map((profile) => (
            <option key={profile.id} value={profile.id}>
              {profile.name}
            </option>
          ))}
        </select>
      </label>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={startEdit}
          disabled={!selectedProfile || loading || mutating}
          className="flex items-center justify-center gap-2 rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-800 disabled:opacity-50"
        >
          <Pencil className="h-3.5 w-3.5" />
          Edit selected
        </button>
        <button
          type="button"
          onClick={handleArchive}
          disabled={!selectedProfile || loading || mutating}
          className="flex items-center justify-center gap-2 rounded-md border border-rose-500/50 px-2.5 py-1 text-xs text-rose-100 hover:bg-rose-500/10 disabled:opacity-50"
        >
          <Trash2 className="h-3.5 w-3.5" />
          Archive
        </button>
      </div>

      {mutationState.kind === 'success' && (
        <div className="mt-3 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-100">
          {mutationState.message}
        </div>
      )}
      {mutationState.kind === 'error' && (
        <div className="mt-3 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-100">
          {mutationState.message}
        </div>
      )}

      {editor && (
        <form onSubmit={handleSave} className="mt-4 space-y-3 border-t border-slate-800 pt-4">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-sm font-medium text-slate-100">
              {editor.mode === 'edit' ? 'Edit brand voice' : 'New brand voice'}
            </h3>
            <button
              type="button"
              onClick={() => {
                resetSampleImport()
                setEditor(null)
              }}
              disabled={mutating}
              className="flex items-center justify-center gap-1.5 rounded-md border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800 disabled:opacity-50"
            >
              <X className="h-3.5 w-3.5" />
              Cancel
            </button>
          </div>

          <div className="rounded-md border border-slate-800 bg-slate-950/40 p-3">
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,1fr)_auto]">
              <label className="block text-sm">
                <span className="text-slate-300">Preset</span>
                <select
                  value={selectedPresetId}
                  onChange={(event) => {
                    setSelectedPresetId(event.target.value)
                    setPresetApplyState({ kind: 'idle' })
                  }}
                  disabled={mutating}
                  className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-hidden disabled:opacity-50"
                >
                  {BRAND_VOICE_PROFILE_PRESETS.map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {preset.label}
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                onClick={handleApplyPreset}
                disabled={!selectedPresetId || mutating}
                className="flex h-9 items-center justify-center gap-2 self-end rounded-md border border-cyan-500/60 px-2.5 text-xs text-cyan-100 hover:bg-cyan-500/10 disabled:opacity-50"
              >
                <Palette className="h-3.5 w-3.5" />
                Apply preset
              </button>
            </div>
            {presetApplyState.kind !== 'idle' && (
              <div
                className={clsx(
                  'mt-2 text-xs',
                  presetApplyState.kind === 'error'
                    ? 'text-rose-200'
                    : 'text-slate-400',
                )}
              >
                {presetApplyState.message}
              </div>
            )}
          </div>

          <div className="rounded-md border border-slate-800 bg-slate-950/40 p-3">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <h4 className="text-xs font-medium uppercase tracking-wide text-slate-400">
                Sample import
              </h4>
              <label className="flex cursor-pointer items-center justify-center gap-2 rounded-md border border-slate-700 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-800">
                <FileUp className="h-3.5 w-3.5" />
                Load file
                <input
                  type="file"
                  accept=".txt,.md,text/plain,text/markdown"
                  disabled={mutating || sampleFetching}
                  onChange={handleSampleFileChange}
                  className="sr-only"
                />
              </label>
            </div>
            <div className="mb-2 grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,1fr)_auto]">
              <label className="block text-sm">
                <span className="text-slate-300">URL</span>
                <input
                  type="url"
                  value={sampleUrl}
                  onChange={(event) => {
                    setSampleUrl(event.target.value)
                    setSampleImportState({ kind: 'idle' })
                  }}
                  disabled={mutating || sampleFetching}
                  placeholder="https://example.com/about"
                  className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-hidden disabled:opacity-50"
                />
              </label>
              <button
                type="button"
                onClick={handleFetchSampleUrl}
                disabled={!sampleUrl.trim() || mutating || sampleFetching}
                className="flex h-9 items-center justify-center gap-2 self-end rounded-md border border-slate-700 px-2.5 text-xs text-slate-300 hover:bg-slate-800 disabled:opacity-50"
              >
                {sampleFetching ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Search className="h-3.5 w-3.5" />
                )}
                Fetch URL
              </button>
            </div>
            <textarea
              value={sampleText}
              onChange={(event) => {
                setSampleText(event.target.value)
                setSampleName('')
                setSampleImportState({ kind: 'idle' })
              }}
              disabled={mutating || sampleFetching}
              rows={4}
              placeholder="Paste customer copy samples here."
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-200 focus:border-cyan-500 focus:outline-hidden disabled:opacity-50"
            />
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
              {sampleImportState.kind !== 'idle' ? (
                <span
                  className={clsx(
                    'text-xs',
                    sampleImportState.kind === 'error'
                      ? 'text-rose-200'
                      : 'text-slate-400',
                  )}
                >
                  {sampleImportState.message}
                </span>
              ) : (
                <span className="text-xs text-slate-500">
                  Fields already filled stay unchanged.
                </span>
              )}
              <button
                type="button"
                onClick={handleApplySample}
                disabled={!sampleText.trim() || mutating || sampleFetching}
                className="flex items-center justify-center gap-2 rounded-md border border-cyan-500/60 px-2.5 py-1 text-xs text-cyan-100 hover:bg-cyan-500/10 disabled:opacity-50"
              >
                <Palette className="h-3.5 w-3.5" />
                Apply sample
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            <label className="block text-sm">
              <span className="text-slate-300">Name</span>
              <input
                value={editor.name}
                onChange={(event) => updateEditor('name', event.target.value)}
                disabled={mutating}
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-hidden disabled:opacity-50"
              />
            </label>
            <label className="block text-sm">
              <span className="text-slate-300">Preferred POV</span>
              <input
                value={editor.preferredPov}
                onChange={(event) => updateEditor('preferredPov', event.target.value)}
                disabled={mutating}
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-hidden disabled:opacity-50"
              />
            </label>
            <label className="block text-sm">
              <span className="text-slate-300">Reading level</span>
              <input
                value={editor.readingLevel}
                onChange={(event) => updateEditor('readingLevel', event.target.value)}
                disabled={mutating}
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-hidden disabled:opacity-50"
              />
            </label>
            <label className="block text-sm lg:col-span-2">
              <span className="text-slate-300">Descriptors</span>
              <textarea
                value={editor.descriptorsText}
                onChange={(event) => updateEditor('descriptorsText', event.target.value)}
                disabled={mutating}
                rows={3}
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-hidden disabled:opacity-50"
              />
            </label>
            <label className="block text-sm">
              <span className="text-slate-300">Exemplars</span>
              <textarea
                value={editor.exemplarsText}
                onChange={(event) => updateEditor('exemplarsText', event.target.value)}
                disabled={mutating}
                rows={4}
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-hidden disabled:opacity-50"
              />
            </label>
            <label className="block text-sm">
              <span className="text-slate-300">Banned terms</span>
              <textarea
                value={editor.bannedTermsText}
                onChange={(event) => updateEditor('bannedTermsText', event.target.value)}
                disabled={mutating}
                rows={4}
                className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 focus:border-cyan-500 focus:outline-hidden disabled:opacity-50"
              />
            </label>
          </div>

          {!canSave && (
            <p className="text-xs text-slate-500">
              Name and at least one guidance field are required.
            </p>
          )}

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={!canSave || mutating || sampleFetching}
              className="flex items-center justify-center gap-2 rounded-md bg-cyan-500 px-3 py-1.5 text-sm font-medium text-slate-950 hover:bg-cyan-400 disabled:opacity-50"
            >
              <Save className="h-4 w-4" />
              {mutationState.kind === 'saving' ? 'Saving' : 'Save profile'}
            </button>
          </div>
        </form>
      )}
    </section>
  )
}

function formatBrandVoiceProfileSummary(
  profile: ContentOpsBrandVoiceProfile,
): string {
  const parts: string[] = []
  if (profile.descriptors.length > 0) {
    parts.push(profile.descriptors.slice(0, 3).join(', '))
  }
  if (profile.preferred_pov) {
    parts.push(profile.preferred_pov.replaceAll('_', ' '))
  }
  if (profile.reading_level) {
    parts.push(profile.reading_level)
  }
  return parts.length > 0 ? parts.join(' - ') : 'Saved profile selected'
}

function brandVoiceSampleFallbackName(url: string): string {
  try {
    const parsed = new URL(url)
    return parsed.hostname.replace(/^www\./, '') || 'sample URL'
  } catch {
    return 'sample URL'
  }
}
