import { useEffect, useState, useCallback } from 'react';
import { X, Save, RotateCcw, Loader } from 'lucide-react';
import clsx from 'clsx';

// ---------- types ----------

interface VoiceSettings {
  // Pipeline
  enabled: boolean;
  streaming_llm_enabled: boolean;
  debug_logging: boolean;
  // Microphone
  input_device: string | null;
  audio_gain: number;
  use_arecord: boolean;
  arecord_device: string;
  // Wake word
  wake_threshold: number;
  wake_confirmation_enabled: boolean;
  // ASR
  asr_url: string | null;
  asr_timeout: number;
  asr_streaming_enabled: boolean;
  asr_ws_url: string | null;
  // TTS
  piper_length_scale: number;
  // VAD
  vad_aggressiveness: number;
  silence_ms: number;
  // Conversation mode
  conversation_mode_enabled: boolean;
  conversation_timeout_ms: number;
  // Filler
  filler_enabled: boolean;
  filler_delay_ms: number;
  // Timeouts
  agent_timeout: number;
}

// ---------- helpers ----------

const API_BASE = '/api/v1/settings';

async function fetchVoiceSettings(): Promise<VoiceSettings> {
  const res = await fetch(`${API_BASE}/voice`);
  if (!res.ok) throw new Error(`Failed to load settings: ${res.status} ${res.statusText}`);
  return res.json();
}

async function saveVoiceSettings(patch: Partial<VoiceSettings>): Promise<VoiceSettings> {
  const res = await fetch(`${API_BASE}/voice`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error(`Failed to save settings: ${res.status} ${res.statusText}`);
  return res.json();
}

// ---------- sub-components ----------

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 mb-3 mt-5 first:mt-0">
      <div className="h-px flex-1 bg-cyan-500/20" />
      <span className="text-[10px] uppercase tracking-widest text-cyan-500/60 font-bold">{children}</span>
      <div className="h-px flex-1 bg-cyan-500/20" />
    </div>
  );
}

function Toggle({
  label, description, checked, onChange,
}: {
  label: string; description?: string; checked: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-2">
      <div className="min-w-0">
        <div className="text-sm text-cyan-200">{label}</div>
        {description && <div className="text-[11px] text-cyan-600 mt-0.5">{description}</div>}
      </div>
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className={clsx(
          'relative shrink-0 w-10 h-5 rounded-full border transition-all duration-200',
          checked
            ? 'bg-cyan-500/30 border-cyan-500 shadow-[0_0_8px_rgba(34,211,238,0.4)]'
            : 'bg-black/30 border-cyan-500/30',
        )}
        aria-checked={checked}
        role="switch"
      >
        <span
          className={clsx(
            'absolute top-0.5 w-4 h-4 rounded-full transition-all duration-200',
            checked ? 'left-5 bg-cyan-400' : 'left-0.5 bg-cyan-700',
          )}
        />
      </button>
    </div>
  );
}

function TextInput({
  label, description, value, onChange, placeholder,
}: {
  label: string; description?: string; value: string; onChange: (v: string) => void; placeholder?: string;
}) {
  return (
    <div className="py-2">
      <label className="text-sm text-cyan-200 block">{label}</label>
      {description && <div className="text-[11px] text-cyan-600 mb-1">{description}</div>}
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full mt-1 bg-black/30 border border-cyan-500/30 rounded px-2 py-1.5 text-sm text-cyan-200 placeholder-cyan-700/50 outline-none focus:border-cyan-500/60 focus:bg-black/40 transition-all"
      />
    </div>
  );
}

function NumberInput({
  label, description, value, onChange, min, max, step, unit,
}: {
  label: string; description?: string; value: number; onChange: (v: number) => void;
  min?: number; max?: number; step?: number; unit?: string;
}) {
  return (
    <div className="py-2">
      <label className="text-sm text-cyan-200 block">{label}</label>
      {description && <div className="text-[11px] text-cyan-600 mb-1">{description}</div>}
      <div className="flex items-center gap-2 mt-1">
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          step={step ?? 1}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-28 bg-black/30 border border-cyan-500/30 rounded px-2 py-1.5 text-sm text-cyan-200 outline-none focus:border-cyan-500/60 focus:bg-black/40 transition-all tabular-nums"
        />
        {unit && <span className="text-[11px] text-cyan-600">{unit}</span>}
      </div>
    </div>
  );
}

function SliderInput({
  label, description, value, onChange, min, max, step, format,
}: {
  label: string; description?: string; value: number; onChange: (v: number) => void;
  min: number; max: number; step?: number; format?: (v: number) => string;
}) {
  const fmt = format ?? ((v) => v.toString());
  return (
    <div className="py-2">
      <div className="flex justify-between items-center">
        <label className="text-sm text-cyan-200">{label}</label>
        <span className="text-sm text-cyan-400 font-mono tabular-nums">{fmt(value)}</span>
      </div>
      {description && <div className="text-[11px] text-cyan-600 mt-0.5 mb-1">{description}</div>}
      <input
        type="range"
        min={min}
        max={max}
        step={step ?? 0.01}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full mt-1 accent-cyan-400 cursor-pointer"
      />
      <div className="flex justify-between text-[10px] text-cyan-700 mt-0.5">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

function SelectInput({
  label, description, value, onChange, options,
}: {
  label: string; description?: string; value: number; onChange: (v: number) => void;
  options: { value: number; label: string }[];
}) {
  return (
    <div className="py-2">
      <label className="text-sm text-cyan-200 block">{label}</label>
      {description && <div className="text-[11px] text-cyan-600 mb-1">{description}</div>}
      <select
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 bg-black/30 border border-cyan-500/30 rounded px-2 py-1.5 text-sm text-cyan-200 outline-none focus:border-cyan-500/60 transition-all"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value} className="bg-gray-900">
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

// ---------- main form ----------

interface VoiceSettingsFormProps {
  onDirtyChange?: (dirty: boolean) => void;
}

export function VoiceSettingsForm({ onDirtyChange }: VoiceSettingsFormProps) {
  const [original, setOriginal] = useState<VoiceSettings | null>(null);
  const [draft, setDraft] = useState<VoiceSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchVoiceSettings();
      setOriginal(data);
      setDraft(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load settings');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const set = <K extends keyof VoiceSettings>(key: K, value: VoiceSettings[K]) => {
    setDraft((d) => d ? { ...d, [key]: value } : d);
    setSaved(false);
  };

  const handleSave = async () => {
    if (!draft || !original) return;
    setSaving(true);
    setError(null);
    try {
      // Only send changed fields
      const patch: Partial<VoiceSettings> = {};
      (Object.keys(draft) as (keyof VoiceSettings)[]).forEach((k) => {
        if (draft[k] !== original[k]) (patch as Partial<VoiceSettings>)[k] = draft[k] as never;
      });
      if (Object.keys(patch).length === 0) {
        setSaved(true);
        return;
      }
      const updated = await saveVoiceSettings(patch);
      setOriginal(updated);
      setDraft(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    if (original) {
      setDraft(original);
      setSaved(false);
    }
  };

  const isDirty = draft && original && JSON.stringify(draft) !== JSON.stringify(original);

  // Notify parent of dirty state
  useEffect(() => {
    onDirtyChange?.(!!isDirty);
  }, [isDirty, onDirtyChange]);

  return (
    <div className="flex flex-col h-full">
      {/* body */}
      <div className="flex-1 overflow-y-auto px-5 py-4 custom-scrollbar">
        {loading && (
          <div className="flex items-center justify-center h-40 gap-2 text-cyan-600">
            <Loader size={16} className="animate-spin" />
            <span className="text-sm">Loading settings…</span>
          </div>
        )}

        {error && (
          <div className="bg-red-900/20 border border-red-500/30 rounded px-3 py-2 text-sm text-red-400 mb-4">
            {error}
          </div>
        )}

        {draft && !loading && (
          <>
            {/* ── Pipeline ── */}
            <SectionHeader>Pipeline</SectionHeader>
              <Toggle
                label="Voice pipeline enabled"
                description="Start the wake-word / voice pipeline when Atlas Brain starts"
                checked={draft.enabled}
                onChange={(v) => set('enabled', v)}
              />
              <Toggle
                label="Streaming LLM to TTS"
                description="Speak sentences as the LLM generates them (lower perceived latency)"
                checked={draft.streaming_llm_enabled}
                onChange={(v) => set('streaming_llm_enabled', v)}
              />
              <Toggle
                label="Debug logging"
                description="Verbose voice pipeline logs (useful for troubleshooting)"
                checked={draft.debug_logging}
                onChange={(v) => set('debug_logging', v)}
              />

              {/* ── Microphone ── */}
              <SectionHeader>Microphone</SectionHeader>
              <TextInput
                label="Input device"
                description="PortAudio device name or index — leave blank for system default"
                value={draft.input_device ?? ''}
                onChange={(v) => set('input_device', v || null)}
                placeholder="e.g. sysdefault:CARD=SoloCast or 13"
              />
              <SliderInput
                label="Microphone gain"
                description="Software amplification applied to the microphone signal"
                value={draft.audio_gain}
                onChange={(v) => set('audio_gain', v)}
                min={0.1}
                max={3.0}
                step={0.1}
                format={(v) => `×${v.toFixed(1)}`}
              />
              <Toggle
                label="Use ALSA (arecord)"
                description="Use arecord instead of PortAudio — required on some headless Linux setups"
                checked={draft.use_arecord}
                onChange={(v) => set('use_arecord', v)}
              />
              {draft.use_arecord && (
                <TextInput
                  label="ALSA device"
                  value={draft.arecord_device}
                  onChange={(v) => set('arecord_device', v)}
                  placeholder="default"
                />
              )}

              {/* ── Wake Word ── */}
              <SectionHeader>Wake Word</SectionHeader>
              <SliderInput
                label="Detection sensitivity"
                description="Higher values require a more confident wake-word match (reduces false positives)"
                value={draft.wake_threshold}
                onChange={(v) => set('wake_threshold', v)}
                min={0.05}
                max={0.95}
                step={0.05}
                format={(v) => v.toFixed(2)}
              />
              <Toggle
                label="Confirmation tone"
                description="Play a short beep when the wake word is detected"
                checked={draft.wake_confirmation_enabled}
                onChange={(v) => set('wake_confirmation_enabled', v)}
              />

              {/* ── Speech Recognition (ASR) ── */}
              <SectionHeader>Speech Recognition (ASR)</SectionHeader>
              <Toggle
                label="WebSocket streaming mode"
                description="Stream audio to ASR in real time instead of uploading a batch recording"
                checked={draft.asr_streaming_enabled}
                onChange={(v) => set('asr_streaming_enabled', v)}
              />
              {draft.asr_streaming_enabled ? (
                <TextInput
                  label="ASR WebSocket URL"
                  value={draft.asr_ws_url ?? ''}
                  onChange={(v) => set('asr_ws_url', v || null)}
                  placeholder="ws://localhost:8081/v1/asr/stream"
                />
              ) : (
                <>
                  <TextInput
                    label="ASR HTTP URL"
                    value={draft.asr_url ?? ''}
                    onChange={(v) => set('asr_url', v || null)}
                    placeholder="http://localhost:8081"
                  />
                  <NumberInput
                    label="Request timeout"
                    value={draft.asr_timeout}
                    onChange={(v) => set('asr_timeout', v)}
                    min={5}
                    max={120}
                    unit="seconds"
                  />
                </>
              )}

              {/* ── Text-to-Speech (TTS) ── */}
              <SectionHeader>Text-to-Speech (TTS)</SectionHeader>
              <SliderInput
                label="Speech rate"
                description="Lower values produce faster speech; higher values produce slower speech"
                value={draft.piper_length_scale}
                onChange={(v) => set('piper_length_scale', v)}
                min={0.5}
                max={2.0}
                step={0.05}
                format={(v) => `${v.toFixed(2)}×`}
              />

              {/* ── Voice Activity Detection ── */}
              <SectionHeader>Voice Activity Detection</SectionHeader>
              <SelectInput
                label="VAD aggressiveness"
                description="Higher values cut off speech faster but may clip short words"
                value={draft.vad_aggressiveness}
                onChange={(v) => set('vad_aggressiveness', v)}
                options={[
                  { value: 0, label: '0 — Gentle (misses some silence)' },
                  { value: 1, label: '1 — Low' },
                  { value: 2, label: '2 — Medium (default)' },
                  { value: 3, label: '3 — Aggressive (fastest cutoff)' },
                ]}
              />
              <NumberInput
                label="Silence timeout"
                description="How long Atlas waits after you stop speaking before finalising the utterance"
                value={draft.silence_ms}
                onChange={(v) => set('silence_ms', v)}
                min={200}
                max={3000}
                step={50}
                unit="ms"
              />

              {/* ── Conversation Mode ── */}
              <SectionHeader>Conversation Mode</SectionHeader>
              <Toggle
                label="Enabled"
                description="Stay in conversation mode after each response — no wake word needed for follow-ups"
                checked={draft.conversation_mode_enabled}
                onChange={(v) => set('conversation_mode_enabled', v)}
              />
              {draft.conversation_mode_enabled && (
                <NumberInput
                  label="Conversation timeout"
                  description="Milliseconds to wait for a follow-up before returning to wake-word mode"
                  value={draft.conversation_timeout_ms}
                  onChange={(v) => set('conversation_timeout_ms', v)}
                  min={1000}
                  max={60000}
                  step={500}
                  unit="ms"
                />
              )}

              {/* ── Filler Phrases ── */}
              <SectionHeader>Filler Phrases</SectionHeader>
              <Toggle
                label="Enabled"
                description="Speak a short filler phrase (e.g. 'Please hold.') while the agent is thinking"
                checked={draft.filler_enabled}
                onChange={(v) => set('filler_enabled', v)}
              />
              {draft.filler_enabled && (
                <NumberInput
                  label="Filler delay"
                  description="How long to wait before speaking the first filler phrase"
                  value={draft.filler_delay_ms}
                  onChange={(v) => set('filler_delay_ms', v)}
                  min={200}
                  max={5000}
                  step={100}
                  unit="ms"
                />
              )}

              {/* ── Timeouts ── */}
              <SectionHeader>Processing Timeouts</SectionHeader>
              <NumberInput
                label="Agent timeout"
                description="Maximum time to wait for the LLM and tools to produce a response"
                value={draft.agent_timeout}
                onChange={(v) => set('agent_timeout', v)}
                min={5}
                max={120}
                step={1}
                unit="seconds"
              />
            </>
          )}
        </div>

      {/* footer */}
      {draft && !loading && (
        <div className="flex items-center justify-between gap-3 px-5 py-3 border-t border-cyan-500/20 shrink-0">
          <button
            onClick={handleReset}
            disabled={!isDirty}
            className={clsx(
              'flex items-center gap-1.5 px-3 py-1.5 rounded border text-xs transition-all',
              isDirty
                ? 'border-cyan-500/40 text-cyan-500 hover:border-cyan-500 hover:bg-cyan-500/10'
                : 'border-cyan-500/15 text-cyan-700 cursor-not-allowed',
            )}
          >
            <RotateCcw size={12} />
            Revert
          </button>

          <div className="flex items-center gap-2">
            {saved && (
              <span className="text-[11px] text-cyan-400 animate-in fade-in duration-300">
                ✓ Saved
              </span>
            )}
            {error && (
              <span className="text-[11px] text-red-400">{error}</span>
            )}
            <button
              onClick={handleSave}
              disabled={saving || !isDirty}
              className={clsx(
                'flex items-center gap-1.5 px-4 py-1.5 rounded border text-xs font-bold uppercase tracking-wider transition-all',
                isDirty && !saving
                  ? 'border-cyan-500 text-cyan-400 bg-cyan-500/10 hover:bg-cyan-500/20 shadow-[0_0_12px_rgba(34,211,238,0.2)]'
                  : 'border-cyan-500/15 text-cyan-700 cursor-not-allowed',
              )}
            >
              {saving ? <Loader size={12} className="animate-spin" /> : <Save size={12} />}
              {saving ? 'Saving…' : 'Save'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// Legacy standalone modal wrapper (kept for backward compatibility)
interface VoiceSettingsModalProps {
  onClose: () => void;
}

export function VoiceSettingsModal({ onClose }: VoiceSettingsModalProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="relative w-full max-w-lg max-h-[90vh] flex flex-col bg-[#020617]/95 border border-cyan-500/30 rounded-sm shadow-[0_0_40px_rgba(34,211,238,0.1)] overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b border-cyan-500/20 shrink-0">
          <div>
            <h2 className="text-sm font-bold uppercase tracking-widest text-cyan-300">Voice Pipeline Settings</h2>
            <p className="text-[10px] text-cyan-600 mt-0.5">Changes apply immediately · restart required for hardware settings</p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded border border-cyan-500/20 hover:border-cyan-500/60 hover:bg-cyan-500/10 transition-all">
            <X size={14} className="text-cyan-500" />
          </button>
        </div>
        <div className="flex-1 min-h-0 flex flex-col">
          <VoiceSettingsForm />
        </div>
        <style>{`
          .custom-scrollbar::-webkit-scrollbar { width: 3px; }
          .custom-scrollbar::-webkit-scrollbar-track { background: rgba(8,145,178,0.05); }
          .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(34,211,238,0.3); border-radius: 10px; }
          input[type=range] { height: 4px; }
          select option { background: #020617; }
        `}</style>
      </div>
    </div>
  );
}
