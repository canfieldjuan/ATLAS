import { useEffect, useState, useCallback } from 'react';
import { Save, RotateCcw, Loader } from 'lucide-react';
import clsx from 'clsx';

// ---------- types ----------

export interface IntelligenceSettings {
  enabled: boolean;
  topics: string;
  regions: string;
  languages: string;
  lookback_days: number;
  pressure_velocity_threshold: number;
  signal_min_articles: number;
  max_articles_per_topic: number;
  llm_model: string;
  schedule_hour: number;
  notify_on_signal: boolean;
  notify_all_runs: boolean;
  include_in_morning_briefing: boolean;
}

// ---------- helpers ----------

const API_BASE = '/api/v1/settings';

async function fetchIntelligenceSettings(): Promise<IntelligenceSettings> {
  const res = await fetch(`${API_BASE}/intelligence`);
  if (!res.ok) throw new Error(`Failed to load intelligence settings: ${res.status} ${res.statusText}`);
  return res.json();
}

async function saveIntelligenceSettings(patch: Partial<IntelligenceSettings>): Promise<IntelligenceSettings> {
  const res = await fetch(`${API_BASE}/intelligence`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error(`Failed to save intelligence settings: ${res.status} ${res.statusText}`);
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
        <span className={clsx(
          'absolute top-0.5 w-4 h-4 rounded-full transition-all duration-200',
          checked ? 'left-5 bg-cyan-400' : 'left-0.5 bg-cyan-700',
        )} />
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
        className="w-full mt-1 bg-black/30 border border-cyan-500/30 rounded px-2 py-1.5 text-sm text-cyan-200 placeholder-cyan-700/50 outline-none focus:border-cyan-500/60 transition-all"
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
          className="w-28 bg-black/30 border border-cyan-500/30 rounded px-2 py-1.5 text-sm text-cyan-200 outline-none focus:border-cyan-500/60 tabular-nums transition-all"
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
  const fmt = format ?? String;
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
        step={step ?? 0.1}
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

function InfoBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-cyan-950/30 border border-cyan-500/20 rounded px-3 py-2 text-[11px] text-cyan-500/70 my-2">
      {children}
    </div>
  );
}

function CredentialNote({ envVar }: { envVar: string }) {
  return (
    <div className="flex items-start gap-2 bg-amber-950/20 border border-amber-500/20 rounded px-3 py-2 text-[11px] text-amber-500/70 my-2">
      <span className="shrink-0 mt-0.5">⚠</span>
      <span>
        API key is a credential — set <code className="text-amber-400">{envVar}</code> in your{' '}
        <code className="text-amber-400">.env</code> file. It will not appear here.
      </span>
    </div>
  );
}

/** Shows how "pressure" concept works as a visual explainer */
function PressureExplainer() {
  return (
    <div className="bg-black/20 border border-cyan-500/15 rounded px-3 py-3 my-3">
      <div className="text-[10px] uppercase tracking-widest text-cyan-500/50 mb-2">How it works</div>
      <div className="flex items-stretch gap-0 text-[10px] mb-1">
        {/* Timeline */}
        <div className="flex flex-col items-center w-5 shrink-0">
          <div className="w-1.5 h-1.5 rounded-full bg-cyan-700 mt-1" />
          <div className="flex-1 w-px bg-gradient-to-b from-cyan-700 via-cyan-500 to-cyan-400" />
          <div className="w-2 h-2 rounded-full bg-cyan-400 mb-1" />
        </div>
        <div className="flex flex-col justify-between ml-2 gap-3">
          <div>
            <span className="text-cyan-600 font-bold">PRESSURE BUILDS</span>
            <span className="text-cyan-700"> — topic mentions accelerate in niche &amp; regional outlets.
            Atlas detects the velocity spike and flags it as a leading indicator.</span>
          </div>
          <div>
            <span className="text-cyan-400 font-bold">NEWS BREAKS</span>
            <span className="text-cyan-600"> — event reaches mainstream headlines, often days later.
            By then Atlas has already surfaced the pattern.</span>
          </div>
        </div>
      </div>
      <div className="text-[10px] text-cyan-700 mt-2">
        Velocity = <em>today's article count</em> ÷ <em>daily average over lookback window</em>.
        A velocity of <strong className="text-cyan-500">1.5×</strong> means 50% more articles than usual — a meaningful acceleration.
      </div>
    </div>
  );
}

// ---------- main form ----------

interface IntelligenceSettingsFormProps {
  onDirtyChange?: (dirty: boolean) => void;
}

export function IntelligenceSettingsForm({ onDirtyChange }: IntelligenceSettingsFormProps) {
  const [original, setOriginal] = useState<IntelligenceSettings | null>(null);
  const [draft, setDraft] = useState<IntelligenceSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchIntelligenceSettings();
      setOriginal(data);
      setDraft(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load intelligence settings');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const set = <K extends keyof IntelligenceSettings>(key: K, value: IntelligenceSettings[K]) => {
    setDraft((d) => d ? { ...d, [key]: value } : d);
    setSaved(false);
  };

  const isDirty = draft && original && JSON.stringify(draft) !== JSON.stringify(original);

  useEffect(() => {
    onDirtyChange?.(!!isDirty);
  }, [isDirty, onDirtyChange]);

  const handleSave = async () => {
    if (!draft || !original) return;
    setSaving(true);
    setError(null);
    try {
      const patch: Partial<IntelligenceSettings> = {};
      (Object.keys(draft) as (keyof IntelligenceSettings)[]).forEach(<K extends keyof IntelligenceSettings>(k: K) => {
        if (draft[k] !== original[k]) patch[k] = draft[k];
      });
      if (Object.keys(patch).length === 0) { setSaved(true); return; }
      const updated = await saveIntelligenceSettings(patch);
      setOriginal(updated);
      setDraft(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save intelligence settings');
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    if (original) { setDraft(original); setSaved(false); }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-40 gap-2 text-cyan-600">
        <Loader size={16} className="animate-spin" />
        <span className="text-sm">Loading intelligence settings…</span>
      </div>
    );
  }

  if (!draft) {
    return (
      <div className="bg-red-900/20 border border-red-500/30 rounded px-3 py-2 text-sm text-red-400">
        {error ?? 'Could not load intelligence settings.'}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* scrollable body */}
      <div className="flex-1 overflow-y-auto px-5 py-4 custom-scrollbar">
        {error && (
          <div className="bg-red-900/20 border border-red-500/30 rounded px-3 py-2 text-sm text-red-400 mb-4">
            {error}
          </div>
        )}

        {/* ── General ── */}
        <SectionHeader>General</SectionHeader>
        <PressureExplainer />
        <Toggle
          label="Enable news intelligence"
          description="Run the daily pressure signal analysis. Requires a NewsAPI.org API key."
          checked={draft.enabled}
          onChange={(v) => set('enabled', v)}
        />
        <CredentialNote envVar="ATLAS_NEWS_API_KEY" />

        {/* ── What to Monitor ── */}
        <SectionHeader>What to Monitor</SectionHeader>
        <InfoBox>
          Topics are the seeds for each analysis. Atlas fetches recent news for each topic,
          measures how fast it's growing, and flags any that are accelerating unusually.
          Separate multiple topics with commas.
        </InfoBox>
        <TextInput
          label="Topics"
          description="Comma-separated — each becomes an independent pressure signal feed"
          value={draft.topics}
          onChange={(v) => set('topics', v)}
          placeholder="supply chain,interest rates,AI regulation,energy prices"
        />
        <TextInput
          label="Geographic focus"
          description="Comma-separated regions or country names prepended to queries to bias toward local sources"
          value={draft.regions}
          onChange={(v) => set('regions', v)}
          placeholder="US,Europe"
        />
        <TextInput
          label="Article languages"
          description="Comma-separated ISO language codes — limits which articles NewsAPI returns"
          value={draft.languages}
          onChange={(v) => set('languages', v)}
          placeholder="en"
        />

        {/* ── Pressure Signal Detection ── */}
        <SectionHeader>Pressure Signal Detection</SectionHeader>
        <InfoBox>
          A <strong className="text-cyan-400">pressure signal</strong> is flagged when a topic's
          article count today is significantly higher than its recent baseline — indicating the
          topic is accelerating in the media before it becomes mainstream news.
        </InfoBox>
        <NumberInput
          label="Baseline window"
          description="Days of history used to calculate each topic's normal article rate — longer = more stable baseline"
          value={draft.lookback_days}
          onChange={(v) => set('lookback_days', v)}
          min={2}
          max={30}
          unit="days"
        />
        <SliderInput
          label="Velocity threshold"
          description="Minimum growth multiplier to flag a pressure signal — 1.5× means 50% more articles than the baseline daily average"
          value={draft.pressure_velocity_threshold}
          onChange={(v) => set('pressure_velocity_threshold', v)}
          min={1.0}
          max={5.0}
          step={0.1}
          format={(v) => `${v.toFixed(1)}×`}
        />
        <NumberInput
          label="Minimum confirming articles"
          description="A topic needs at least this many articles today before it's flagged — prevents single-source noise from triggering false alerts"
          value={draft.signal_min_articles}
          onChange={(v) => set('signal_min_articles', v)}
          min={1}
          max={20}
          unit="articles"
        />

        {/* ── Operations ── */}
        <SectionHeader>Operations</SectionHeader>
        <NumberInput
          label="Max articles per topic"
          description="Caps NewsAPI calls per topic per run — each topic uses one API request"
          value={draft.max_articles_per_topic}
          onChange={(v) => set('max_articles_per_topic', v)}
          min={5}
          max={100}
          unit="articles"
        />
        <TextInput
          label="LLM model"
          description="Ollama model used to write the intelligence briefing summary"
          value={draft.llm_model}
          onChange={(v) => set('llm_model', v)}
          placeholder="qwen3:14b"
        />
        <NumberInput
          label="Run hour"
          description="Hour of day (0–23) to run the analysis — runs before the 7 AM morning briefing by default"
          value={draft.schedule_hour}
          onChange={(v) => set('schedule_hour', Math.min(23, Math.max(0, v)))}
          min={0}
          max={23}
          unit="h (24-hr)"
        />

        {/* ── Notifications & Output ── */}
        <SectionHeader>Notifications & Output</SectionHeader>
        <Toggle
          label="Notify on pressure signals"
          description="Send a push notification (via ntfy) when at least one new pressure signal is detected"
          checked={draft.notify_on_signal}
          onChange={(v) => set('notify_on_signal', v)}
        />
        <Toggle
          label="Notify on every run"
          description="Send a push notification even when no new signals are detected — useful for confirming the task ran"
          checked={draft.notify_all_runs}
          onChange={(v) => set('notify_all_runs', v)}
        />
        <Toggle
          label="Include in morning briefing"
          description="Surface active pressure signals in the 7 AM morning briefing summary"
          checked={draft.include_in_morning_briefing}
          onChange={(v) => set('include_in_morning_briefing', v)}
        />
      </div>

      {/* footer */}
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
          {saved && <span className="text-[11px] text-cyan-400 animate-in fade-in duration-300">✓ Saved</span>}
          {error && <span className="text-[11px] text-red-400">{error}</span>}
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
    </div>
  );
}
