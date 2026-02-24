import { useEffect, useState, useCallback } from 'react';
import { Save, RotateCcw, Loader } from 'lucide-react';
import clsx from 'clsx';

// ---------- types ----------

export interface DailySettings {
  // Persona
  persona_name: string;
  persona_owner_name: string;
  persona_system_prompt: string;
  // LLM
  llm_temperature: number;
  llm_max_tokens: number;
  llm_max_history: number;
  // Autonomous
  autonomous_enabled: boolean;
  autonomous_timezone: string;
  notify_results: boolean;
  announce_results: boolean;
  synthesis_enabled: boolean;
  synthesis_temperature: number;
  // Briefing
  briefing_calendar_hours: number;
  briefing_security_hours: number;
  // Nightly sync
  nightly_sync_enabled: boolean;
  nightly_sync_max_turns: number;
  memory_purge_days: number;
  // Learning
  pattern_learning_lookback_days: number;
  preference_learning_lookback_days: number;
  preference_learning_min_turns: number;
  // Proactive
  proactive_lookback_hours: number;
  action_escalation_stale_days: number;
  action_escalation_overdue_days: number;
  // Device & security
  device_health_battery_threshold: number;
  device_health_stale_hours: number;
  security_summary_hours: number;
}

// ---------- helpers ----------

const API_BASE = '/api/v1/settings';

async function fetchDailySettings(): Promise<DailySettings> {
  const res = await fetch(`${API_BASE}/daily`);
  if (!res.ok) throw new Error(`Failed to load daily settings: ${res.status} ${res.statusText}`);
  return res.json();
}

async function saveDailySettings(patch: Partial<DailySettings>): Promise<DailySettings> {
  const res = await fetch(`${API_BASE}/daily`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error(`Failed to save daily settings: ${res.status} ${res.statusText}`);
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

function TextArea({
  label, description, value, onChange, rows = 5,
}: {
  label: string; description?: string; value: string; onChange: (v: string) => void; rows?: number;
}) {
  return (
    <div className="py-2">
      <label className="text-sm text-cyan-200 block">{label}</label>
      {description && <div className="text-[11px] text-cyan-600 mb-1">{description}</div>}
      <textarea
        value={value}
        rows={rows}
        onChange={(e) => onChange(e.target.value)}
        className="w-full mt-1 bg-black/30 border border-cyan-500/30 rounded px-2 py-1.5 text-sm text-cyan-200 outline-none focus:border-cyan-500/60 resize-none transition-all leading-relaxed"
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

function InfoBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-cyan-950/30 border border-cyan-500/20 rounded px-3 py-2 text-[11px] text-cyan-500/70 my-2">
      {children}
    </div>
  );
}

// ---------- schedule badge ----------

function ScheduleBadge({ time, label }: { time: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1 text-[10px] text-cyan-600 border border-cyan-500/20 rounded px-1.5 py-0.5 ml-2">
      <span className="text-cyan-500/60">⏱</span> {time} · {label}
    </span>
  );
}

// ---------- main form ----------

interface DailySettingsFormProps {
  onDirtyChange?: (dirty: boolean) => void;
}

export function DailySettingsForm({ onDirtyChange }: DailySettingsFormProps) {
  const [original, setOriginal] = useState<DailySettings | null>(null);
  const [draft, setDraft] = useState<DailySettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchDailySettings();
      setOriginal(data);
      setDraft(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load daily settings');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const set = <K extends keyof DailySettings>(key: K, value: DailySettings[K]) => {
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
      const patch: Partial<DailySettings> = {};
      (Object.keys(draft) as (keyof DailySettings)[]).forEach(<K extends keyof DailySettings>(k: K) => {
        if (draft[k] !== original[k]) patch[k] = draft[k];
      });
      if (Object.keys(patch).length === 0) { setSaved(true); return; }
      const updated = await saveDailySettings(patch);
      setOriginal(updated);
      setDraft(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save daily settings');
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
        <span className="text-sm">Loading daily settings…</span>
      </div>
    );
  }

  if (!draft) {
    return (
      <div className="bg-red-900/20 border border-red-500/30 rounded px-3 py-2 text-sm text-red-400">
        {error ?? 'Could not load daily settings.'}
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

        {/* ── Atlas Identity ── */}
        <SectionHeader>Atlas Identity</SectionHeader>
        <InfoBox>
          These settings define who Atlas is and who it works for. Changes take effect on the next conversation.
        </InfoBox>
        <TextInput
          label="Assistant name"
          description="How Atlas refers to itself"
          value={draft.persona_name}
          onChange={(v) => set('persona_name', v)}
          placeholder="Atlas"
        />
        <TextInput
          label="Your name"
          description="Used in emails, sign-offs, reminders, and personalised responses"
          value={draft.persona_owner_name}
          onChange={(v) => set('persona_owner_name', v)}
          placeholder="Juan"
        />
        <TextArea
          label="Personality & instructions"
          description="Core system prompt — defines Atlas's tone, priorities, and behaviour for every conversation"
          value={draft.persona_system_prompt}
          onChange={(v) => set('persona_system_prompt', v)}
          rows={6}
        />

        {/* ── LLM Response Behaviour ── */}
        <SectionHeader>LLM Response Behaviour</SectionHeader>
        <SliderInput
          label="Conversational temperature"
          description="Higher = more creative and varied responses; lower = more consistent and precise"
          value={draft.llm_temperature}
          onChange={(v) => set('llm_temperature', v)}
          min={0.0}
          max={2.0}
          step={0.05}
          format={(v) => v.toFixed(2)}
        />
        <NumberInput
          label="Max response tokens"
          description="Maximum tokens per conversational reply (longer = more detailed but slower)"
          value={draft.llm_max_tokens}
          onChange={(v) => set('llm_max_tokens', v)}
          min={50}
          max={1024}
          unit="tokens"
        />
        <NumberInput
          label="Conversation memory depth"
          description="Previous exchanges passed as context — higher gives better continuity but uses more tokens"
          value={draft.llm_max_history}
          onChange={(v) => set('llm_max_history', v)}
          min={0}
          max={20}
          unit="turns"
        />

        {/* ── Autonomous Scheduler ── */}
        <SectionHeader>Autonomous Scheduler</SectionHeader>
        <InfoBox>
          The scheduler runs all daily and nightly tasks. When disabled, no background reasoning,
          briefings, or learning tasks will run.
        </InfoBox>
        <Toggle
          label="Enable autonomous scheduler"
          description="Master switch for all daily briefings, nightly learning, and periodic intelligence tasks"
          checked={draft.autonomous_enabled}
          onChange={(v) => set('autonomous_enabled', v)}
        />
        {draft.autonomous_enabled && (
          <>
            <TextInput
              label="Timezone"
              description="Timezone used for scheduling all tasks (e.g. America/Chicago, America/New_York)"
              value={draft.autonomous_timezone}
              onChange={(v) => set('autonomous_timezone', v)}
              placeholder="America/Chicago"
            />
            <Toggle
              label="Push notifications for task results"
              description="Send a push notification (via ntfy) when a nightly task produces a synthesised result"
              checked={draft.notify_results}
              onChange={(v) => set('notify_results', v)}
            />
            <Toggle
              label="Speak task results aloud"
              description="Broadcast synthesised nightly results via TTS on edge nodes"
              checked={draft.announce_results}
              onChange={(v) => set('announce_results', v)}
            />
            <Toggle
              label="LLM result synthesis"
              description="Use the LLM to turn raw task output into a plain-language summary"
              checked={draft.synthesis_enabled}
              onChange={(v) => set('synthesis_enabled', v)}
            />
            {draft.synthesis_enabled && (
              <SliderInput
                label="Synthesis temperature"
                description="Lower = more consistent summaries; higher = more varied phrasing"
                value={draft.synthesis_temperature}
                onChange={(v) => set('synthesis_temperature', v)}
                min={0.0}
                max={1.0}
                step={0.05}
                format={(v) => v.toFixed(2)}
              />
            )}
          </>
        )}

        {/* ── Daily Briefing ── */}
        <SectionHeader>
          Daily Briefing
          <ScheduleBadge time="7:00 AM" label="daily" />
        </SectionHeader>
        <InfoBox>
          Every morning Atlas assembles a briefing from your calendar, overnight security events,
          device health, pending actions, and email drafts awaiting approval.
        </InfoBox>
        <NumberInput
          label="Calendar look-ahead"
          description="How many hours ahead to pull events into the morning briefing"
          value={draft.briefing_calendar_hours}
          onChange={(v) => set('briefing_calendar_hours', v)}
          min={1}
          max={48}
          unit="hours"
        />
        <NumberInput
          label="Overnight security lookback"
          description="Hours to look back when summarising overnight security events"
          value={draft.briefing_security_hours}
          onChange={(v) => set('briefing_security_hours', v)}
          min={1}
          max={24}
          unit="hours"
        />

        {/* ── Nightly Memory Sync ── */}
        <SectionHeader>
          Nightly Memory Sync
          <ScheduleBadge time="3:00 AM" label="nightly" />
        </SectionHeader>
        <InfoBox>
          Each night Atlas extracts entities, facts, and relationships from today's conversations
          and stores them in the knowledge graph — enabling Atlas to remember context across sessions.
        </InfoBox>
        <Toggle
          label="Enable nightly memory sync"
          description="Sync today's conversations into the knowledge graph while you sleep"
          checked={draft.nightly_sync_enabled}
          onChange={(v) => set('nightly_sync_enabled', v)}
        />
        {draft.nightly_sync_enabled && (
          <>
            <NumberInput
              label="Max turns per sync run"
              description="Caps how many conversation turns are processed per night — any remainder carries over to the following night"
              value={draft.nightly_sync_max_turns}
              onChange={(v) => set('nightly_sync_max_turns', v)}
              min={10}
              max={1000}
              step={10}
              unit="turns"
            />
            <NumberInput
              label="Raw message retention"
              description="Delete raw conversation turns from the database after this many days — the knowledge graph keeps the extracted facts permanently"
              value={draft.memory_purge_days}
              onChange={(v) => set('memory_purge_days', v)}
              min={7}
              max={365}
              unit="days"
            />
          </>
        )}

        {/* ── Pattern & Preference Learning ── */}
        <SectionHeader>
          Pattern & Preference Learning
          <ScheduleBadge time="2:00–2:30 AM" label="nightly" />
        </SectionHeader>
        <InfoBox>
          Atlas studies your behaviour over time — learning your presence patterns, device habits,
          and communication preferences so it can anticipate your needs.
        </InfoBox>
        <NumberInput
          label="Behaviour pattern lookback"
          description="Days of presence and device history used to learn your temporal patterns"
          value={draft.pattern_learning_lookback_days}
          onChange={(v) => set('pattern_learning_lookback_days', v)}
          min={7}
          max={90}
          unit="days"
        />
        <NumberInput
          label="Preference lookback"
          description="Days of conversation history analysed to infer your communication preferences"
          value={draft.preference_learning_lookback_days}
          onChange={(v) => set('preference_learning_lookback_days', v)}
          min={1}
          max={30}
          unit="days"
        />
        <NumberInput
          label="Min turns before updating preferences"
          description="Atlas won't update preferences until you've had at least this many exchanges — prevents premature changes from a single conversation"
          value={draft.preference_learning_min_turns}
          onChange={(v) => set('preference_learning_min_turns', v)}
          min={5}
          max={100}
          unit="messages"
        />

        {/* ── Proactive Intelligence ── */}
        <SectionHeader>
          Proactive Intelligence
          <ScheduleBadge time="6:30 AM + every 4h" label="daily" />
        </SectionHeader>
        <InfoBox>
          Atlas scans recent conversations for commitments, open questions, and action items you may
          have forgotten — then surfaces them as nudges before they slip through the cracks.
        </InfoBox>
        <NumberInput
          label="Action item lookback"
          description="Hours of recent conversation Atlas scans to extract outstanding action items"
          value={draft.proactive_lookback_hours}
          onChange={(v) => set('proactive_lookback_hours', v)}
          min={1}
          max={72}
          unit="hours"
        />
        <NumberInput
          label="Flag as stale after"
          description="Days without activity before Atlas flags a pending action as stale and resurfaces it"
          value={draft.action_escalation_stale_days}
          onChange={(v) => set('action_escalation_stale_days', v)}
          min={1}
          max={30}
          unit="days"
        />
        <NumberInput
          label="Flag as overdue after"
          description="Days past its deadline before Atlas escalates a pending action as overdue"
          value={draft.action_escalation_overdue_days}
          onChange={(v) => set('action_escalation_overdue_days', v)}
          min={1}
          max={14}
          unit="days"
        />

        {/* ── Device & Security ── */}
        <SectionHeader>
          Device & Security
          <ScheduleBadge time="6:00 AM + every 6h" label="daily" />
        </SectionHeader>
        <NumberInput
          label="Low battery threshold"
          description="Devices below this battery percentage are flagged in the daily health check"
          value={draft.device_health_battery_threshold}
          onChange={(v) => set('device_health_battery_threshold', v)}
          min={5}
          max={50}
          unit="%"
        />
        <NumberInput
          label="Stale device threshold"
          description="Hours without a state update before a device is considered offline or stale"
          value={draft.device_health_stale_hours}
          onChange={(v) => set('device_health_stale_hours', v)}
          min={1}
          max={72}
          unit="hours"
        />
        <NumberInput
          label="Security summary lookback"
          description="Hours to look back when generating the periodic security event summary"
          value={draft.security_summary_hours}
          onChange={(v) => set('security_summary_hours', v)}
          min={1}
          max={48}
          unit="hours"
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
