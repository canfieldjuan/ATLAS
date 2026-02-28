import { useEffect, useState, useCallback } from 'react';
import { Save, RotateCcw, Loader } from 'lucide-react';
import clsx from 'clsx';

// ---------- types ----------

export interface EmailSettings {
  // General
  enabled: boolean;
  default_from: string | null;
  gmail_send_enabled: boolean;
  timeout: number;
  // IMAP
  imap_host: string;
  imap_port: number;
  imap_username: string;
  imap_ssl: boolean;
  imap_mailbox: string;
  gmail_query: string;
  gmail_max_results: number;
  // Intake
  intake_enabled: boolean;
  intake_interval_seconds: number;
  intake_crm_enabled: boolean;
  intake_action_plan_enabled: boolean;
  intake_max_action_plans_per_cycle: number;
  // Draft
  draft_enabled: boolean;
  draft_auto_draft_enabled: boolean;
  draft_model_name: string;
  draft_temperature: number;
  draft_expiry_hours: number;
  draft_notify_drafts: boolean;
  draft_schedule_interval_seconds: number;
  draft_triage_enabled: boolean;
}

// ---------- helpers ----------

const API_BASE = '/api/v1/settings';

async function fetchEmailSettings(): Promise<EmailSettings> {
  const res = await fetch(`${API_BASE}/email`);
  if (!res.ok) throw new Error(`Failed to load email settings: ${res.status} ${res.statusText}`);
  return res.json();
}

async function saveEmailSettings(patch: Partial<EmailSettings>): Promise<EmailSettings> {
  const res = await fetch(`${API_BASE}/email`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error(`Failed to save email settings: ${res.status} ${res.statusText}`);
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
  label, description, value, onChange, placeholder, type = 'text',
}: {
  label: string; description?: string; value: string; onChange: (v: string) => void;
  placeholder?: string; type?: string;
}) {
  return (
    <div className="py-2">
      <label className="text-sm text-cyan-200 block">{label}</label>
      {description && <div className="text-[11px] text-cyan-600 mb-1">{description}</div>}
      <input
        type={type}
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

// ---------- main form ----------

interface EmailSettingsFormProps {
  onDirtyChange?: (dirty: boolean) => void;
}

export function EmailSettingsForm({ onDirtyChange }: EmailSettingsFormProps) {
  const [original, setOriginal] = useState<EmailSettings | null>(null);
  const [draft, setDraft] = useState<EmailSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchEmailSettings();
      setOriginal(data);
      setDraft(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load email settings');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const set = <K extends keyof EmailSettings>(key: K, value: EmailSettings[K]) => {
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
      const patch: Partial<EmailSettings> = {};
      (Object.keys(draft) as (keyof EmailSettings)[]).forEach(<K extends keyof EmailSettings>(k: K) => {
        if (draft[k] !== original[k]) patch[k] = draft[k];
      });
      if (Object.keys(patch).length === 0) { setSaved(true); return; }
      const updated = await saveEmailSettings(patch);
      setOriginal(updated);
      setDraft(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save email settings');
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
        <span className="text-sm">Loading email settings…</span>
      </div>
    );
  }

  if (!draft) {
    return (
      <div className="bg-red-900/20 border border-red-500/30 rounded px-3 py-2 text-sm text-red-400">
        {error ?? 'Could not load email settings.'}
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
        <Toggle
          label="Email system enabled"
          description="Master switch — enables sending, reading, and all autonomous email features"
          checked={draft.enabled}
          onChange={(v) => set('enabled', v)}
        />
        <TextInput
          label="Default sender address"
          description="From: address used when no sender is specified"
          value={draft.default_from ?? ''}
          onChange={(v) => set('default_from', v || null)}
          placeholder="you@gmail.com"
        />
        <Toggle
          label="Prefer Gmail API for sending"
          description="Use OAuth2 Gmail API when available; falls back to Resend API when off"
          checked={draft.gmail_send_enabled}
          onChange={(v) => set('gmail_send_enabled', v)}
        />
        <NumberInput
          label="API request timeout"
          value={draft.timeout}
          onChange={(v) => set('timeout', v)}
          min={3}
          max={60}
          unit="seconds"
        />

        {/* ── Inbox / IMAP ── */}
        <SectionHeader>Inbox (IMAP)</SectionHeader>
        <InfoBox>
          IMAP works with any provider — Gmail, Outlook, Yahoo, or self-hosted.
          Passwords/app-passwords are set via the <code className="text-cyan-400">ATLAS_EMAIL_IMAP_PASSWORD</code> env variable.
        </InfoBox>
        <TextInput
          label="IMAP server host"
          description="e.g. imap.gmail.com · outlook.office365.com · mail.example.com"
          value={draft.imap_host}
          onChange={(v) => set('imap_host', v)}
          placeholder="imap.gmail.com"
        />
        <div className="flex gap-4">
          <div className="flex-1">
            <NumberInput
              label="IMAP port"
              description="993 = SSL  ·  143 = STARTTLS"
              value={draft.imap_port}
              onChange={(v) => set('imap_port', v)}
              min={1}
              max={65535}
            />
          </div>
          <div className="pt-2 flex items-end pb-3">
            <Toggle
              label="Use SSL"
              checked={draft.imap_ssl}
              onChange={(v) => set('imap_ssl', v)}
            />
          </div>
        </div>
        <TextInput
          label="IMAP username"
          description="Usually your full email address"
          value={draft.imap_username}
          onChange={(v) => set('imap_username', v)}
          placeholder="you@gmail.com"
        />
        <TextInput
          label="Default mailbox"
          description="IMAP folder to read — typically INBOX"
          value={draft.imap_mailbox}
          onChange={(v) => set('imap_mailbox', v)}
          placeholder="INBOX"
        />
        <TextInput
          label="Default inbox query"
          description="Filter applied when fetching inbox — e.g. is:unread newer_than:1d"
          value={draft.gmail_query}
          onChange={(v) => set('gmail_query', v)}
          placeholder="is:unread newer_than:1d"
        />
        <NumberInput
          label="Max emails per fetch"
          description="Cap on the number of messages returned in a single inbox read"
          value={draft.gmail_max_results}
          onChange={(v) => set('gmail_max_results', v)}
          min={1}
          max={50}
          unit="emails"
        />

        {/* ── Autonomous Intake ── */}
        <SectionHeader>Autonomous Intake Polling</SectionHeader>
        <InfoBox>
          When enabled, Atlas polls your inbox every <strong>{Math.round(draft.intake_interval_seconds / 60)} min</strong>,
          classifies emails, cross-references senders with your CRM, and can generate AI action plans.
        </InfoBox>
        <Toggle
          label="Enable background polling"
          description="Atlas silently checks for new emails on a schedule"
          checked={draft.intake_enabled}
          onChange={(v) => set('intake_enabled', v)}
        />
        {draft.intake_enabled && (
          <>
            <NumberInput
              label="Poll interval"
              description="Minimum 300 s (5 min) — more frequent polls increase API usage"
              value={draft.intake_interval_seconds}
              onChange={(v) => set('intake_interval_seconds', Math.max(300, v))}
              min={300}
              max={3600}
              step={60}
              unit="seconds"
            />
            <Toggle
              label="CRM cross-reference"
              description="Look up incoming senders in your contacts database and enrich notifications"
              checked={draft.intake_crm_enabled}
              onChange={(v) => set('intake_crm_enabled', v)}
            />
            <Toggle
              label="AI action plans"
              description="Generate suggested follow-up actions for CRM-matched emails (uses LLM credits)"
              checked={draft.intake_action_plan_enabled}
              onChange={(v) => set('intake_action_plan_enabled', v)}
            />
            {draft.intake_action_plan_enabled && (
              <NumberInput
                label="Max action plans per cycle"
                description="Caps LLM calls per polling run to control cost"
                value={draft.intake_max_action_plans_per_cycle}
                onChange={(v) => set('intake_max_action_plans_per_cycle', v)}
                min={1}
                max={20}
                unit="plans"
              />
            )}
          </>
        )}

        {/* ── AI Draft Generation ── */}
        <SectionHeader>AI Draft Generation</SectionHeader>
        <InfoBox>
          Atlas can draft replies using an LLM. Drafts are held for your approval — nothing is
          sent automatically unless you click <em>Approve &amp; Send</em>.
        </InfoBox>
        <Toggle
          label="Enable draft generation"
          description="Allow Atlas to compose reply drafts for incoming emails"
          checked={draft.draft_enabled}
          onChange={(v) => set('draft_enabled', v)}
        />
        {draft.draft_enabled && (
          <>
            <Toggle
              label="Auto-generate on a schedule"
              description="Automatically draft replies for action-required emails — when off, trigger drafts manually via the 'Draft Reply' notification button"
              checked={draft.draft_auto_draft_enabled}
              onChange={(v) => set('draft_auto_draft_enabled', v)}
            />
            {draft.draft_auto_draft_enabled && (
              <NumberInput
                label="Draft check interval"
                description="How often to scan for emails that need a draft"
                value={draft.draft_schedule_interval_seconds}
                onChange={(v) => set('draft_schedule_interval_seconds', Math.max(300, v))}
                min={300}
                max={7200}
                step={300}
                unit="seconds"
              />
            )}
            <TextInput
              label="LLM model"
              description="Anthropic model used to write drafts — larger models produce better results"
              value={draft.draft_model_name}
              onChange={(v) => set('draft_model_name', v)}
              placeholder="claude-sonnet-4-5-20250929"
            />
            <SliderInput
              label="Draft creativity"
              description="Lower values produce safer, more predictable replies; higher values more varied"
              value={draft.draft_temperature}
              onChange={(v) => set('draft_temperature', v)}
              min={0.0}
              max={1.0}
              step={0.05}
              format={(v) => v.toFixed(2)}
            />
            <NumberInput
              label="Draft expiry"
              description="Hours a pending draft stays available in the approval queue before auto-expiring"
              value={draft.draft_expiry_hours}
              onChange={(v) => set('draft_expiry_hours', v)}
              min={1}
              max={168}
              unit="hours"
            />
            <Toggle
              label="Push notifications for new drafts"
              description="Send a push notification (via ntfy) when a new draft is ready to review"
              checked={draft.draft_notify_drafts}
              onChange={(v) => set('draft_notify_drafts', v)}
            />
            <Toggle
              label="Email triage"
              description="Use a fast, cheap LLM to decide whether an email deserves a reply before spending tokens on drafting"
              checked={draft.draft_triage_enabled}
              onChange={(v) => set('draft_triage_enabled', v)}
            />
          </>
        )}
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
