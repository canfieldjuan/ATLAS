/**
 * NotificationSettings — Alerts, ntfy push, reminders, and call intelligence.
 */
import { useState, useEffect, useCallback } from 'react';
import { Save, RefreshCw, Bell, AlertCircle } from 'lucide-react';
import clsx from 'clsx';

const API = '/api/v1/settings/notifications';

interface NotificationSettings {
  alerts_enabled: boolean;
  alerts_cooldown_seconds: number;
  alerts_tts_enabled: boolean;
  alerts_persist: boolean;
  ntfy_enabled: boolean;
  ntfy_url: string;
  ntfy_topic: string;
  reminders_enabled: boolean;
  reminder_timezone: string;
  reminder_max_per_user: number;
  call_intel_enabled: boolean;
  call_min_duration_seconds: number;
  call_notify_enabled: boolean;
}

/* ─── primitives ─────────────────────────────────────────────────────────── */
function Label({ children }: { children: React.ReactNode }) {
  return (
    <span className="block text-[10px] font-semibold uppercase tracking-wider text-cyan-600 mb-1">
      {children}
    </span>
  );
}

function TextInput({
  value,
  onChange,
  placeholder,
  mono,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  mono?: boolean;
}) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className={clsx(
        'w-full bg-[#0a1628] border border-cyan-500/20 rounded-sm px-2.5 py-1.5 text-xs',
        'text-cyan-100 placeholder-cyan-700 focus:outline-none focus:border-cyan-500/60',
        mono && 'font-mono',
      )}
    />
  );
}

function NumInput({
  value,
  onChange,
  min,
  max,
}: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full bg-[#0a1628] border border-cyan-500/20 rounded-sm px-2.5 py-1.5 text-xs text-cyan-100 focus:outline-none focus:border-cyan-500/60"
    />
  );
}

function Toggle({
  value,
  onChange,
  label,
  hint,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
}) {
  return (
    <label className="flex items-start gap-3 cursor-pointer group">
      <div
        onClick={() => onChange(!value)}
        className={clsx(
          'relative shrink-0 mt-0.5 w-8 h-4 rounded-full transition-colors cursor-pointer',
          value ? 'bg-cyan-500' : 'bg-cyan-900/60',
        )}
      >
        <div
          className={clsx(
            'absolute top-0.5 w-3 h-3 rounded-full bg-white shadow transition-transform',
            value ? 'translate-x-4' : 'translate-x-0.5',
          )}
        />
      </div>
      <div>
        <p className="text-xs font-medium text-cyan-200 group-hover:text-cyan-100">{label}</p>
        {hint && <p className="text-[10px] text-cyan-700 mt-0.5">{hint}</p>}
      </div>
    </label>
  );
}

function Section({
  title,
  children,
  badge,
}: {
  title: string;
  children: React.ReactNode;
  badge?: string;
}) {
  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-[10px] font-bold uppercase tracking-widest text-cyan-500">{title}</span>
        {badge && (
          <span className="px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider rounded-sm bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            {badge}
          </span>
        )}
        <div className="flex-1 h-px bg-cyan-500/10" />
      </div>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

/* ─── main component ─────────────────────────────────────────────────────── */
export function NotificationSettingsForm() {
  const [form, setForm] = useState<NotificationSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<{ ok: boolean; msg: string } | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await fetch(API);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setForm(await r.json());
    } catch (e) {
      setStatus({ ok: false, msg: `Load failed: ${e}` });
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const set = <K extends keyof NotificationSettings>(k: K, v: NotificationSettings[K]) =>
    setForm((f) => f ? { ...f, [k]: v } : f);

  const save = async () => {
    if (!form) return;
    setSaving(true);
    setStatus(null);
    try {
      const r = await fetch(API, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setForm(await r.json());
      setStatus({ ok: true, msg: 'Saved successfully' });
    } catch (e) {
      setStatus({ ok: false, msg: `Save failed: ${e}` });
    } finally {
      setSaving(false);
    }
  };

  if (!form) {
    return (
      <div className="flex-1 flex items-center justify-center text-cyan-600 text-xs">
        <RefreshCw size={14} className="animate-spin mr-2" /> Loading…
      </div>
    );
  }

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* scrollable body */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-5 py-4 space-y-1">

        {/* ── Alert System ── */}
        <Section title="Alert System">
          <Toggle
            value={form.alerts_enabled}
            onChange={(v) => set('alerts_enabled', v)}
            label="Enable alert system"
            hint="Central hub that routes security, device, and AI alerts"
          />
          {form.alerts_enabled && (
            <>
              <Toggle
                value={form.alerts_tts_enabled}
                onChange={(v) => set('alerts_tts_enabled', v)}
                label="Speak alerts aloud via TTS"
                hint="Announces alerts on edge nodes with speakers"
              />
              <Toggle
                value={form.alerts_persist}
                onChange={(v) => set('alerts_persist', v)}
                label="Persist alerts to database"
                hint="Saves every alert for history and reporting"
              />
              <div>
                <Label>Cooldown between duplicate alerts (seconds)</Label>
                <NumInput value={form.alerts_cooldown_seconds}
                  onChange={(v) => set('alerts_cooldown_seconds', v)} min={0} max={3600} />
                <p className="text-[10px] text-cyan-700 mt-0.5">
                  Prevents the same alert type from firing repeatedly in a short window.
                </p>
              </div>
            </>
          )}
        </Section>

        {/* ── ntfy Push Notifications ── */}
        <Section title="ntfy Push Notifications">
          <Toggle
            value={form.ntfy_enabled}
            onChange={(v) => set('ntfy_enabled', v)}
            label="Enable ntfy push notifications"
            hint="Delivers alerts to your phone or browser via the ntfy app"
          />
          {form.ntfy_enabled && (
            <>
              <div>
                <Label>ntfy server URL</Label>
                <TextInput value={form.ntfy_url} onChange={(v) => set('ntfy_url', v)}
                  placeholder="https://ntfy.sh" mono />
                <p className="text-[10px] text-cyan-700 mt-0.5">
                  Use ntfy.sh (free) or your self-hosted server.
                </p>
              </div>
              <div>
                <Label>ntfy topic</Label>
                <TextInput value={form.ntfy_topic} onChange={(v) => set('ntfy_topic', v)}
                  placeholder="atlas-alerts" mono />
                <p className="text-[10px] text-cyan-700 mt-0.5">
                  Subscribe to this exact topic in the ntfy mobile or desktop app.
                </p>
              </div>
            </>
          )}
        </Section>

        {/* ── Reminders ── */}
        <Section title="Reminders">
          <Toggle
            value={form.reminders_enabled}
            onChange={(v) => set('reminders_enabled', v)}
            label="Enable reminder system"
            hint='Set reminders by voice: "Remind me to call Bob at 3 PM"'
          />
          {form.reminders_enabled && (
            <>
              <div>
                <Label>Default timezone</Label>
                <TextInput value={form.reminder_timezone}
                  onChange={(v) => set('reminder_timezone', v)}
                  placeholder="America/Chicago" />
                <p className="text-[10px] text-cyan-700 mt-0.5">
                  IANA timezone (e.g. America/New_York, Europe/London, UTC).
                </p>
              </div>
              <div>
                <Label>Max reminders per user</Label>
                <NumInput value={form.reminder_max_per_user}
                  onChange={(v) => set('reminder_max_per_user', v)} min={1} max={1000} />
              </div>
            </>
          )}
        </Section>

        {/* ── Call Intelligence ── */}
        <Section title="Call Intelligence" badge="Twilio">
          <Toggle
            value={form.call_intel_enabled}
            onChange={(v) => set('call_intel_enabled', v)}
            label="Enable post-call AI analysis"
            hint="Transcribes Twilio call recordings and extracts CRM-ready data after every call"
          />
          {form.call_intel_enabled && (
            <>
              <div>
                <Label>Minimum call duration to analyze (seconds)</Label>
                <NumInput value={form.call_min_duration_seconds}
                  onChange={(v) => set('call_min_duration_seconds', v)} min={0} max={600} />
                <p className="text-[10px] text-cyan-700 mt-0.5">
                  Calls shorter than this are skipped to avoid wasting tokens on voicemail drops.
                </p>
              </div>
              <Toggle
                value={form.call_notify_enabled}
                onChange={(v) => set('call_notify_enabled', v)}
                label="Push call summary via ntfy"
                hint="Sends a notification with the AI-generated call summary when processing finishes"
              />
            </>
          )}
        </Section>

      </div>

      {/* sticky footer */}
      <div className="shrink-0 px-5 py-3 border-t border-cyan-500/20 flex items-center gap-3">
        {status && (
          <div className={clsx(
            'flex items-center gap-1.5 text-[10px] flex-1',
            status.ok ? 'text-emerald-400' : 'text-red-400',
          )}>
            {!status.ok && <AlertCircle size={11} />}
            {status.msg}
          </div>
        )}
        <div className="ml-auto flex gap-2">
          <button onClick={load}
            className="flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider border border-cyan-500/20 rounded-sm text-cyan-600 hover:text-cyan-400 transition-all">
            <Bell size={11} /> Reset
          </button>
          <button onClick={save} disabled={saving}
            className="flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider border border-cyan-500/40 rounded-sm bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20 transition-all disabled:opacity-50">
            <Save size={11} />
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
