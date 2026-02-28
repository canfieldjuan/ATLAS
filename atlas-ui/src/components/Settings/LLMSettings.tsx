/**
 * LLMSettings — AI Model configuration form.
 *
 * Covers: backend selection, Ollama / Groq / Together model names,
 * cloud hybrid routing, and automatic day/night model swap.
 */
import { useState, useEffect, useCallback } from 'react';
import { Save, RefreshCw, Cpu, AlertCircle } from 'lucide-react';
import clsx from 'clsx';

const API = '/api/v1/settings/llm';

const BACKENDS = [
  { value: 'ollama',    label: 'Ollama (local)' },
  { value: 'groq',      label: 'Groq (cloud — fast)' },
  { value: 'together',  label: 'Together AI (cloud)' },
  { value: 'llama-cpp', label: 'llama-cpp (GGUF file)' },
  { value: 'hybrid',    label: 'Hybrid (local + cloud)' },
];

interface LLMSettings {
  backend: string;
  ollama_model: string;
  ollama_url: string;
  groq_model: string;
  together_model: string;
  cloud_enabled: boolean;
  cloud_ollama_model: string;
  model_swap_enabled: boolean;
  day_model: string;
  night_model: string;
  model_swap_day_cron: string;
  model_swap_night_cron: string;
}

/* ─── tiny reusable primitives ────────────────────────────────────────────── */
function Label({ children }: { children: React.ReactNode }) {
  return (
    <span className="block text-[10px] font-semibold uppercase tracking-wider text-cyan-600 mb-1">
      {children}
    </span>
  );
}

function Input({
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

/* ─── main component ───────────────────────────────────────────────────────── */
export function LLMSettingsForm() {
  const [form, setForm] = useState<LLMSettings | null>(null);
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

  const set = <K extends keyof LLMSettings>(k: K, v: LLMSettings[K]) =>
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
      setStatus({ ok: true, msg: 'Saved — backend/model-swap changes need a server restart' });
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

  const useOllama   = form.backend === 'ollama'   || form.backend === 'hybrid';
  const useGroq     = form.backend === 'groq';
  const useTogether = form.backend === 'together';

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* scrollable body */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-5 py-4 space-y-1">

        {/* ── Backend ── */}
        <Section title="LLM Backend">
          <div>
            <Label>Active backend</Label>
            <div className="grid grid-cols-2 gap-1.5">
              {BACKENDS.map(({ value, label }) => (
                <button
                  key={value}
                  onClick={() => set('backend', value)}
                  className={clsx(
                    'px-2.5 py-1.5 rounded-sm border text-[10px] font-semibold uppercase tracking-wider transition-all',
                    form.backend === value
                      ? 'border-cyan-400 bg-cyan-500/10 text-cyan-300'
                      : 'border-cyan-500/20 text-cyan-700 hover:border-cyan-500/40 hover:text-cyan-500',
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-cyan-700 mt-1.5">
              Backend changes require a server restart to take full effect.
            </p>
          </div>
        </Section>

        {/* ── Ollama (local) ── */}
        {(useOllama || form.backend === 'llama-cpp') && (
          <Section title="Ollama / Local" badge="local">
            <div>
              <Label>Model tag</Label>
              <Input value={form.ollama_model} onChange={(v) => set('ollama_model', v)}
                placeholder="qwen3:14b" mono />
            </div>
            <div>
              <Label>Ollama API URL</Label>
              <Input value={form.ollama_url} onChange={(v) => set('ollama_url', v)}
                placeholder="http://localhost:11434" mono />
            </div>
          </Section>
        )}

        {/* ── Groq ── */}
        {useGroq && (
          <Section title="Groq" badge="cloud">
            <div>
              <Label>Model name</Label>
              <Input value={form.groq_model} onChange={(v) => set('groq_model', v)}
                placeholder="llama-3.3-70b-versatile" mono />
              <p className="text-[10px] text-cyan-700 mt-1">
                Set <span className="font-mono text-cyan-600">GROQ_API_KEY</span> in your .env file.
              </p>
            </div>
          </Section>
        )}

        {/* ── Together AI ── */}
        {useTogether && (
          <Section title="Together AI" badge="cloud">
            <div>
              <Label>Model name</Label>
              <Input value={form.together_model} onChange={(v) => set('together_model', v)}
                placeholder="meta-llama/Llama-3.3-70B-Instruct-Turbo" mono />
              <p className="text-[10px] text-cyan-700 mt-1">
                Set <span className="font-mono text-cyan-600">TOGETHER_API_KEY</span> in your .env file.
              </p>
            </div>
          </Section>
        )}

        {/* ── Cloud / Hybrid routing ── */}
        <Section title="Cloud Hybrid Routing">
          <Toggle
            value={form.cloud_enabled}
            onChange={(v) => set('cloud_enabled', v)}
            label="Enable cloud LLM for business workflows"
            hint="Runs a second cloud model alongside local for email drafting and bookings"
          />
          {form.cloud_enabled && (
            <div>
              <Label>Cloud model tag (Ollama relay)</Label>
              <Input value={form.cloud_ollama_model} onChange={(v) => set('cloud_ollama_model', v)}
                placeholder="minimax-m2:cloud" mono />
            </div>
          )}
        </Section>

        {/* ── Model Swap ── */}
        <Section title="Day / Night Model Swap">
          <Toggle
            value={form.model_swap_enabled}
            onChange={(v) => set('model_swap_enabled', v)}
            label="Auto-swap models on a schedule"
            hint="Loads a lighter model during the day and a heavier model at night for background tasks"
          />
          {form.model_swap_enabled && (
            <>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>Day model</Label>
                  <Input value={form.day_model} onChange={(v) => set('day_model', v)}
                    placeholder="qwen3:14b" mono />
                </div>
                <div>
                  <Label>Night model</Label>
                  <Input value={form.night_model} onChange={(v) => set('night_model', v)}
                    placeholder="qwen3:32b" mono />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>Day swap cron</Label>
                  <Input value={form.model_swap_day_cron} onChange={(v) => set('model_swap_day_cron', v)}
                    placeholder="30 7 * * *" mono />
                  <p className="text-[10px] text-cyan-700 mt-0.5">Default: 7:30 AM</p>
                </div>
                <div>
                  <Label>Night swap cron</Label>
                  <Input value={form.model_swap_night_cron} onChange={(v) => set('model_swap_night_cron', v)}
                    placeholder="0 0 * * *" mono />
                  <p className="text-[10px] text-cyan-700 mt-0.5">Default: midnight</p>
                </div>
              </div>
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
            <Cpu size={11} /> Reset
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
