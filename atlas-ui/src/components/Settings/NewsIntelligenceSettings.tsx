import { useEffect, useState, useCallback } from 'react';
import { Save, RotateCcw, Loader, Plus, Trash2, ChevronDown, ChevronUp } from 'lucide-react';
import clsx from 'clsx';

// ---------- types ----------

export interface IntelligenceSettings {
  enabled: boolean;
  watchlist: string;          // JSON string
  topics: string;             // simple-mode fallback
  regions: string;
  languages: string;
  lookback_days: number;
  pressure_velocity_threshold: number;
  signal_min_articles: number;
  sentiment_enabled: boolean;
  source_diversity_enabled: boolean;
  composite_score_threshold: number;
  max_articles_per_topic: number;
  llm_model: string;
  schedule_hour: number;
  notify_on_signal: boolean;
  notify_all_runs: boolean;
  include_in_morning_briefing: boolean;
  // Linguistic pre-indicator patterns
  linguistic_analysis_enabled: boolean;
  linguistic_hedge_enabled: boolean;
  linguistic_deflection_enabled: boolean;
  linguistic_insider_enabled: boolean;
  linguistic_escalation_enabled: boolean;
  linguistic_permission_enabled: boolean;
  linguistic_certainty_enabled: boolean;
  linguistic_dissociation_enabled: boolean;
  // SORAM Framework
  soram_enabled: boolean;
  soram_societal_enabled: boolean;
  soram_operational_enabled: boolean;
  soram_regulatory_enabled: boolean;
  soram_alignment_enabled: boolean;
  soram_media_novelty_enabled: boolean;
  // Alternative data sources
  sec_edgar_enabled: boolean;
  usaspending_enabled: boolean;
  state_sos_enabled: boolean;
  county_recorder_enabled: boolean;
  bls_enabled: boolean;
  // Signal streak / correlation
  signal_streak_enabled: boolean;
  signal_streak_threshold: number;
  cross_entity_correlation_enabled: boolean;
  cross_entity_min_signals: number;
}

export type EntityType = 'company' | 'sports_team' | 'market' | 'crypto' | 'custom';

export interface WatchlistEntry {
  name: string;
  type: EntityType;
  query: string;
  ticker: string;
}

// ---------- helpers ----------

const API_BASE = '/api/v1/settings';

async function fetchIntelligenceSettings(): Promise<IntelligenceSettings> {
  const res = await fetch(`${API_BASE}/intelligence`);
  if (!res.ok) throw new Error(`Failed to load settings: ${res.status} ${res.statusText}`);
  return res.json();
}

async function saveIntelligenceSettings(patch: Partial<IntelligenceSettings>): Promise<IntelligenceSettings> {
  const res = await fetch(`${API_BASE}/intelligence`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error(`Failed to save settings: ${res.status} ${res.statusText}`);
  return res.json();
}

function parseWatchlist(raw: string): WatchlistEntry[] {
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed as WatchlistEntry[];
  } catch { /* ignore */ }
  return [];
}

function serializeWatchlist(entries: WatchlistEntry[]): string {
  return JSON.stringify(entries);
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

function Toggle({ label, description, checked, onChange }: {
  label: string; description?: string; checked: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-2">
      <div className="min-w-0">
        <div className="text-sm text-cyan-200">{label}</div>
        {description && <div className="text-[11px] text-cyan-600 mt-0.5">{description}</div>}
      </div>
      <button type="button" onClick={() => onChange(!checked)}
        className={clsx(
          'relative shrink-0 w-10 h-5 rounded-full border transition-all duration-200',
          checked ? 'bg-cyan-500/30 border-cyan-500 shadow-[0_0_8px_rgba(34,211,238,0.4)]' : 'bg-black/30 border-cyan-500/30',
        )}
        aria-checked={checked} role="switch">
        <span className={clsx('absolute top-0.5 w-4 h-4 rounded-full transition-all duration-200',
          checked ? 'left-5 bg-cyan-400' : 'left-0.5 bg-cyan-700')} />
      </button>
    </div>
  );
}

function NumberInput({ label, description, value, onChange, min, max, step, unit }: {
  label: string; description?: string; value: number; onChange: (v: number) => void;
  min?: number; max?: number; step?: number; unit?: string;
}) {
  return (
    <div className="py-2">
      <label className="text-sm text-cyan-200 block">{label}</label>
      {description && <div className="text-[11px] text-cyan-600 mb-1">{description}</div>}
      <div className="flex items-center gap-2 mt-1">
        <input type="number" value={value} min={min} max={max} step={step ?? 1}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-28 bg-black/30 border border-cyan-500/30 rounded px-2 py-1.5 text-sm text-cyan-200 outline-none focus:border-cyan-500/60 tabular-nums transition-all" />
        {unit && <span className="text-[11px] text-cyan-600">{unit}</span>}
      </div>
    </div>
  );
}

function SliderInput({ label, description, value, onChange, min, max, step, format }: {
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
      <input type="range" min={min} max={max} step={step ?? 0.1} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full mt-1 accent-cyan-400 cursor-pointer" />
      <div className="flex justify-between text-[10px] text-cyan-700 mt-0.5">
        <span>{min}</span><span>{max}</span>
      </div>
    </div>
  );
}

function TextInput({ label, description, value, onChange, placeholder }: {
  label: string; description?: string; value: string; onChange: (v: string) => void; placeholder?: string;
}) {
  return (
    <div className="py-2">
      <label className="text-sm text-cyan-200 block">{label}</label>
      {description && <div className="text-[11px] text-cyan-600 mb-1">{description}</div>}
      <input type="text" value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder}
        className="w-full mt-1 bg-black/30 border border-cyan-500/30 rounded px-2 py-1.5 text-sm text-cyan-200 placeholder-cyan-700/50 outline-none focus:border-cyan-500/60 transition-all" />
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

// ---------- How It Works Explainer ----------

function PressureExplainer() {
  return (
    <div className="bg-black/20 border border-cyan-500/15 rounded px-3 py-3 my-3">
      <div className="text-[10px] uppercase tracking-widest text-cyan-500/50 mb-2">How pre-movement pressure works</div>

      {/* Timeline */}
      <div className="flex items-stretch gap-0 text-[10px] mb-3">
        <div className="flex flex-col items-center w-5 shrink-0">
          <div className="w-1.5 h-1.5 rounded-full bg-cyan-800 mt-1" />
          <div className="flex-1 w-px bg-gradient-to-b from-cyan-800 via-cyan-500 to-cyan-400" />
          <div className="w-2 h-2 rounded-full bg-cyan-400 mb-1" />
        </div>
        <div className="flex flex-col justify-between ml-2 gap-3">
          <div>
            <span className="text-cyan-600 font-bold">PRESSURE BUILDS</span>
            <span className="text-cyan-700"> — article velocity accelerates in niche &amp; trade outlets,
            tone shifts, new sources start covering it. Still below mainstream radar.</span>
          </div>
          <div>
            <span className="text-cyan-400 font-bold">MOVEMENT</span>
            <span className="text-cyan-600"> — price moves, odds shift, or narrative changes.
            Wall Street / mainstream media report it. By then Atlas already flagged the build-up.</span>
          </div>
        </div>
      </div>

      {/* What you can track */}
      <div className="grid grid-cols-2 gap-1.5 mb-3">
        {[
          { icon: '📈', label: 'Companies', ex: 'AAPL, TSLA, NVDA — supply chain, earnings whispers' },
          { icon: '📊', label: 'Markets', ex: 'S&P 500, Oil, Gold — macro pressure shifts' },
          { icon: '₿', label: 'Crypto', ex: 'BTC, ETH — regulatory & adoption signals' },
          { icon: '🏆', label: 'Sports Teams', ex: 'Cowboys, Lakers — roster & injury news' },
        ].map(({ icon, label, ex }) => (
          <div key={label} className="bg-black/20 rounded px-2 py-1.5 border border-cyan-500/10">
            <div className="text-[10px] font-bold text-cyan-400">{icon} {label}</div>
            <div className="text-[10px] text-cyan-700 mt-0.5">{ex}</div>
          </div>
        ))}
      </div>

      {/* Five signals */}
      <div className="text-[10px] text-cyan-700">
        <span className="text-cyan-500 font-semibold">Five pressure dimensions</span>
        {' — '}score = velocity × sentiment × diversity × linguistic × SORAM
      </div>
      <div className="grid grid-cols-5 gap-1 mt-1.5">
        {[
          { label: 'Volume',    color: 'text-cyan-400',   desc: 'More articles?' },
          { label: 'Sentiment', color: 'text-amber-400',  desc: 'Tone shifting?' },
          { label: 'Diversity', color: 'text-purple-400', desc: 'New outlets?' },
          { label: 'Linguistic',color: 'text-rose-400',   desc: 'Pre-event language?' },
          { label: 'SORAM',     color: 'text-green-400',  desc: 'Societal levers?' },
        ].map(({ label, color, desc }) => (
          <div key={label} className="bg-black/20 rounded px-2 py-1 border border-cyan-500/10 text-center">
            <div className={clsx('text-[10px] font-bold', color)}>{label}</div>
            <div className="text-[9px] text-cyan-700 mt-0.5">{desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------- SORAM Explainer ----------

function SORAMExplainer() {
  return (
    <div className="bg-black/20 border border-green-500/15 rounded px-3 py-3 my-3">
      <div className="text-[10px] uppercase tracking-widest text-green-500/50 mb-2">SORAM Framework — Chase Hughes</div>
      <div className="text-[10px] text-cyan-700 mb-2">
        Hughes identifies five societal "levers" pulled simultaneously before major events.
        When multiple channels activate at once, the probability of imminent movement rises sharply.
      </div>
      <div className="grid grid-cols-1 gap-1">
        {[
          { code: 'S', label: 'Societal',    color: 'text-green-400',  desc: 'Coordinated threat/fear framing spread across unrelated platforms' },
          { code: 'O', label: 'Operational', color: 'text-cyan-400',   desc: 'Increase in "drills", "exercises", and "readiness" reporting' },
          { code: 'R', label: 'Regulatory',  color: 'text-amber-400',  desc: 'Quiet introduction of emergency powers or laws only useful in a specific crisis' },
          { code: 'A', label: 'Alignment',   color: 'text-purple-400', desc: 'Govt, media, and tech using the exact same phrasing — scripted consensus' },
          { code: 'M', label: 'Media',       color: 'text-rose-400',   desc: '"Breaking news" novelty hijacking — keeps brains in high suggestibility' },
        ].map(({ code, label, color, desc }) => (
          <div key={code} className="flex items-start gap-2 bg-black/20 rounded px-2 py-1.5 border border-green-500/10">
            <span className={clsx('text-[11px] font-bold shrink-0 w-4', color)}>{code}</span>
            <div>
              <span className={clsx('text-[10px] font-bold', color)}>{label}</span>
              <span className="text-[10px] text-cyan-700 ml-1">{desc}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------- Entity Type Badge ----------

const TYPE_META: Record<EntityType, { icon: string; color: string; label: string }> = {
  company:     { icon: '📈', color: 'text-cyan-400 border-cyan-500/40 bg-cyan-500/10', label: 'Company' },
  sports_team: { icon: '🏆', color: 'text-amber-400 border-amber-500/40 bg-amber-500/10', label: 'Sports Team' },
  market:      { icon: '📊', color: 'text-purple-400 border-purple-500/40 bg-purple-500/10', label: 'Market' },
  crypto:      { icon: '₿', color: 'text-orange-400 border-orange-500/40 bg-orange-500/10', label: 'Crypto' },
  custom:      { icon: '◉', color: 'text-cyan-600 border-cyan-700/40 bg-cyan-900/20', label: 'Custom' },
};

function TypeBadge({ type }: { type: EntityType }) {
  const meta = TYPE_META[type] ?? TYPE_META.custom;
  return (
    <span className={clsx('text-[10px] px-1.5 py-0.5 rounded border font-bold', meta.color)}>
      {meta.icon} {meta.label}
    </span>
  );
}

// ---------- Single Watchlist Entry Card ----------

interface EntryCardProps {
  entry: WatchlistEntry;
  onChange: (e: WatchlistEntry) => void;
  onDelete: () => void;
}

function EntryCard({ entry, onChange, onDelete }: EntryCardProps) {
  const [expanded, setExpanded] = useState(false);

  const set = <K extends keyof WatchlistEntry>(k: K, v: WatchlistEntry[K]) =>
    onChange({ ...entry, [k]: v });

  return (
    <div className="border border-cyan-500/20 rounded bg-black/20 overflow-hidden">
      {/* collapsed header */}
      <div className="flex items-center gap-2 px-3 py-2">
        <button type="button" onClick={() => setExpanded((x) => !x)} className="flex-1 flex items-center gap-2 text-left min-w-0">
          {entry.name
            ? <span className="text-sm text-cyan-200 font-semibold truncate">{entry.name}</span>
            : <span className="text-sm text-cyan-700 font-semibold truncate">Unnamed entity</span>
          }
          {entry.ticker && (
            <span className="text-[10px] text-cyan-600 font-mono border border-cyan-700/40 px-1 rounded shrink-0">
              {entry.ticker}
            </span>
          )}
          <TypeBadge type={entry.type} />
          <span className="ml-auto shrink-0 text-cyan-600">{expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}</span>
        </button>
        <button type="button" onClick={onDelete} title="Remove entity"
          className="shrink-0 text-red-500/50 hover:text-red-400 transition-colors p-0.5">
          <Trash2 size={13} />
        </button>
      </div>

      {/* expanded editor */}
      {expanded && (
        <div className="px-3 pb-3 border-t border-cyan-500/10 pt-2 space-y-2">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] text-cyan-600 uppercase tracking-wider">Name</label>
              <input value={entry.name} onChange={(e) => set('name', e.target.value)}
                placeholder="e.g. Apple Inc"
                className="w-full mt-0.5 bg-black/30 border border-cyan-500/30 rounded px-2 py-1 text-xs text-cyan-200 outline-none focus:border-cyan-500/60" />
            </div>
            <div>
              <label className="text-[10px] text-cyan-600 uppercase tracking-wider">Ticker / Symbol</label>
              <input value={entry.ticker} onChange={(e) => set('ticker', e.target.value)}
                placeholder="e.g. AAPL"
                className="w-full mt-0.5 bg-black/30 border border-cyan-500/30 rounded px-2 py-1 text-xs text-cyan-200 font-mono outline-none focus:border-cyan-500/60" />
            </div>
          </div>

          <div>
            <label className="text-[10px] text-cyan-600 uppercase tracking-wider">Type</label>
            <select value={entry.type} onChange={(e) => set('type', e.target.value as EntityType)}
              className="w-full mt-0.5 bg-black/50 border border-cyan-500/30 rounded px-2 py-1 text-xs text-cyan-200 outline-none focus:border-cyan-500/60">
              {(Object.keys(TYPE_META) as EntityType[]).map((t) => (
                <option key={t} value={t}>{TYPE_META[t].icon} {TYPE_META[t].label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-[10px] text-cyan-600 uppercase tracking-wider">Search query</label>
            <div className="text-[10px] text-cyan-700 mb-0.5">NewsAPI search string — use OR for aliases, quotes for exact phrases</div>
            <input value={entry.query} onChange={(e) => set('query', e.target.value)}
              placeholder='e.g. Apple OR AAPL "supply chain" earnings'
              className="w-full mt-0.5 bg-black/30 border border-cyan-500/30 rounded px-2 py-1 text-xs text-cyan-200 outline-none focus:border-cyan-500/60" />
          </div>
        </div>
      )}
    </div>
  );
}

// ---------- Watchlist Editor ----------

interface WatchlistEditorProps {
  value: string;
  onChange: (v: string) => void;
}

const ENTITY_TEMPLATES: Record<EntityType, Omit<WatchlistEntry, 'name' | 'ticker'>> = {
  company:     { type: 'company',     query: 'Company OR TICKER earnings supply chain' },
  sports_team: { type: 'sports_team', query: 'Team Name roster injury trade' },
  market:      { type: 'market',      query: 'INDEX OR SYMBOL price outlook' },
  crypto:      { type: 'crypto',      query: 'COIN regulation adoption price' },
  custom:      { type: 'custom',      query: 'your search terms here' },
};

function WatchlistEditor({ value, onChange }: WatchlistEditorProps) {
  const entries = parseWatchlist(value);
  const [addType, setAddType] = useState<EntityType>('company');

  const update = (newEntries: WatchlistEntry[]) => onChange(serializeWatchlist(newEntries));

  const addEntry = () => {
    const tmpl = ENTITY_TEMPLATES[addType];
    update([...entries, { name: '', ticker: '', ...tmpl }]);
  };

  const removeEntry = (i: number) => update(entries.filter((_, idx) => idx !== i));
  const changeEntry = (i: number, e: WatchlistEntry) => update(entries.map((x, idx) => idx === i ? e : x));

  return (
    <div className="space-y-2">
      {entries.length === 0 && (
        <div className="border border-dashed border-cyan-500/20 rounded px-3 py-4 text-center text-[11px] text-cyan-700">
          No entities in the watchlist — add one below to start tracking
        </div>
      )}

      {entries.map((entry, i) => (
        <EntryCard key={i} entry={entry} onChange={(e) => changeEntry(i, e)} onDelete={() => removeEntry(i)} />
      ))}

      {/* Add new entity row */}
      <div className="flex items-center gap-2 pt-1">
        <select value={addType} onChange={(e) => setAddType(e.target.value as EntityType)}
          className="flex-1 bg-black/40 border border-cyan-500/25 rounded px-2 py-1.5 text-xs text-cyan-400 outline-none focus:border-cyan-500/50">
          {(Object.keys(TYPE_META) as EntityType[]).map((t) => (
            <option key={t} value={t}>{TYPE_META[t].icon} {TYPE_META[t].label}</option>
          ))}
        </select>
        <button type="button" onClick={addEntry}
          className="flex items-center gap-1 px-3 py-1.5 rounded border border-cyan-500/40 text-xs text-cyan-400 hover:border-cyan-500 hover:bg-cyan-500/10 transition-all">
          <Plus size={12} />
          Add entity
        </button>
      </div>
    </div>
  );
}

// ---------- Collapsible Advanced Section ----------

function Collapsible({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-cyan-500/15 rounded mt-3">
      <button type="button" onClick={() => setOpen((x) => !x)}
        className="w-full flex items-center justify-between px-3 py-2 text-xs text-cyan-600 hover:text-cyan-400 transition-colors">
        <span className="font-bold uppercase tracking-wider">{title}</span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      {open && <div className="px-3 pb-3 border-t border-cyan-500/10">{children}</div>}
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
      setError(e instanceof Error ? e.message : 'Failed to load settings');
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

  useEffect(() => { onDirtyChange?.(!!isDirty); }, [isDirty, onDirtyChange]);

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
      setError(e instanceof Error ? e.message : 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => { if (original) { setDraft(original); setSaved(false); } };

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
          label="Enable intelligence analysis"
          description="Run the daily pre-movement pressure scan. Requires a NewsAPI.org API key."
          checked={draft.enabled}
          onChange={(v) => set('enabled', v)}
        />
        <CredentialNote envVar="ATLAS_NEWS_API_KEY" />

        {/* ── Watchlist ── */}
        <SectionHeader>Watchlist</SectionHeader>
        <InfoBox>
          Each entity gets its own independent pressure scan. Use specific, targeted search queries
          — include ticker symbols, common abbreviations, and key terms associated with the entity.
          <br /><br />
          <strong className="text-cyan-400">Good query:</strong>{' '}
          <code className="text-cyan-500">Apple OR AAPL "supply chain" earnings</code>
          <br />
          <strong className="text-cyan-400">Avoid:</strong>{' '}
          generic single words that match unrelated stories.
        </InfoBox>
        <WatchlistEditor
          value={draft.watchlist}
          onChange={(v) => set('watchlist', v)}
        />

        {/* ── Signal Detection ── */}
        <SectionHeader>Signal Detection</SectionHeader>

        <div className="space-y-1">
          <Toggle
            label="Sentiment shift scoring"
            description="Amplify signal when tone suddenly becomes more negative or positive"
            checked={draft.sentiment_enabled}
            onChange={(v) => set('sentiment_enabled', v)}
          />
          <Toggle
            label="Source diversity scoring"
            description="Amplify signal when coverage spreads to new outlets — stories moving from trade press to mainstream are strong leading indicators"
            checked={draft.source_diversity_enabled}
            onChange={(v) => set('source_diversity_enabled', v)}
          />
        </div>

        <SliderInput
          label="Composite score threshold"
          description="Minimum composite score to flag a signal — lower = more sensitive, higher = fewer false positives"
          value={draft.composite_score_threshold}
          onChange={(v) => set('composite_score_threshold', v)}
          min={1.0}
          max={5.0}
          step={0.1}
          format={(v) => `${v.toFixed(1)}×`}
        />
        <NumberInput
          label="Baseline window"
          description="Days of history used to calculate each entity's normal article rate"
          value={draft.lookback_days}
          onChange={(v) => set('lookback_days', v)}
          min={2}
          max={30}
          unit="days"
        />
        <NumberInput
          label="Minimum confirming articles"
          description="Entity needs at least this many articles today to confirm a signal"
          value={draft.signal_min_articles}
          onChange={(v) => set('signal_min_articles', v)}
          min={1}
          max={20}
          unit="articles"
        />

        {/* ── Linguistic Pre-Indicators ── */}
        <SectionHeader>Linguistic Pre-Indicators</SectionHeader>
        <InfoBox>
          Seven language pattern types that statistically appear in coverage <strong className="text-cyan-400">before</strong> a
          major movement. The system scans every article headline and description for these patterns and uses them to amplify
          the composite pressure score.
        </InfoBox>

        <Toggle
          label="Enable linguistic analysis"
          description="Master toggle for all linguistic pre-indicator pattern detection"
          checked={draft.linguistic_analysis_enabled}
          onChange={(v) => set('linguistic_analysis_enabled', v)}
        />

        {draft.linguistic_analysis_enabled && (
          <div className="ml-3 border-l border-cyan-500/20 pl-3 space-y-0.5 mt-1">
            <div className="text-[10px] uppercase tracking-widest text-cyan-600/60 mb-1.5">Original four (pre-event signals)</div>
            <Toggle label="Hedging language"
              description="'reportedly', 'could', 'may', 'unconfirmed' — uncertainty builds before disclosure"
              checked={draft.linguistic_hedge_enabled}
              onChange={(v) => set('linguistic_hedge_enabled', v)} />
            <Toggle label="Deflection / denial"
              description="'denies', 'refuses to comment', 'pushes back' — denial clusters appear right before breaks"
              checked={draft.linguistic_deflection_enabled}
              onChange={(v) => set('linguistic_deflection_enabled', v)} />
            <Toggle label="Insider sourcing"
              description="'sources say', 'people familiar', 'obtained by' — information leakage before official disclosure"
              checked={draft.linguistic_insider_enabled}
              onChange={(v) => set('linguistic_insider_enabled', v)} />
            <Toggle label="Escalation language"
              description="'breaking', 'crisis', 'urgent', 'imminent' — urgency in trade press before mainstream pickup"
              checked={draft.linguistic_escalation_enabled}
              onChange={(v) => set('linguistic_escalation_enabled', v)} />

            <div className="text-[10px] uppercase tracking-widest text-cyan-600/60 mt-3 mb-1.5">SORAM behavioral layer (three new)</div>
            <Toggle label="Permission shifts"
              description="'must be stopped', 'for the greater good', 'no option but' — moral permission granted to act against prior values"
              checked={draft.linguistic_permission_enabled}
              onChange={(v) => set('linguistic_permission_enabled', v)} />
            <Toggle label="Certainty / moral panic"
              description="'undeniable', 'settled', 'beyond doubt', 'no debate' — absolute language + emotional triggers precede coordinated narratives"
              checked={draft.linguistic_certainty_enabled}
              onChange={(v) => set('linguistic_certainty_enabled', v)} />
            <Toggle label="Linguistic dissociation"
              description="'these people', 'their kind', 'outsiders' — we/us → they/them shift precedes mobilization events"
              checked={draft.linguistic_dissociation_enabled}
              onChange={(v) => set('linguistic_dissociation_enabled', v)} />
          </div>
        )}

        {/* ── SORAM Framework ── */}
        <SectionHeader>SORAM Framework</SectionHeader>
        <SORAMExplainer />

        <Toggle
          label="Enable SORAM analysis"
          description="Score the five societal pressure channels (S/O/R/A/M) — when multiple channels activate simultaneously, imminent movement probability rises"
          checked={draft.soram_enabled}
          onChange={(v) => set('soram_enabled', v)}
        />

        {draft.soram_enabled && (
          <div className="ml-3 border-l border-green-500/20 pl-3 space-y-0.5 mt-1">
            <Toggle label="S — Societal"
              description="Detect coordinated threat/fear framing: 'national security', 'existential threat', 'misinformation'"
              checked={draft.soram_societal_enabled}
              onChange={(v) => set('soram_societal_enabled', v)} />
            <Toggle label="O — Operational"
              description="Detect drills and exercises: 'tabletop', 'readiness exercise', 'war game', 'simulation'"
              checked={draft.soram_operational_enabled}
              onChange={(v) => set('soram_operational_enabled', v)} />
            <Toggle label="R — Regulatory"
              description="Detect new emergency powers/laws: 'executive order', 'emergency declaration', 'emergency measure'"
              checked={draft.soram_regulatory_enabled}
              onChange={(v) => set('soram_regulatory_enabled', v)} />
            <Toggle label="A — Alignment"
              description="Detect scripted consensus: 'experts agree', 'officials confirm', 'fact-checkers', 'scientists agree'"
              checked={draft.soram_alignment_enabled}
              onChange={(v) => set('soram_alignment_enabled', v)} />
            <Toggle label="M — Media Novelty"
              description="Detect novelty hijacking: 'breaking:', 'just in', 'bombshell', 'stunning' — high-suggestibility priming"
              checked={draft.soram_media_novelty_enabled}
              onChange={(v) => set('soram_media_novelty_enabled', v)} />
          </div>
        )}

        {/* ── Data Sources ── */}
        <SectionHeader>Alternative Data Sources</SectionHeader>
        <InfoBox>
          External data sources provide independent confirmation signals beyond news articles.
          When a source flags elevated activity for the same entity simultaneously with a news signal,
          the composite pressure score receives a bonus multiplier.
          <br /><br />
          <strong className="text-cyan-400">Free APIs (no key required):</strong> SEC EDGAR, USAspending.gov
          <br />
          <strong className="text-cyan-400">Custom setup required:</strong> State SoS, County Recorder, BLS/Census
        </InfoBox>

        <div className="space-y-1">
          <Toggle
            label="SEC EDGAR 8-K filings"
            description="Fetch recent material-event disclosures for company/crypto entities — elevated 8-K activity signals undisclosed events. Free EDGAR API, no key needed."
            checked={draft.sec_edgar_enabled}
            onChange={(v) => set('sec_edgar_enabled', v)}
          />
          <Toggle
            label="USAspending.gov contracts"
            description="Monitor government contract awards — sudden awards signal business momentum or regulatory attention. Free API, no key needed."
            checked={draft.usaspending_enabled}
            onChange={(v) => set('usaspending_enabled', v)}
          />
          <Toggle
            label="State Secretary of State filings"
            description="Track new business formations near watched entities. Requires custom regional integration — see docs."
            checked={draft.state_sos_enabled}
            onChange={(v) => set('state_sos_enabled', v)}
          />
          <Toggle
            label="County recorder / building permits"
            description="Monitor commercial development signals. Requires custom regional integration — see docs."
            checked={draft.county_recorder_enabled}
            onChange={(v) => set('county_recorder_enabled', v)}
          />
          <Toggle
            label="BLS / Census employment data"
            description="Fetch employment and industry trend data for macro context on company and market entities."
            checked={draft.bls_enabled}
            onChange={(v) => set('bls_enabled', v)}
          />
        </div>

        {/* ── Signal Accumulation ── */}
        <SectionHeader>Signal Accumulation</SectionHeader>
        <div className="space-y-1">
          <Toggle
            label="Signal streak tracking"
            description="Track consecutive days with elevated signals — a multi-day streak is far more predictive than a single spike"
            checked={draft.signal_streak_enabled}
            onChange={(v) => set('signal_streak_enabled', v)}
          />
          {draft.signal_streak_enabled && (
            <div className="ml-3 border-l border-cyan-500/20 pl-3 mt-1">
              <NumberInput
                label="Streak alert threshold"
                description="Consecutive elevated-signal days before triggering a 'building pressure' alert"
                value={draft.signal_streak_threshold}
                onChange={(v) => set('signal_streak_threshold', Math.min(14, Math.max(2, v)))}
                min={2}
                max={14}
                unit="days"
              />
            </div>
          )}
          <Toggle
            label="Cross-entity correlation"
            description="Detect macro events when multiple same-type entities signal simultaneously — correlated spikes indicate sector-wide catalysts, not individual noise"
            checked={draft.cross_entity_correlation_enabled}
            onChange={(v) => set('cross_entity_correlation_enabled', v)}
          />
          {draft.cross_entity_correlation_enabled && (
            <div className="ml-3 border-l border-cyan-500/20 pl-3 mt-1">
              <NumberInput
                label="Macro correlation threshold"
                description="Minimum number of same-type entities signalling simultaneously to flag a macro alert"
                value={draft.cross_entity_min_signals}
                onChange={(v) => set('cross_entity_min_signals', Math.min(10, Math.max(2, v)))}
                min={2}
                max={10}
                unit="entities"
              />
            </div>
          )}
        </div>

        {/* ── Notifications ── */}
        <SectionHeader>Notifications & Output</SectionHeader>
        <Toggle
          label="Notify on signals"
          description="Send a push notification when pre-movement pressure signals are detected"
          checked={draft.notify_on_signal}
          onChange={(v) => set('notify_on_signal', v)}
        />
        <Toggle
          label="Notify on every run"
          description="Send a notification even when no signals are found — confirms the task ran"
          checked={draft.notify_all_runs}
          onChange={(v) => set('notify_all_runs', v)}
        />
        <Toggle
          label="Include in morning briefing"
          description="Surface active pre-movement signals in the 7 AM daily briefing"
          checked={draft.include_in_morning_briefing}
          onChange={(v) => set('include_in_morning_briefing', v)}
        />

        {/* ── Advanced (collapsed) ── */}
        <Collapsible title="Advanced">
          <SliderInput
            label="Volume-only threshold"
            description="Minimum volume velocity to flag a signal when all other scoring is disabled"
            value={draft.pressure_velocity_threshold}
            onChange={(v) => set('pressure_velocity_threshold', v)}
            min={1.0}
            max={5.0}
            step={0.1}
            format={(v) => `${v.toFixed(1)}×`}
          />
          <NumberInput
            label="Max articles per entity"
            description="Caps NewsAPI calls per entity per run (free tier: 100 req/day total)"
            value={draft.max_articles_per_topic}
            onChange={(v) => set('max_articles_per_topic', v)}
            min={5}
            max={100}
            unit="articles"
          />
          <TextInput
            label="LLM model"
            description="Ollama model for synthesising the intelligence briefing"
            value={draft.llm_model}
            onChange={(v) => set('llm_model', v)}
            placeholder="qwen3:14b"
          />
          <NumberInput
            label="Run hour"
            description="Hour of day (0–23) to run — default 5 AM, before the 7 AM morning briefing"
            value={draft.schedule_hour}
            onChange={(v) => set('schedule_hour', Math.min(23, Math.max(0, v)))}
            min={0}
            max={23}
            unit="h"
          />
          <TextInput
            label="Geographic focus (simple mode)"
            description="Prepended to simple-mode topic queries when no watchlist is configured"
            value={draft.regions}
            onChange={(v) => set('regions', v)}
            placeholder="US"
          />
          <TextInput
            label="Article language filter"
            description="Comma-separated ISO language codes"
            value={draft.languages}
            onChange={(v) => set('languages', v)}
            placeholder="en"
          />
        </Collapsible>
      </div>

      {/* footer */}
      <div className="flex items-center justify-between gap-3 px-5 py-3 border-t border-cyan-500/20 shrink-0">
        <button onClick={handleReset} disabled={!isDirty}
          className={clsx('flex items-center gap-1.5 px-3 py-1.5 rounded border text-xs transition-all',
            isDirty ? 'border-cyan-500/40 text-cyan-500 hover:border-cyan-500 hover:bg-cyan-500/10'
                    : 'border-cyan-500/15 text-cyan-700 cursor-not-allowed')}>
          <RotateCcw size={12} />
          Revert
        </button>
        <div className="flex items-center gap-2">
          {saved && <span className="text-[11px] text-cyan-400 animate-in fade-in duration-300">✓ Saved</span>}
          {error && <span className="text-[11px] text-red-400">{error}</span>}
          <button onClick={handleSave} disabled={saving || !isDirty}
            className={clsx('flex items-center gap-1.5 px-4 py-1.5 rounded border text-xs font-bold uppercase tracking-wider transition-all',
              isDirty && !saving
                ? 'border-cyan-500 text-cyan-400 bg-cyan-500/10 hover:bg-cyan-500/20 shadow-[0_0_12px_rgba(34,211,238,0.2)]'
                : 'border-cyan-500/15 text-cyan-700 cursor-not-allowed')}>
            {saving ? <Loader size={12} className="animate-spin" /> : <Save size={12} />}
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
