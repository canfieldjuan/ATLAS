/**
 * IntegrationSettings — Home Assistant, MQTT broker, and MCP server toggles.
 */
import { useState, useEffect, useCallback } from 'react';
import { Save, RefreshCw, Plug, AlertCircle } from 'lucide-react';
import clsx from 'clsx';

const API = '/api/v1/settings/integrations';

interface IntegrationSettings {
  ha_enabled: boolean;
  ha_url: string;
  ha_entity_filter: string;
  ha_websocket_enabled: boolean;
  ha_websocket_reconnect_interval: number;
  ha_state_cache_ttl: number;
  mqtt_enabled: boolean;
  mqtt_host: string;
  mqtt_port: number;
  mqtt_username: string;
  mcp_crm_enabled: boolean;
  mcp_email_enabled: boolean;
  mcp_calendar_enabled: boolean;
  mcp_twilio_enabled: boolean;
  mcp_transport: string;
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
export function IntegrationSettingsForm() {
  const [form, setForm] = useState<IntegrationSettings | null>(null);
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

  const set = <K extends keyof IntegrationSettings>(k: K, v: IntegrationSettings[K]) =>
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
      setStatus({ ok: true, msg: 'Saved — backend changes require a server restart' });
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

        {/* ── Home Assistant ── */}
        <Section title="Home Assistant" badge="smart home">
          <Toggle
            value={form.ha_enabled}
            onChange={(v) => set('ha_enabled', v)}
            label="Enable Home Assistant backend"
            hint="Connects Atlas to HA for real-time device control and state monitoring"
          />
          {form.ha_enabled && (
            <>
              <div>
                <Label>Home Assistant URL</Label>
                <TextInput value={form.ha_url} onChange={(v) => set('ha_url', v)}
                  placeholder="http://homeassistant.local:8123" mono />
                <p className="text-[10px] text-cyan-700 mt-0.5">
                  Set <span className="font-mono text-cyan-600">ATLAS_HA_TOKEN</span> in your .env file.
                </p>
              </div>
              <div>
                <Label>Entity filter (JSON array of prefixes)</Label>
                <TextInput value={form.ha_entity_filter} onChange={(v) => set('ha_entity_filter', v)}
                  placeholder='["light.","switch.","media_player."]' mono />
                <p className="text-[10px] text-cyan-700 mt-0.5">
                  Only entity types listed here are auto-discovered and registered.
                </p>
              </div>
              <Toggle
                value={form.ha_websocket_enabled}
                onChange={(v) => set('ha_websocket_enabled', v)}
                label="Enable WebSocket for real-time state"
                hint="Recommended — receives live state changes without polling"
              />
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>WS reconnect interval (s)</Label>
                  <NumInput value={form.ha_websocket_reconnect_interval}
                    onChange={(v) => set('ha_websocket_reconnect_interval', v)} min={1} max={60} />
                </div>
                <div>
                  <Label>State cache TTL (s)</Label>
                  <NumInput value={form.ha_state_cache_ttl}
                    onChange={(v) => set('ha_state_cache_ttl', v)} min={10} max={3600} />
                </div>
              </div>
            </>
          )}
        </Section>

        {/* ── MQTT ── */}
        <Section title="MQTT Broker" badge="IoT">
          <Toggle
            value={form.mqtt_enabled}
            onChange={(v) => set('mqtt_enabled', v)}
            label="Enable MQTT backend"
            hint="Direct messaging to ESP32, Zigbee, or other MQTT-capable devices"
          />
          {form.mqtt_enabled && (
            <>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label>Broker host</Label>
                  <TextInput value={form.mqtt_host} onChange={(v) => set('mqtt_host', v)}
                    placeholder="localhost" mono />
                </div>
                <div>
                  <Label>Broker port</Label>
                  <NumInput value={form.mqtt_port} onChange={(v) => set('mqtt_port', v)}
                    min={1} max={65535} />
                </div>
              </div>
              <div>
                <Label>Username (optional)</Label>
                <TextInput value={form.mqtt_username} onChange={(v) => set('mqtt_username', v)}
                  placeholder="leave empty if no auth" mono />
                <p className="text-[10px] text-cyan-700 mt-0.5">
                  Set <span className="font-mono text-cyan-600">ATLAS_MQTT_PASSWORD</span> in your .env file.
                </p>
              </div>
            </>
          )}
        </Section>

        {/* ── MCP Servers ── */}
        <Section title="MCP Tool Servers" badge="Claude / Cursor">
          <p className="text-[10px] text-cyan-700 -mt-1 mb-2">
            Each MCP server exposes Atlas tools to AI clients (Claude Desktop, Cursor). Disable
            servers you don't need to reduce startup overhead.
          </p>
          <Toggle
            value={form.mcp_crm_enabled}
            onChange={(v) => set('mcp_crm_enabled', v)}
            label="CRM server — contacts, interactions, appointments"
          />
          <Toggle
            value={form.mcp_email_enabled}
            onChange={(v) => set('mcp_email_enabled', v)}
            label="Email server — send, read, search, draft"
          />
          <Toggle
            value={form.mcp_calendar_enabled}
            onChange={(v) => set('mcp_calendar_enabled', v)}
            label="Calendar server — events, free slots, sync"
          />
          <Toggle
            value={form.mcp_twilio_enabled}
            onChange={(v) => set('mcp_twilio_enabled', v)}
            label="Twilio server — calls, SMS, recordings"
          />

          <div className="pt-1">
            <Label>Transport mode</Label>
            <div className="flex gap-2">
              {(['stdio', 'sse'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => set('mcp_transport', mode)}
                  className={clsx(
                    'px-3 py-1.5 rounded-sm border text-[10px] font-semibold uppercase tracking-wider transition-all',
                    form.mcp_transport === mode
                      ? 'border-cyan-400 bg-cyan-500/10 text-cyan-300'
                      : 'border-cyan-500/20 text-cyan-700 hover:border-cyan-500/40 hover:text-cyan-500',
                  )}
                >
                  {mode === 'stdio' ? 'stdio (Claude Desktop / Cursor)' : 'sse (HTTP endpoint)'}
                </button>
              ))}
            </div>
          </div>
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
            <Plug size={11} /> Reset
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
