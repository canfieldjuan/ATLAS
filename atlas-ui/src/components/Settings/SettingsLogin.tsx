/**
 * SettingsLogin — token-entry gate for the settings-admin API (#2343 / #2335).
 *
 * The operator pastes the admin token once; it is exchanged for an HttpOnly
 * session cookie via POST /api/v1/settings/session (see settingsApi.ts). The
 * token is never stored client-side — it is discarded after the exchange.
 */
import { useState } from 'react';
import { KeyRound, Loader, AlertCircle } from 'lucide-react';
import { loginSettings } from './settingsApi';

export function SettingsLogin({ onAuthed }: { onAuthed: () => void }) {
  const [token, setToken] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    const result = await loginSettings(token.trim());
    setSubmitting(false);
    if (result === 'ok') {
      setToken(''); // never retain the token
      onAuthed();
      return;
    }
    if (result === 'invalid') setError('Invalid admin token.');
    else if (result === 'unavailable')
      setError('Settings admin is not configured on the server.');
    else setError('Could not reach the settings admin API.');
  };

  return (
    <div className="flex-1 flex items-center justify-center px-6 py-8">
      <form onSubmit={submit} className="w-full max-w-sm space-y-4">
        <div className="flex items-center gap-2 text-cyan-300">
          <KeyRound size={16} />
          <span className="text-sm font-bold uppercase tracking-widest">Unlock settings</span>
        </div>

        <div className="bg-cyan-950/30 border border-cyan-500/20 rounded px-3 py-2 text-xs text-cyan-500/70">
          The settings API is protected. Paste the admin token to start a session — it is
          exchanged for a secure, HttpOnly cookie and is not stored in the browser.
        </div>

        <input
          type="password"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          autoFocus
          placeholder="Settings admin token"
          className="w-full bg-black/30 border border-cyan-500/30 rounded px-3 py-2 text-sm text-cyan-100 placeholder-cyan-800 focus:outline-none focus:border-cyan-500/60"
        />

        {error && (
          <div className="flex items-center gap-1.5 bg-red-900/20 border border-red-500/30 rounded px-3 py-2 text-sm text-red-400">
            <AlertCircle size={14} className="shrink-0" />
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={submitting || !token.trim()}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs font-bold uppercase tracking-wider rounded border border-cyan-500/40 text-cyan-200 hover:bg-cyan-500/10 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
        >
          {submitting ? <Loader size={13} className="animate-spin" /> : <KeyRound size={13} />}
          {submitting ? 'Unlocking…' : 'Unlock'}
        </button>
      </form>
    </div>
  );
}
