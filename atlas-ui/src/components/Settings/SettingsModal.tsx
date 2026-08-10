/**
 * SettingsModal — tabbed settings panel combining all configurable Atlas systems.
 *
 * Opens as a full modal overlay.  Each tab hosts its own form which manages
 * its own load/save lifecycle independently.
 */
import { useEffect, useState } from 'react';
import { X, Mic, Mail, Brain, Newspaper, Cpu, Bell, Plug, LogOut, Loader, AlertCircle } from 'lucide-react';
import clsx from 'clsx';
import { VoiceSettingsForm } from './VoiceSettings';
import { EmailSettingsForm } from './EmailSettings';
import { DailySettingsForm } from './DailyIntelligenceSettings';
import { IntelligenceSettingsForm } from './NewsIntelligenceSettings';
import { LLMSettingsForm } from './LLMSettings';
import { NotificationSettingsForm } from './NotificationSettings';
import { IntegrationSettingsForm } from './IntegrationSettings';
import { SettingsLogin } from './SettingsLogin';
import { probeSettingsAuth, logoutSettings, type AuthState } from './settingsApi';

type Tab = 'voice' | 'email' | 'daily' | 'intelligence' | 'llm' | 'notifications' | 'integrations';

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: 'voice',          label: 'Voice Pipeline', icon: <Mic size={13} /> },
  { id: 'email',          label: 'Email',          icon: <Mail size={13} /> },
  { id: 'daily',          label: 'Daily Ops',      icon: <Brain size={13} /> },
  { id: 'intelligence',   label: 'Intelligence',   icon: <Newspaper size={13} /> },
  { id: 'llm',            label: 'AI Model',       icon: <Cpu size={13} /> },
  { id: 'notifications',  label: 'Notifications',  icon: <Bell size={13} /> },
  { id: 'integrations',   label: 'Integrations',   icon: <Plug size={13} /> },
];

interface SettingsModalProps {
  onClose: () => void;
  initialTab?: Tab;
}

export function SettingsModal({ onClose, initialTab = 'voice' }: SettingsModalProps) {
  const [activeTab, setActiveTab] = useState<Tab>(initialTab);
  // 'checking' until the initial auth probe resolves.
  const [auth, setAuth] = useState<AuthState | 'checking'>('checking');

  useEffect(() => {
    let active = true;
    probeSettingsAuth().then((state) => active && setAuth(state));
    return () => {
      active = false;
    };
  }, []);

  const handleLogout = async () => {
    await logoutSettings();
    setAuth('need-login');
  };

  return (
    /* backdrop */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      {/* panel */}
      <div className="relative w-full max-w-lg max-h-[90vh] flex flex-col bg-[#020617]/95 border border-cyan-500/30 rounded-sm shadow-[0_0_40px_rgba(34,211,238,0.1)] overflow-hidden">

        {/* header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-cyan-500/20 shrink-0">
          <div>
            <h2 className="text-sm font-bold uppercase tracking-widest text-cyan-300">Settings</h2>
            <p className="text-[10px] text-cyan-600 mt-0.5">
              Changes apply immediately · restart may be required for some settings
            </p>
          </div>
          <div className="flex items-center gap-1.5">
            {auth === 'authed' && (
              <button
                onClick={handleLogout}
                title="Lock settings (clear session)"
                className="flex items-center gap-1 px-2 py-1.5 rounded border border-cyan-500/20 hover:border-cyan-500/60 hover:bg-cyan-500/10 text-[10px] font-bold uppercase tracking-wider text-cyan-500 transition-all"
              >
                <LogOut size={12} /> Lock
              </button>
            )}
            <button
              onClick={onClose}
              className="p-1.5 rounded border border-cyan-500/20 hover:border-cyan-500/60 hover:bg-cyan-500/10 transition-all"
            >
              <X size={14} className="text-cyan-500" />
            </button>
          </div>
        </div>

        {/* tabs — only once authenticated */}
        {auth === 'authed' && (
          <div className="flex border-b border-cyan-500/20 shrink-0 px-5">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={clsx(
                  'flex items-center gap-1.5 px-3 py-2 text-xs font-bold uppercase tracking-wider border-b-2 transition-all -mb-px',
                  activeTab === tab.id
                    ? 'border-cyan-400 text-cyan-300'
                    : 'border-transparent text-cyan-700 hover:text-cyan-500',
                )}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>
        )}

        {/* body — auth gate, then the tab content */}
        <div className="flex-1 min-h-0 flex flex-col">
          {auth === 'checking' && (
            <div className="flex-1 flex items-center justify-center text-cyan-600 text-xs">
              <Loader size={14} className="animate-spin mr-2" /> Checking access…
            </div>
          )}
          {(auth === 'unavailable' || auth === 'error') && (
            <div className="flex-1 flex items-center justify-center px-6">
              <div className="flex items-center gap-1.5 bg-red-900/20 border border-red-500/30 rounded px-3 py-2 text-sm text-red-400">
                <AlertCircle size={14} className="shrink-0" />
                {auth === 'unavailable'
                  ? 'Settings admin is not configured on the server.'
                  : 'Could not reach the settings admin API.'}
              </div>
            </div>
          )}
          {auth === 'need-login' && <SettingsLogin onAuthed={() => setAuth('authed')} />}
          {auth === 'authed' && (
            <>
              {activeTab === 'voice'         && <VoiceSettingsForm />}
              {activeTab === 'email'         && <EmailSettingsForm />}
              {activeTab === 'daily'         && <DailySettingsForm />}
              {activeTab === 'intelligence'  && <IntelligenceSettingsForm />}
              {activeTab === 'llm'           && <LLMSettingsForm />}
              {activeTab === 'notifications' && <NotificationSettingsForm />}
              {activeTab === 'integrations'  && <IntegrationSettingsForm />}
            </>
          )}
        </div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 3px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(8,145,178,0.05); }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(34,211,238,0.3); border-radius: 10px; }
        input[type=range] { height: 4px; }
        select option { background: #020617; }
      `}</style>
    </div>
  );
}
