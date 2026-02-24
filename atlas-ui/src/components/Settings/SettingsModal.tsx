/**
 * SettingsModal — tabbed settings panel combining Voice Pipeline, Email, and Daily Ops.
 *
 * Opens as a full modal overlay.  Each tab hosts its own form which manages
 * its own load/save lifecycle independently.
 */
import { useState } from 'react';
import { X, Mic, Mail, Brain } from 'lucide-react';
import clsx from 'clsx';
import { VoiceSettingsForm } from './VoiceSettings';
import { EmailSettingsForm } from './EmailSettings';
import { DailySettingsForm } from './DailyIntelligenceSettings';

type Tab = 'voice' | 'email' | 'daily';

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: 'voice', label: 'Voice Pipeline', icon: <Mic size={13} /> },
  { id: 'email', label: 'Email', icon: <Mail size={13} /> },
  { id: 'daily', label: 'Daily Ops', icon: <Brain size={13} /> },
];

interface SettingsModalProps {
  onClose: () => void;
  initialTab?: Tab;
}

export function SettingsModal({ onClose, initialTab = 'voice' }: SettingsModalProps) {
  const [activeTab, setActiveTab] = useState<Tab>(initialTab);

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
          <button
            onClick={onClose}
            className="p-1.5 rounded border border-cyan-500/20 hover:border-cyan-500/60 hover:bg-cyan-500/10 transition-all"
          >
            <X size={14} className="text-cyan-500" />
          </button>
        </div>

        {/* tabs */}
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

        {/* tab content — flex-1 so the form footer sticks to the bottom */}
        <div className="flex-1 min-h-0 flex flex-col">
          {activeTab === 'voice' && <VoiceSettingsForm />}
          {activeTab === 'email' && <EmailSettingsForm />}
          {activeTab === 'daily' && <DailySettingsForm />}
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
