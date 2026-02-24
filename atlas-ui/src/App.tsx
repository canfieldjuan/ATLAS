import { useEffect, useState, useRef } from 'react';
import { Avatar } from './components/Avatar/Avatar';
import { useAtlas } from './hooks/useAtlas';
import { useAtlasStore } from './state/store';
import clsx from 'clsx';
import {
  Cpu, Activity, Terminal, Wifi, VolumeX, Volume2, Send, X
} from 'lucide-react';

function App() {
  const { sendText, setPrivacy } = useAtlas();
  const {
    transcript, response, status, isConnected, audioAnalysis,
    privacyMode, setPrivacyMode, textInput, setTextInput,
    media, setMedia, conversationHistory, systemEvents,
  } = useAtlasStore();
  const inputRef = useRef<HTMLInputElement>(null);
  const [showLeftPanel, setShowLeftPanel] = useState(false);
  const [showRightPanel, setShowRightPanel] = useState(false);
  const [cpuLoad, setCpuLoad] = useState(0);
  const [networkSpeed, setNetworkSpeed] = useState(0);
  const [memPercent, setMemPercent] = useState(0);

  // Auto-scale network display: KB/s when under 1 Mb/s, Mb/s otherwise
  const netDisplay = networkSpeed < 1
    ? { value: (networkSpeed * 1000).toFixed(0), unit: 'KB/s' }
    : { value: networkSpeed.toFixed(1), unit: 'Mb/s' };
  const logEndRef = useRef<HTMLDivElement>(null);
  const historyEndRef = useRef<HTMLDivElement>(null);

  const togglePrivacy = () => {
    const next = !privacyMode;
    setPrivacyMode(next);
    setPrivacy(next);
  };

  const handleSendText = () => {
    const text = textInput.trim();
    if (!text) return;
    sendText(text);
    setTextInput('');
    inputRef.current?.blur();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendText();
    }
  };

  // Real system stats — polled from Atlas Brain every 2s
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch('/api/v1/system/stats');
        if (res.ok) {
          const data = await res.json();
          setCpuLoad(data.cpu_percent ?? 0);
          setNetworkSpeed(data.net_mbps ?? 0);
          setMemPercent(data.mem_percent ?? 0);
        }
      } catch {
        // backend unreachable — leave last values
      }
    };

    fetchStats(); // immediate first read
    const interval = setInterval(fetchStats, 2000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll system events
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [systemEvents]);

  // Auto-scroll conversation history
  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversationHistory, transcript, response]);


  return (
    <div className="fixed inset-0 bg-[#020617] text-cyan-400 font-mono overflow-hidden flex flex-col p-4 select-none">

      {/* --- DEEP SPACE 3D BACKGROUND --- */}

      {/* Layer 0: Deep Space Stars (Far Background) */}
      <div className="fixed inset-0 pointer-events-none">
        {[...Array(100)].map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full bg-white"
            style={{
              width: `${Math.random() * 2}px`,
              height: `${Math.random() * 2}px`,
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
              opacity: Math.random() * 0.5,
              animation: `twinkle ${2 + Math.random() * 3}s ease-in-out infinite`,
              animationDelay: `${Math.random() * 3}s`
            }}
          />
        ))}
      </div>

      {/* Layer 1: Distant Nebula Clouds */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-[-20%] left-[-20%] w-[70%] h-[70%] rounded-full bg-cyan-950/30 blur-[150px] animate-[pulse_8s_ease-in-out_infinite]" />
        <div className="absolute bottom-[-20%] right-[-20%] w-[60%] h-[60%] rounded-full bg-purple-900/20 blur-[120px] animate-[pulse_12s_ease-in-out_infinite_reverse]" />
        <div className="absolute top-[30%] right-[-10%] w-[40%] h-[40%] rounded-full bg-blue-900/15 blur-[100px] animate-[pulse_10s_ease-in-out_infinite]" />
      </div>

      {/* Layer 3: Orbital Rings */}
      <div className="fixed inset-0 pointer-events-none flex items-center justify-center">
        <div className="absolute w-[120%] h-[120%] rounded-full border border-cyan-400/10 animate-[spin_60s_linear_infinite]" style={{ transform: 'rotateX(70deg)' }} />
        <div className="absolute w-[100%] h-[100%] rounded-full border border-cyan-500/5 animate-[spin_45s_linear_infinite_reverse]" style={{ transform: 'rotateX(70deg) rotateY(20deg)' }} />
        <div className="absolute w-[80%] h-[80%] rounded-full border-t border-cyan-400/8 animate-[spin_30s_linear_infinite]" style={{ transform: 'rotateX(70deg) rotateY(-20deg)' }} />
      </div>

      {/* Layer 4: Floating Particles (Close) */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden opacity-60">
        {[...Array(30)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400 rounded-full animate-[float-up_var(--duration)_linear_infinite]"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              ['--duration' as any]: `${8 + Math.random() * 15}s`,
              opacity: Math.random() * 0.6,
              boxShadow: '0 0 4px rgba(34,211,238,0.5)'
            }}
          />
        ))}
      </div>

      {/* Layer 6: Scanning Grid Overlay */}
      <div className="fixed inset-0 pointer-events-none z-50">
        <div className="w-full h-[2px] bg-cyan-400/20 shadow-[0_0_20px_rgba(34,211,238,0.6)] animate-[scan_6s_linear_infinite] top-0 absolute" />
      </div>

      {/* --- END BACKGROUND LAYERS --- */}

      {/* Header Bar */}
      <header className="flex justify-between items-center border-b border-cyan-500/10 pb-4 mb-4 relative z-20 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className={clsx(
            "w-3 h-3 rounded-full transition-all duration-300",
            isConnected
              ? "bg-cyan-400 shadow-[0_0_10px_3px_rgba(34,211,238,0.6)] animate-pulse"
              : "bg-red-500 shadow-[0_0_10px_3px_rgba(239,68,68,0.5)]"
          )} />
          <div>
            <h1 className="text-lg font-bold tracking-widest uppercase">ATLAS</h1>
            <p className="text-[9px] text-cyan-600">NEURAL CORE - v0.2.0</p>
          </div>
        </div>

        <div className="flex gap-6 text-[11px]">
          <div className="flex gap-3 items-center mr-4">
            <button
              onClick={togglePrivacy}
              className="p-1.5 rounded border border-cyan-500/30 hover:border-cyan-500 hover:bg-cyan-500/10 transition-all"
              title={privacyMode ? "Privacy Mode ON (text only)" : "Privacy Mode OFF (voice + text)"}
            >
              {privacyMode
                ? <VolumeX size={14} className="text-amber-400" />
                : <Volume2 size={14} className="text-cyan-700" />
              }
            </button>
            <button
              onClick={() => setShowLeftPanel(!showLeftPanel)}
              className="p-1.5 rounded border border-cyan-500/30 hover:border-cyan-500 hover:bg-cyan-500/10 transition-all"
              title="Toggle System Stats"
            >
              <Cpu size={14} className={showLeftPanel ? "text-cyan-400" : "text-cyan-700"} />
            </button>
            <button
              onClick={() => setShowRightPanel(!showRightPanel)}
              className="p-1.5 rounded border border-cyan-500/30 hover:border-cyan-500 hover:bg-cyan-500/10 transition-all"
              title="Toggle System Logs"
            >
              <Terminal size={14} className={showRightPanel ? "text-cyan-400" : "text-cyan-700"} />
            </button>
          </div>
          <div className="flex flex-col items-end">
            <span className="text-cyan-600 uppercase text-[9px]">Network</span>
            <div className="flex gap-1 items-center">
              <span className="text-base font-bold tabular-nums">{netDisplay.value} {netDisplay.unit}</span>
              <Wifi size={12} />
            </div>
          </div>
          <div className="flex flex-col items-end">
            <span className="text-cyan-600 uppercase text-[9px]">System Load</span>
            <div className="flex gap-1 items-center">
              <span className="text-base font-bold tabular-nums w-[3ch] text-right inline-block">{cpuLoad.toFixed(0)}%</span>
              <Activity size={12} />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Grid */}
      <main className={clsx(
        "flex-1 grid gap-4 relative z-10 transition-all duration-300",
        showLeftPanel ? "grid-cols-[theme(spacing.72)_1fr]" : "grid-cols-1"
      )}>

        {/* Left Panel: System Stats */}
        {showLeftPanel && (
          <section className="flex flex-col gap-4 min-h-0 animate-in fade-in slide-in-from-left duration-300">
            <div className="relative bg-black/15 p-4 rounded-sm flex-1 flex flex-col backdrop-blur-md overflow-hidden">
              <div className="flex items-center gap-2 mb-4 pb-2 border-b border-cyan-400/20">
                <Cpu size={16} className="text-cyan-400" />
                <span className="text-sm uppercase tracking-wider font-bold text-cyan-300">System</span>
              </div>
              <div className="flex flex-col gap-5 relative z-20">

                {/* CPU */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-[10px] font-mono">
                    <span className="text-cyan-600 uppercase tracking-wider">CPU Load</span>
                    <span className="text-cyan-300 font-bold tabular-nums">{cpuLoad.toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-cyan-900/40 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: `${cpuLoad}%`,
                        background: cpuLoad > 80 ? '#f87171' : cpuLoad > 60 ? '#fbbf24' : '#22d3ee',
                        boxShadow: `0 0 6px ${cpuLoad > 80 ? '#f87171' : cpuLoad > 60 ? '#fbbf24' : '#22d3ee'}88`,
                      }}
                    />
                  </div>
                </div>

                {/* Memory */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-[10px] font-mono">
                    <span className="text-cyan-600 uppercase tracking-wider">Memory</span>
                    <span className="text-cyan-300 font-bold tabular-nums">{memPercent.toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-cyan-900/40 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{
                        width: `${memPercent}%`,
                        background: memPercent > 90 ? '#f87171' : memPercent > 75 ? '#fbbf24' : '#a78bfa',
                        boxShadow: `0 0 6px ${memPercent > 90 ? '#f87171' : memPercent > 75 ? '#fbbf24' : '#a78bfa'}88`,
                      }}
                    />
                  </div>
                </div>

                {/* Network */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between text-[10px] font-mono">
                    <span className="text-cyan-600 uppercase tracking-wider">Network</span>
                    <span className="text-cyan-300 font-bold tabular-nums">{netDisplay.value} {netDisplay.unit}</span>
                  </div>
                  <div className="h-1.5 w-full bg-cyan-900/40 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-green-400 transition-all duration-700"
                      style={{
                        width: `${Math.min(100, networkSpeed * 10)}%`,
                        boxShadow: '0 0 6px #4ade8088',
                      }}
                    />
                  </div>
                </div>

                {/* Connection Status */}
                <div className="mt-2 pt-3 border-t border-cyan-400/10 flex flex-col gap-2">
                  <div className="flex justify-between text-[10px] font-mono">
                    <span className="text-cyan-600 uppercase tracking-wider">Atlas WS</span>
                    <span className={isConnected ? 'text-green-400 font-bold' : 'text-red-400 font-bold'}>
                      {isConnected ? 'CONNECTED' : 'OFFLINE'}
                    </span>
                  </div>
                  <div className="flex justify-between text-[10px] font-mono">
                    <span className="text-cyan-600 uppercase tracking-wider">Status</span>
                    <span className="text-cyan-300 font-bold uppercase">{status}</span>
                  </div>
                </div>

              </div>
            </div>
          </section>
        )}

        {/* Center Panel: Main Orb */}
        <section className="flex flex-col items-center justify-center relative">
          <div className="mb-6 animate-[float_6s_ease-in-out_infinite]">
            <Avatar />
          </div>

          {/* Audio Waveform Visualization */}
          {(status === 'listening' || status === 'speaking' || audioAnalysis.isActive) && (
            <div className="w-full max-w-2xl h-24 flex items-center justify-center gap-1 mb-4">
              {audioAnalysis.waveform.length > 0 ? (
                // Real-time waveform from audio data
                audioAnalysis.waveform.slice(0, 64).map((value, i) => (
                  <div
                    key={i}
                    className={clsx(
                      "flex-1 rounded-full transition-all duration-75",
                      status === 'listening' && "bg-gradient-to-t from-cyan-500 to-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]",
                      status === 'speaking' && "bg-gradient-to-t from-purple-500 to-purple-400 shadow-[0_0_8px_rgba(168,85,247,0.6)]",
                      !status || status === 'processing' && "bg-gradient-to-t from-cyan-500 to-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]"
                    )}
                    style={{
                      height: `${Math.max(8, Math.abs(value) * 80)}%`,
                      opacity: 0.6 + Math.abs(value) * 0.4
                    }}
                  />
                ))
              ) : (
                // Animated bars when no waveform data
                [...Array(64)].map((_, i) => (
                  <div
                    key={i}
                    className={clsx(
                      "flex-1 rounded-full animate-pulse",
                      status === 'listening' && "bg-gradient-to-t from-cyan-500 to-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]",
                      status === 'speaking' && "bg-gradient-to-t from-purple-500 to-purple-400 shadow-[0_0_8px_rgba(168,85,247,0.6)]"
                    )}
                    style={{
                      height: `${20 + Math.sin(i * 0.3 + Date.now() / 200) * 40}%`,
                      animationDelay: `${i * 0.02}s`,
                      animationDuration: '1s'
                    }}
                  />
                ))
              )}
            </div>
          )}

          {/* Conversation History + Current Text */}
          <div className="w-full max-w-xl max-h-64 overflow-y-auto flex flex-col gap-2 custom-scrollbar">
            {/* Recent conversation turns (last 6), latest assistant highlighted */}
            {conversationHistory.slice(-6).map((turn, i, arr) => {
              const isLatestAssistant = turn.role === 'assistant'
                && i === arr.length - 1;
              return (
                <p
                  key={i}
                  className={clsx(
                    "leading-relaxed",
                    turn.role === 'user' && "text-sm text-right italic text-cyan-300/60",
                    turn.role === 'assistant' && !isLatestAssistant && "text-sm text-left text-white/70",
                    isLatestAssistant && "text-base text-left text-white/90",
                  )}
                >
                  {turn.role === 'user' ? `"${turn.text}"` : turn.text}
                </p>
              );
            })}
            {/* Current transcript (pulsing, visible during processing) */}
            {transcript && (
              <p className="text-sm text-cyan-300/80 italic text-right animate-pulse">"{transcript}"</p>
            )}
            <div ref={historyEndRef} />
          </div>

          {/* Media Overlay (camera feed) */}
          {media && (
            <div className="relative w-full max-w-xl mt-4 rounded border border-cyan-500/30 overflow-hidden bg-black/40">
              <button
                onClick={() => setMedia(null)}
                className="absolute top-2 right-2 z-10 p-1 rounded bg-black/60 hover:bg-black/80 transition-colors"
                title="Dismiss"
              >
                <X size={14} className="text-white/80" />
              </button>
              {media.label && (
                <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded bg-black/60 text-[10px] text-cyan-300 uppercase tracking-wider">
                  {media.label}
                </div>
              )}
              <img
                src={media.url}
                alt={media.label || 'Camera feed'}
                className="w-full h-auto max-h-64 object-contain"
              />
            </div>
          )}

          {/* Status Text */}
          <p className="mt-6 text-cyan-600 text-[9px] font-mono tracking-wider uppercase">
            {status === 'idle' && "Standing By"}
            {status === 'listening' && 'Listening...'}
            {status === 'processing' && "Processing..."}
            {status === 'speaking' && "Speaking..."}
            {status === 'reading' && "Reading..."}
          </p>
        </section>

      </main>

      {/* Right Panel: System Feed — fixed overlay, orb unaffected */}
      {showRightPanel && (
        <div className="fixed top-0 right-0 h-full w-72 z-50 p-4 flex flex-col animate-in fade-in slide-in-from-right duration-300">
          <div className="relative bg-black/60 p-4 rounded-sm flex-1 min-h-0 flex flex-col backdrop-blur-xl border border-cyan-400/15 overflow-hidden">
            <div className="flex items-center justify-between mb-4 pb-2 border-b border-cyan-400/20">
              <div className="flex items-center gap-2">
                <Terminal size={16} className="text-cyan-400" />
                <span className="text-sm uppercase tracking-wider font-bold text-cyan-300">System Feed</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
                  <span className="text-[10px] text-cyan-400 font-bold tracking-wider">LIVE</span>
                </div>
                <button
                  onClick={() => setShowRightPanel(false)}
                  className="p-0.5 rounded hover:bg-cyan-500/10 transition-colors"
                  title="Close"
                >
                  <X size={12} className="text-cyan-600 hover:text-cyan-400" />
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar scroll-smooth">
              {systemEvents.length === 0 ? (
                <div className="text-[10px] text-cyan-700 italic font-mono mt-2">Waiting for system events...</div>
              ) : (
                systemEvents.map((evt) => {
                  const hhmm = new Date(evt.ts).toLocaleTimeString([], {hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'});
                  const catColor: Record<string, string> = {
                    ha: 'text-cyan-400',
                    alert: 'text-yellow-400',
                    reminder: 'text-purple-400',
                    task: 'text-green-400',
                    llm: 'text-blue-400',
                    comms: 'text-orange-400',
                  };
                  const levelColor = evt.level === 'error' ? 'text-red-400' : evt.level === 'warning' ? 'text-yellow-400' : '';
                  return (
                    <div key={evt.id} className="text-[10px] leading-tight flex gap-2 font-mono">
                      <span className="text-cyan-600 shrink-0 font-bold opacity-70">[{hhmm}]</span>
                      <span className={`shrink-0 uppercase font-bold ${catColor[evt.category] || 'text-cyan-400'}`}>{evt.category}</span>
                      <span className={`${levelColor || 'text-cyan-400'} break-all`}>{evt.message}</span>
                    </div>
                  );
                })
              )}
              <div ref={logEndRef} />
            </div>
          </div>
        </div>
      )}

      {/* Footer: Text Input */}
      <footer className="mt-4 flex gap-2 z-10 relative">
        <input
          ref={inputRef}
          type="text"
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={privacyMode ? "Type a message..." : "Type or speak..."}
          disabled={!isConnected}
          className={clsx(
            "flex-1 bg-black/30 backdrop-blur-sm border rounded px-3 py-2 text-sm text-cyan-200 placeholder-cyan-700/50",
            "outline-none transition-all",
            "focus:border-cyan-500/60 focus:bg-black/40",
            isConnected ? "border-cyan-500/30" : "border-red-500/30 opacity-50",
          )}
        />
        <button
          onClick={handleSendText}
          disabled={!isConnected || !textInput.trim()}
          className={clsx(
            "p-2 rounded border transition-all",
            isConnected && textInput.trim()
              ? "border-cyan-500/50 hover:border-cyan-400 hover:bg-cyan-500/10 text-cyan-400"
              : "border-cyan-500/20 text-cyan-700/40 cursor-not-allowed",
          )}
          title="Send message"
        >
          <Send size={16} />
        </button>
      </footer>

      <style dangerouslySetInnerHTML={{ __html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(8, 145, 178, 0.05);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(34, 211, 238, 0.3);
          border-radius: 10px;
        }
        @keyframes float-up {
          from { transform: translateY(100vh); }
          to { transform: translateY(-100px); }
        }
        @keyframes scroll-down {
          from { transform: translateY(-50%); }
          to { transform: translateY(0); }
        }
        @keyframes scan {
          0% { transform: translateY(-10vh); }
          100% { transform: translateY(110vh); }
        }
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        @keyframes twinkle {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 1; }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px) translateX(0px); }
          25% { transform: translateY(-15px) translateX(10px); }
          50% { transform: translateY(-8px) translateX(-8px); }
          75% { transform: translateY(-20px) translateX(5px); }
        }
      `}} />
    </div>
  );
}

export default App;
