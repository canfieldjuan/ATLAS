import { useEffect, useState, useRef } from 'react';
import { Avatar } from './components/Avatar/Avatar';
import { useAtlas } from './hooks/useAtlas';
import { useAtlasStore } from './state/store';
import { useAudioRecorder } from './hooks/useAudioRecorder';
import clsx from 'clsx';
import { 
  Mic, MicOff, Radio, Cpu, Activity, Terminal, Wifi,
  Settings, Shield, Volume2, Search, BarChart3
} from 'lucide-react';

// Generate static random heights outside component to avoid impure function calls
const IDLE_BAR_HEIGHTS = Array.from({ length: 24 }, () => 20 + Math.random() * 30);
const IDLE_WAVE_HEIGHTS = Array.from({ length: 16 }, () => 12 + Math.random() * 20);

function App() {
  const { sendAudioData, stopRecording: sendStopCommand } = useAtlas();
  const { transcript, response, status, isConnected, audioAnalysis } = useAtlasStore();
  const [alwaysOn, setAlwaysOn] = useState(true);
  const [showLeftPanel, setShowLeftPanel] = useState(false);
  const [showRightPanel, setShowRightPanel] = useState(false);
  const [logs, setLogs] = useState([
    "System initialized.",
    "Neural link established.",
    "Audio subsystem ready.",
    "Awaiting voice commands."
  ]);
  const [cpuLoad, setCpuLoad] = useState(42);
  const [memoryLoad, setMemoryLoad] = useState(38);
  const [networkSpeed, setNetworkSpeed] = useState(420);
  const [cpuTemp, setCpuTemp] = useState(42.1);
  const logEndRef = useRef<HTMLDivElement>(null);

  const { isRecording, startRecording, stopRecording, error } = useAudioRecorder(
    (audioData) => {
      sendAudioData(audioData);
    }
  );

  // Auto-start recording when always-on mode is enabled and connected
  useEffect(() => {
    if (alwaysOn && isConnected && !isRecording) {
      startRecording();
    }
    if (!alwaysOn && isRecording) {
      stopRecording();
    }
  }, [alwaysOn, isConnected, isRecording, startRecording, stopRecording]);

  // Simulated system stats and logs
  useEffect(() => {
    const interval = setInterval(() => {
      setCpuLoad(c => Math.max(20, Math.min(85, c + (Math.random() - 0.5) * 5)));
      setMemoryLoad(m => Math.max(25, Math.min(75, m + (Math.random() - 0.5) * 3)));
      setNetworkSpeed(() => Math.floor(400 + Math.random() * 50));
      setCpuTemp(t => Math.max(38, Math.min(55, t + (Math.random() - 0.5) * 0.5)));

      // Add logs based on status changes
      if (Math.random() > 0.92) {
        const statusLogs = [
          "Analyzing audio stream...",
          "Processing neural patterns...",
          "Voice recognition active.",
          "Network uplink stable.",
          "Encryption keys rotated."
        ];
        const newLog = statusLogs[Math.floor(Math.random() * statusLogs.length)];
        setLogs(prev => [...prev.slice(-15), newLog]);
      }
    }, 1200);

    return () => clearInterval(interval);
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const handleMicClick = () => {
    if (alwaysOn) {
      // Toggle always-on mode off
      setAlwaysOn(false);
      stopRecording();
      sendStopCommand();
    } else if (isRecording) {
      stopRecording();
      sendStopCommand();
    } else {
      startRecording();
    }
  };

  const toggleAlwaysOn = () => {
    setAlwaysOn(!alwaysOn);
  };

  return (
    <div className="min-h-screen w-full bg-[#020617] text-cyan-400 font-mono overflow-hidden flex flex-col p-4 select-none relative">
      
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
        <div className="absolute w-[120%] h-[120%] rounded-full border border-cyan-400/10 animate-[spin_60s_linear_infinite,float_8s_ease-in-out_infinite]" style={{ transform: 'rotateX(70deg)' }} />
        <div className="absolute w-[100%] h-[100%] rounded-full border border-cyan-500/5 animate-[spin_45s_linear_infinite_reverse,float_10s_ease-in-out_infinite]" style={{ transform: 'rotateX(70deg) rotateY(20deg)' }} />
        <div className="absolute w-[80%] h-[80%] rounded-full border-t border-cyan-400/8 animate-[spin_30s_linear_infinite,float_12s_ease-in-out_infinite]" style={{ transform: 'rotateX(70deg) rotateY(-20deg)' }} />
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
              <span className="text-base font-bold">{networkSpeed} Mb/s</span>
              <Wifi size={12} />
            </div>
          </div>
          <div className="flex flex-col items-end">
            <span className="text-cyan-600 uppercase text-[9px]">System Load</span>
            <div className="flex gap-1 items-center">
              <span className="text-base font-bold">{cpuLoad.toFixed(0)}%</span>
              <Activity size={12} />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Grid */}
      <main className={clsx(
        "flex-1 grid gap-4 relative z-10 transition-all duration-300",
        showLeftPanel && showRightPanel && "grid-cols-12",
        showLeftPanel && !showRightPanel && "grid-cols-9",
        !showLeftPanel && showRightPanel && "grid-cols-9",
        !showLeftPanel && !showRightPanel && "grid-cols-1"
      )}>
        
        {/* Left Panel: System Stats */}
        {showLeftPanel && (
          <section className="col-span-3 flex flex-col gap-4 min-h-0 animate-in fade-in slide-in-from-left duration-300">
        </section>
        )}

        {/* Center Panel: Main Orb */}
        <section className={clsx(
          "flex flex-col items-center justify-center relative transition-all duration-300",
          showLeftPanel && showRightPanel && "col-span-6",
          showLeftPanel && !showRightPanel && "col-span-6",
          !showLeftPanel && showRightPanel && "col-span-6",
          !showLeftPanel && !showRightPanel && "col-span-1"
        )}>
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

          {/* Text Display */}
          <div className="w-full max-w-xl max-h-32 overflow-y-auto flex flex-col gap-3 text-center">
            {transcript && (
              <p className="text-sm text-cyan-300/80 italic animate-pulse">"{transcript}"</p>
            )}
            {response && (
              <p className="text-base text-white/90 leading-relaxed">{response}</p>
            )}
          </div>

          {/* Voice Input Indicator */}
          <div className="mt-8 flex flex-col items-center gap-4">
            <div className="flex gap-1 h-8 items-center">
              {[...Array(16)].map((_, i) => (
                <div 
                  key={i} 
                  className={clsx(
                    "w-1 bg-cyan-400 rounded-full transition-all duration-300",
                    isRecording ? "opacity-100 animate-bounce" : "opacity-20 h-1"
                  )}
                  style={{ 
                    height: isRecording ? `${IDLE_WAVE_HEIGHTS[i]}px` : '4px',
                    animationDelay: `${i * 0.05}s`
                  }}
                />
              ))}
            </div>

            <p className="text-cyan-600 text-[9px] font-mono tracking-wider uppercase">
              {status === 'idle' && "Standing By"}
              {status === 'listening' && 'Listening...'}
              {status === 'processing' && "Processing..."}
              {status === 'speaking' && "Speaking..."}
            </p>
          </div>
        </section>

        {/* Right Panel: Live Logs */}
        {showRightPanel && (
          <section className="col-span-3 flex flex-col gap-4 min-h-0 animate-in fade-in slide-in-from-right duration-300">
          <div className="relative bg-black/15 p-4 rounded-sm flex-1 flex flex-col backdrop-blur-md overflow-hidden group">
            <div className="flex items-center justify-between mb-4 pb-2 relative z-20 border-b border-cyan-400/20">
              <div className="flex items-center gap-2">
                <Terminal size={16} className="text-cyan-400" />
                <span className="text-sm uppercase tracking-wider font-bold text-cyan-300">System Feed</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-cyan-400 rounded-full shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
                <span className="text-[10px] text-cyan-400 font-bold tracking-wider">LIVE</span>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto space-y-2 pr-2 custom-scrollbar scroll-smooth relative z-20">
              {logs.map((log, i) => (
                <div key={i} className="text-[10px] leading-tight flex gap-2 font-mono">
                  <span className="text-cyan-600 shrink-0 font-bold opacity-70">
                    [{new Date().toLocaleTimeString([], {hour12: false, minute: '2-digit', second: '2-digit'})}]
                  </span>
                  <span className={log.startsWith(">>>") ? "text-cyan-300 font-bold" : "text-cyan-400"}>
                    {log}
                  </span>
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>

          {error && (
            <div className="relative bg-black/20 p-3 rounded-sm backdrop-blur-md border border-red-900/30 overflow-hidden">
              <div className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
              <div className="text-[10px] text-red-400 relative z-10 font-bold font-mono">{error}</div>
            </div>
          )}
        </section>
        )}
      </main>

      {/* Footer Interface */}
      <footer className="mt-4 flex gap-4 z-10">
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
