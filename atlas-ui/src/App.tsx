import { Avatar } from './components/Avatar/Avatar';
import { useAtlas } from './hooks/useAtlas';
import { useAtlasStore } from './state/store';
import { useAudioRecorder } from './hooks/useAudioRecorder';
import clsx from 'clsx';
import { Mic, MicOff } from 'lucide-react';

function App() {
  const { sendAudioData, stopRecording: sendStopCommand } = useAtlas();
  const { transcript, response, status, isConnected } = useAtlasStore();

  const { isRecording, startRecording, stopRecording, error } = useAudioRecorder(
    (audioData) => {
      sendAudioData(audioData);
    }
  );

  const handleMicClick = () => {
    if (isRecording) {
      stopRecording();
      sendStopCommand();
    } else {
      startRecording();
    }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-gray-950 via-black to-gray-950 flex flex-col items-center justify-center p-8 overflow-hidden relative">
      {/* Background effects */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-purple-900/20 via-transparent to-transparent pointer-events-none" />
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMSIgY3k9IjEiIHI9IjEiIGZpbGw9InJnYmEoMjU1LDI1NSwyNTUsMC4wMykiLz48L3N2Zz4=')] pointer-events-none opacity-50" />

      {/* Header */}
      <div className="absolute top-8 left-8 flex items-center gap-3">
        <div className={clsx(
          "w-3 h-3 rounded-full transition-all duration-300",
          isConnected
            ? "bg-emerald-400 shadow-[0_0_10px_3px_rgba(52,211,153,0.5)]"
            : "bg-red-500 shadow-[0_0_10px_3px_rgba(239,68,68,0.5)]"
        )} />
        <span className="text-sm font-mono text-gray-400 tracking-wider">ATLAS SYSTEM</span>
        <span className="text-xs font-mono text-gray-600">v0.2.0</span>
      </div>

      {/* Main Orb */}
      <div className="mb-8">
        <Avatar />
      </div>

      {/* Text Display */}
      <div className="w-full max-w-2xl max-h-[30vh] overflow-y-auto flex flex-col gap-4 text-center z-10 mb-4 px-4">
        <div className={clsx(
          "transition-all duration-500",
          status === 'listening' ? "opacity-100" : "opacity-50"
        )}>
          {transcript && (
            <p className="text-xl font-light text-blue-300/80 italic">"{transcript}"</p>
          )}
        </div>
        <div className={clsx(
          "transition-all duration-500",
          (status === 'speaking' || response) ? "opacity-100" : "opacity-0"
        )}>
          {response && (
            <p className="text-lg font-medium text-white/90 leading-relaxed">{response}</p>
          )}
        </div>
      </div>

      {/* Mic Button */}
      <div className="absolute bottom-20 w-full flex flex-col items-center gap-3 z-20">
        {error && (
          <div className="text-red-400 text-sm bg-red-950/50 px-4 py-2 rounded-full">{error}</div>
        )}
        <button
          onClick={handleMicClick}
          disabled={!isConnected}
          className={clsx(
            "w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300",
            "border-2 backdrop-blur-sm",
            isRecording
              ? "bg-red-500/20 border-red-500 shadow-[0_0_30px_10px_rgba(239,68,68,0.3)] scale-110"
              : "bg-white/5 border-white/20 hover:bg-white/10 hover:border-white/40 hover:scale-105",
            !isConnected && "bg-gray-800/50 border-gray-700 cursor-not-allowed opacity-50"
          )}
        >
          {isRecording
            ? <MicOff className="w-7 h-7 text-red-400" />
            : <Mic className="w-7 h-7 text-white/80" />
          }
        </button>
        <p className="text-gray-500 text-xs font-mono tracking-wide">
          {isRecording ? "RECORDING" : "TAP TO SPEAK"}
        </p>
      </div>

      {/* Status Bar */}
      <div className="absolute bottom-6 flex items-center gap-2">
        <div className={clsx(
          "w-2 h-2 rounded-full",
          status === 'idle' && "bg-gray-600",
          status === 'listening' && "bg-blue-400 animate-pulse",
          status === 'processing' && "bg-yellow-400 animate-spin",
          status === 'speaking' && "bg-purple-400 animate-pulse",
          status === 'error' && "bg-red-400"
        )} />
        <span className="font-mono text-xs text-gray-600 tracking-widest">{status.toUpperCase()}</span>
      </div>
    </div>
  );
}

export default App;
