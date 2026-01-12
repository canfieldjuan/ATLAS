export type AtlasState = 'idle' | 'listening' | 'processing' | 'speaking' | 'error';

export interface AtlasMessage {
  type: string;
  payload?: any;
}

export interface AudioAnalysis {
  volume: number;       // 0-100
  frequency: number;    // Average frequency
  isActive: boolean;    // Whether audio is playing
  waveform: number[];   // Time-domain waveform data (128 samples)
}

export interface AtlasStore {
  status: AtlasState;
  transcript: string;
  response: string;
  isConnected: boolean;
  audioAnalysis: AudioAnalysis;
  setStatus: (status: AtlasState) => void;
  setTranscript: (text: string) => void;
  setResponse: (text: string) => void;
  setConnected: (connected: boolean) => void;
  setAudioAnalysis: (analysis: AudioAnalysis) => void;
  reset: () => void;
}
