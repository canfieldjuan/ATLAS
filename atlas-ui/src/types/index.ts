export type AtlasState = 'idle' | 'listening' | 'processing' | 'speaking' | 'reading' | 'error';

export interface AtlasMessage {
  type: string;
  payload?: unknown;
}

export interface AudioAnalysis {
  volume: number;       // 0-100
  frequency: number;    // Average frequency
  isActive: boolean;    // Whether audio is playing
  waveform: number[];   // Time-domain waveform data (128 samples)
}

export interface MediaAttachment {
  type: 'mjpeg' | 'image';
  url: string;
  label?: string;
}

export interface ConversationTurn {
  role: 'user' | 'assistant';
  text: string;
}

export interface SystemEvent {
  id: string;
  ts: string;
  category: 'ha' | 'alert' | 'reminder' | 'task' | 'llm' | 'comms';
  level: 'info' | 'warning' | 'error';
  message: string;
}

export interface AtlasStore {
  status: AtlasState;
  transcript: string;
  response: string;
  isConnected: boolean;
  audioAnalysis: AudioAnalysis;
  privacyMode: boolean;
  textInput: string;
  media: MediaAttachment | null;
  conversationHistory: ConversationTurn[];
  systemEvents: SystemEvent[];
  setStatus: (status: AtlasState) => void;
  setTranscript: (text: string) => void;
  setResponse: (text: string) => void;
  setConnected: (connected: boolean) => void;
  setAudioAnalysis: (analysis: AudioAnalysis) => void;
  setPrivacyMode: (enabled: boolean) => void;
  setTextInput: (text: string) => void;
  setMedia: (media: MediaAttachment | null) => void;
  addConversationTurn: (role: 'user' | 'assistant', text: string) => void;
  addSystemEvent: (event: SystemEvent) => void;
  reset: () => void;
}
