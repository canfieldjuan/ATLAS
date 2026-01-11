export type AtlasState = 'idle' | 'listening' | 'processing' | 'speaking' | 'error';

export interface AtlasMessage {
  type: string;
  payload?: any;
}

export interface AtlasStore {
  status: AtlasState;
  transcript: string;
  response: string;
  isConnected: boolean;
  setStatus: (status: AtlasState) => void;
  setTranscript: (text: string) => void;
  setResponse: (text: string) => void;
  setConnected: (connected: boolean) => void;
  reset: () => void;
}
