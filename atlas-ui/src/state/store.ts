import { create } from 'zustand';
import type { AtlasStore } from '../types/index';

export const useAtlasStore = create<AtlasStore>((set) => ({
  status: 'idle',
  transcript: '',
  response: '',
  isConnected: false,
  audioAnalysis: { volume: 0, frequency: 0, isActive: false, waveform: [] },
  
  setStatus: (status) => set({ status }),
  setTranscript: (transcript) => set({ transcript }),
  setResponse: (response) => set({ response }),
  setConnected: (isConnected) => set({ isConnected }),
  setAudioAnalysis: (audioAnalysis) => set({ audioAnalysis }),
  
  reset: () => set({ 
    status: 'idle', 
    transcript: '', 
    response: '',
    audioAnalysis: { volume: 0, frequency: 0, isActive: false, waveform: [] }
  })
}));
