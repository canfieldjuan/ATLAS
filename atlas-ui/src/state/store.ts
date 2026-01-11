import { create } from 'zustand';
import type { AtlasStore } from '../types/index';

export const useAtlasStore = create<AtlasStore>((set) => ({
  status: 'idle',
  transcript: '',
  response: '',
  isConnected: false,
  
  setStatus: (status) => set({ status }),
  setTranscript: (transcript) => set({ transcript }),
  setResponse: (response) => set({ response }),
  setConnected: (isConnected) => set({ isConnected }),
  
  reset: () => set({ 
    status: 'idle', 
    transcript: '', 
    response: '' 
  })
}));
