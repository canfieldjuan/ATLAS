import { useEffect, useRef, useCallback } from 'react';
import { useAtlasStore } from '../state/store';
import type { AtlasState } from '../types/index';

export const useAtlas = (url: string = 'ws://localhost:8000/api/v1/ws/orchestrated') => {
  const { setStatus, setTranscript, setResponse, setConnected, setAudioAnalysis } = useAtlasStore();
  const ws = useRef<WebSocket | null>(null);
  const currentAudio = useRef<HTMLAudioElement | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const analyser = useRef<AnalyserNode | null>(null);
  const animationFrame = useRef<number | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 10;
  const baseReconnectDelay = 1000;

  const connect = useCallback(() => {
    // Don't create new connection if one exists and is open or connecting
    if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    // Close any existing connection in closing state
    if (ws.current) {
      ws.current.close();
    }

    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      console.log('Connected to Atlas');
      setConnected(true);
      reconnectAttempts.current = 0;
    };

    ws.current.onclose = () => {
      console.log('Disconnected from Atlas');
      setConnected(false);

      // Auto-reconnect with exponential backoff
      if (reconnectAttempts.current < maxReconnectAttempts) {
        const delay = baseReconnectDelay * Math.pow(2, reconnectAttempts.current);
        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current + 1})`);
        reconnectTimeout.current = setTimeout(() => {
          reconnectAttempts.current++;
          connect();
        }, delay);
      }
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WS Message:', data);
        
        switch (data.state) {
          case 'idle':
            setStatus('idle');
            break;
          case 'listening':
            setStatus('idle'); 
            break;
          case 'wake_detected':
          case 'recording':
            setStatus('listening');
            break;
          case 'transcribing':
          case 'processing':
          case 'executing':
            setStatus('processing');
            break;
          case 'responding':
            setStatus('speaking');
            break;
          case 'error':
            setStatus('error');
            console.error('Atlas Error:', data.message);
            break;
          case 'transcript':
            setTranscript(data.text);
            break;
          case 'response':
            setResponse(data.text);
            setStatus('speaking');

            if (data.audio_base64) {
              try {
                // Stop any currently playing audio
                if (currentAudio.current) {
                  currentAudio.current.pause();
                  currentAudio.current = null;
                }
                if (animationFrame.current) {
                  cancelAnimationFrame(animationFrame.current);
                }

                const binaryString = atob(data.audio_base64);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                  bytes[i] = binaryString.charCodeAt(i);
                }
                const blob = new Blob([bytes], { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(blob);
                const audio = new Audio(audioUrl);
                currentAudio.current = audio;

                // Set up audio analysis
                if (!audioContext.current) {
                  audioContext.current = new AudioContext();
                }
                if (!analyser.current) {
                  analyser.current = audioContext.current.createAnalyser();
                  analyser.current.fftSize = 256;
                }

                const source = audioContext.current.createMediaElementSource(audio);
                source.connect(analyser.current);
                analyser.current.connect(audioContext.current.destination);

                // Start analyzing audio
                const dataArray = new Uint8Array(analyser.current.frequencyBinCount);
                const waveformData = new Uint8Array(128); // 128 samples for waveform
                const analyzeAudio = () => {
                  if (!analyser.current || !currentAudio.current) return;
                  
                  analyser.current.getByteFrequencyData(dataArray);
                  
                  // Get time-domain data for waveform visualization
                  analyser.current.getByteTimeDomainData(waveformData);
                  const waveform = Array.from(waveformData).map(v => (v - 128) / 128); // Normalize to -1 to 1
                  
                  // Calculate volume (average of all frequencies)
                  const sum = dataArray.reduce((a, b) => a + b, 0);
                  const average = sum / dataArray.length;
                  const volume = Math.min(100, (average / 255) * 100);
                  
                  // Calculate dominant frequency
                  const maxIndex = dataArray.indexOf(Math.max(...Array.from(dataArray)));
                  const frequency = (maxIndex / dataArray.length) * (audioContext.current?.sampleRate || 44100) / 2;
                  
                  setAudioAnalysis({
                    volume,
                    frequency,
                    isActive: !currentAudio.current.paused && !currentAudio.current.ended,
                    waveform
                  });
                  
                  animationFrame.current = requestAnimationFrame(analyzeAudio);
                };
                analyzeAudio();

                audio.onended = () => {
                  URL.revokeObjectURL(audioUrl);
                  currentAudio.current = null;
                  if (animationFrame.current) {
                    cancelAnimationFrame(animationFrame.current);
                  }
                  setAudioAnalysis({ volume: 0, frequency: 0, isActive: false, waveform: [] });
                  setStatus('idle');
                };
                audio.onerror = (err) => {
                  console.error('Audio playback error:', err);
                  currentAudio.current = null;
                  if (animationFrame.current) {
                    cancelAnimationFrame(animationFrame.current);
                  }
                  setAudioAnalysis({ volume: 0, frequency: 0, isActive: false, waveform: [] });
                  setStatus('idle');
                };
                audio.play().catch(err => console.error('Audio play failed:', err));
              } catch (err) {
                console.error('Audio decode error:', err);
              }
            }
            break;
            
          default:
            console.log('Unhandled state:', data.state);
        }
      } catch (error) {
        console.error('Failed to parse message:', error);
      }
    };

  }, [url, setStatus, setTranscript, setResponse, setConnected, setAudioAnalysis]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      ws.current?.close();
    };
  }, [connect]);

  const sendAudioData = (data: ArrayBuffer) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      console.log('Sending audio chunk:', data.byteLength, 'bytes');
      ws.current.send(data);
    } else {
      console.warn('WebSocket not open, state:', ws.current?.readyState);
    }
  };

  const stopRecording = () => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      console.log('Sending stop_recording command');
      ws.current.send(JSON.stringify({ command: 'stop_recording' }));
    }
  };

  return {
    isConnected: useAtlasStore((s) => s.isConnected),
    sendAudioData,
    stopRecording,
    ws: ws.current
  };
};
