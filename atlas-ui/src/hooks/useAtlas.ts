import { useEffect, useRef } from 'react';
import { useAtlasStore } from '../state/store';
import type { AtlasState } from '../types/index';

export const useAtlas = (url: string = 'ws://localhost:8000/api/v1/ws/orchestrated') => {
  const { setStatus, setTranscript, setResponse, setConnected } = useAtlasStore();
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      console.log('Connected to Atlas');
      setConnected(true);
    };

    ws.current.onclose = () => {
      console.log('Disconnected from Atlas');
      setConnected(false);
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
                const binaryString = atob(data.audio_base64);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                  bytes[i] = binaryString.charCodeAt(i);
                }
                const blob = new Blob([bytes], { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(blob);
                const audio = new Audio(audioUrl);
                audio.onended = () => {
                  URL.revokeObjectURL(audioUrl);
                  setStatus('idle');
                };
                audio.onerror = (err) => {
                  console.error('Audio playback error:', err);
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

    return () => {
      ws.current?.close();
    };
  }, [url, setStatus, setTranscript, setResponse, setConnected]);

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
