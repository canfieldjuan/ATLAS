import { useCallback, useRef } from 'react';
import {
  useAudioRecorder,
  type RecordingConfig,
  type AudioDataEvent,
} from '@siteed/audio-studio';
import { AUDIO_CONFIG } from '../utils/constants';
import { useAtlasStore } from '../state/store';

/**
 * Mic capture using expo-audio-studio.
 * Streams PCM chunks to the provided sendAudio callback.
 */
export function useAudioCapture(sendAudio: (buffer: ArrayBuffer) => void) {
  const { setAudioAnalysis } = useAtlasStore();
  const sendAudioRef = useRef(sendAudio);
  sendAudioRef.current = sendAudio;

  const recorder = useAudioRecorder();

  const recordingConfig: RecordingConfig = {
    sampleRate: AUDIO_CONFIG.sampleRate as 16000,
    channels: AUDIO_CONFIG.channels as 1,
    encoding: AUDIO_CONFIG.encoding,
    interval: AUDIO_CONFIG.interval,
    onAudioStream: async (event: AudioDataEvent) => {
      if (event.data && typeof event.data === 'string') {
        // Native: event.data is base64-encoded PCM
        const binaryString = atob(event.data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        sendAudioRef.current(bytes.buffer);
      }

      // Update volume meter from position/totalSize
      if (event.eventDataSize > 0) {
        const normalizedVolume = Math.min(100, (event.eventDataSize / 3200) * 50);
        setAudioAnalysis({
          volume: normalizedVolume,
          isActive: true,
        });
      }
    },
  };

  const startCapture = useCallback(async () => {
    try {
      await recorder.startRecording(recordingConfig);
    } catch (e) {
      console.error('[AudioCapture] Start failed:', e);
    }
  }, [recorder]);

  const stopCapture = useCallback(async () => {
    try {
      await recorder.stopRecording();
      setAudioAnalysis({ volume: 0, isActive: false });
    } catch (e) {
      console.error('[AudioCapture] Stop failed:', e);
    }
  }, [recorder]);

  return {
    startCapture,
    stopCapture,
    isRecording: recorder.isRecording,
  };
}
