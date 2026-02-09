"""
FastAPI server for Nemotron Speech ASR (.nemo) checkpoints.

Supports both batch HTTP and streaming WebSocket modes.

Usage:
  python asr_server.py --model /path/to/nemotron-speech-streaming-en-0.6b.nemo --port 8080

Endpoints:
  POST /v1/asr - Batch mode: multipart/form-data with file=audio.wav (16 kHz mono)
  WS /v1/asr/stream - Streaming mode: send audio chunks, receive transcripts

Install extra deps:
  pip install -r requirements.asr.txt
System deps: libsndfile1 ffmpeg (for soundfile and NeMo audio loading).
"""

import argparse
import asyncio
import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

torch.backends.cudnn.benchmark = True
try:
    import nemo.collections.asr as nemo_asr
except ImportError as exc:  # pragma: no cover - import guard for missing optional deps
    raise SystemExit(
        "Install optional ASR deps first: pip install -r requirements.asr.txt"
    ) from exc

log = logging.getLogger("nemotron-asr-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Nemotron Speech ASR")


def load_model(model_path_or_name: str, device: str):
    log.info("Loading model from %s on %s", model_path_or_name, device)
    path = Path(model_path_or_name)
    # If it's an existing file, use restore_from; otherwise use from_pretrained (HuggingFace model name)
    if path.is_file():
        log.info("Loading from local file: %s", model_path_or_name)
        model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path_or_name, map_location=device)
    else:
        log.info("Loading from HuggingFace: %s", model_path_or_name)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_path_or_name, map_location=device)
    model = model.to(device)
    model.eval()
    model.freeze()
    if hasattr(model, "decoding"):
        try:
            from omegaconf import OmegaConf

            dec_cfg = model.cfg.get("decoding")
            if dec_cfg is not None:
                OmegaConf.set_struct(dec_cfg, False)
                if dec_cfg.get("greedy") is None:
                    dec_cfg["greedy"] = {}
                dec_cfg.greedy["use_cuda_graph_decoder"] = False
                dec_cfg["strategy"] = "greedy_batch"
                model.change_decoding_strategy(decoding_cfg=dec_cfg, verbose=False)
                OmegaConf.set_struct(dec_cfg, True)
        except Exception:
            log.warning("Could not adjust decoding config; CUDA graphs may remain enabled.")
        if hasattr(model.decoding, "use_cuda_graphs"):
            model.decoding.use_cuda_graphs = False
    target_sr = getattr(getattr(model, "preprocessor", None), "_sample_rate", 16000)
    return model, int(target_sr)


def prep_audio(wav_bytes: bytes, target_sr: int, device: torch.device) -> np.ndarray:
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    if sr != target_sr:
        tensor = torch.from_numpy(data).to(device)
        tensor = torchaudio.functional.resample(tensor, orig_freq=sr, new_freq=target_sr)
        data = tensor.cpu().numpy()
    return data


def build_app(model_path: str, device: str):
    model, target_sr = load_model(model_path, device)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "sample_rate": target_sr,
            "streaming_enabled": True,
            "endpoints": {
                "batch": "POST /v1/asr",
                "stream": "WS /v1/asr/stream",
            },
        }

    @app.post("/v1/asr")
    async def transcribe(file: UploadFile = File(...)):
        try:
            wav_bytes = await file.read()
            audio = prep_audio(wav_bytes, target_sr, device=model.device)
            with torch.inference_mode():
                transcripts = model.transcribe(
                    [audio],
                    batch_size=1,
                    use_lhotse=False,
                    num_workers=0,
                    verbose=False,
                )
            hyp = transcripts[0] if transcripts else ""
            if hasattr(hyp, "text"):
                text = hyp.text
            else:
                text = str(hyp)
            return {"text": text}
        except Exception as exc:
            log.exception("ASR error")
            return JSONResponse(status_code=500, content={"error": str(exc)})

    # =========================================================================
    # Streaming WebSocket Endpoint
    # =========================================================================

    class StreamingASRSession:
        """Manages state for a streaming ASR session."""

        def __init__(self, session_id: str, sample_rate: int = 16000):
            self.session_id = session_id
            self.sample_rate = sample_rate
            self.audio_buffer: list[np.ndarray] = []
            self.total_samples = 0
            self.last_transcript = ""
            self.created_at = time.time()

        def add_audio(self, audio_bytes: bytes) -> None:
            """Add raw PCM audio (int16) to buffer."""
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_buffer.append(audio)
            self.total_samples += len(audio)

        def get_audio(self) -> np.ndarray:
            """Get all accumulated audio as single array."""
            if not self.audio_buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(self.audio_buffer)

        def clear(self) -> None:
            """Clear audio buffer for next utterance."""
            self.audio_buffer = []
            self.total_samples = 0
            self.last_transcript = ""

        @property
        def duration_seconds(self) -> float:
            """Duration of accumulated audio in seconds."""
            return self.total_samples / self.sample_rate

    # Track active streaming sessions
    streaming_sessions: Dict[str, StreamingASRSession] = {}

    @app.websocket("/v1/asr/stream")
    async def websocket_stream(websocket: WebSocket):
        """
        WebSocket endpoint for streaming ASR.

        Protocol:
        - Client sends binary frames: raw PCM audio (int16, 16kHz, mono)
        - Client sends JSON frames for control: {"type": "finalize"} or {"type": "reset"}
        - Server sends JSON frames: {"type": "partial/final", "text": "...", "is_final": bool}
        """
        await websocket.accept()
        session_id = f"ws-{id(websocket)}-{int(time.time() * 1000)}"
        session = StreamingASRSession(session_id, sample_rate=target_sr)
        streaming_sessions[session_id] = session

        log.info("Streaming session started: %s", session_id)

        try:
            while True:
                message = await websocket.receive()

                # Handle binary audio data
                if "bytes" in message:
                    audio_bytes = message["bytes"]
                    session.add_audio(audio_bytes)

                    # Send partial transcript periodically (every 500ms of audio)
                    if session.duration_seconds >= 0.5:
                        audio = session.get_audio()
                        if len(audio) > 0:
                            try:
                                with torch.inference_mode():
                                    transcripts = model.transcribe(
                                        [audio],
                                        batch_size=1,
                                        use_lhotse=False,
                                        num_workers=0,
                                        verbose=False,
                                    )
                                hyp = transcripts[0] if transcripts else ""
                                text = hyp.text if hasattr(hyp, "text") else str(hyp)

                                # Only send if transcript changed
                                if text and text != session.last_transcript:
                                    session.last_transcript = text
                                    await websocket.send_json({
                                        "type": "partial",
                                        "text": text,
                                        "is_final": False,
                                        "duration_sec": session.duration_seconds,
                                    })
                            except Exception as e:
                                log.warning("Partial transcription error: %s", e)

                # Handle text/JSON control messages
                elif "text" in message:
                    try:
                        data = json.loads(message["text"])
                        msg_type = data.get("type", "")

                        if msg_type == "finalize":
                            # Final transcription of accumulated audio
                            audio = session.get_audio()
                            if len(audio) > 0:
                                try:
                                    with torch.inference_mode():
                                        transcripts = model.transcribe(
                                            [audio],
                                            batch_size=1,
                                            use_lhotse=False,
                                            num_workers=0,
                                            verbose=False,
                                        )
                                    hyp = transcripts[0] if transcripts else ""
                                    text = hyp.text if hasattr(hyp, "text") else str(hyp)
                                    await websocket.send_json({
                                        "type": "final",
                                        "text": text,
                                        "is_final": True,
                                        "duration_sec": session.duration_seconds,
                                    })
                                    log.info(
                                        "Session %s finalized: %.2fs audio -> '%s'",
                                        session_id,
                                        session.duration_seconds,
                                        text[:50] if text else "(empty)",
                                    )
                                except Exception as e:
                                    log.exception("Finalize transcription error")
                                    await websocket.send_json({
                                        "type": "error",
                                        "message": str(e),
                                    })
                            else:
                                await websocket.send_json({
                                    "type": "final",
                                    "text": "",
                                    "is_final": True,
                                    "duration_sec": 0,
                                })
                            # Clear buffer for next utterance
                            session.clear()

                        elif msg_type == "reset":
                            # Clear buffer without transcribing
                            session.clear()
                            await websocket.send_json({
                                "type": "reset",
                                "message": "Buffer cleared",
                            })
                            log.info("Session %s reset", session_id)

                        elif msg_type == "config":
                            # Configuration update (reserved for future use)
                            await websocket.send_json({
                                "type": "config",
                                "message": "Config acknowledged",
                                "sample_rate": target_sr,
                            })

                    except json.JSONDecodeError:
                        log.warning("Invalid JSON received: %s", message["text"][:100])

        except WebSocketDisconnect:
            log.info("Streaming session disconnected: %s", session_id)
        except RuntimeError as e:
            if "disconnect" in str(e).lower():
                log.info("Streaming session disconnected: %s", session_id)
            else:
                log.exception("WebSocket error in session %s", session_id)
        except Exception as e:
            log.exception("WebSocket error in session %s", session_id)
        finally:
            # Cleanup session
            if session_id in streaming_sessions:
                del streaming_sessions[session_id]
            log.info("Streaming session cleaned up: %s", session_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Nemotron Speech ASR HTTP server.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to nemotron-speech .nemo checkpoint.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port.")
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda if available else cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model).expanduser().resolve()
    # If it's a local file, use the resolved path; otherwise treat as HuggingFace model name
    if model_path.is_file():
        model_ref = str(model_path)
    else:
        # Assume it's a HuggingFace model name (e.g., nvidia/nemotron-speech-streaming-en-0.6b)
        model_ref = args.model
    build_app(model_ref, device)
    import uvicorn

    log.info("Serving on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
