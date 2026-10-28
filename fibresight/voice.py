"""Speech to text with Meta Muse Voice Transcribe (realtime WebSocket API).

The API is streaming only: open a socket, send a JSON handshake that carries
the API key and audio format, stream raw PCM16 audio as binary frames, and
read JSON events back ("transcript" partials and "speechComplete" finals).
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid

import websockets

URL = "wss://api.meta.ai/v1/asr/realtime"
MODEL = "muse-voice-transcribe-1.0"


async def transcribe_pcm(pcm16: bytes, api_key: str, rate: int = 16000, keywords=None,
                         chunk_ms: int = 100, speed: float = 1.5) -> dict:
    """Stream mono PCM16 audio and return the final transcript segments."""
    handshake = {"authorization": {"accessToken": f"Bearer {api_key}"},
                 "audioEncoding": f"PCM_{rate // 1000}KHZ", "model": MODEL, "mode": "ENDPOINTING",
                 "partialMode": "CUMULATIVE", "emitAudioProgress": False, "languageBias": ["English"]}
    if keywords:
        handshake["keywords"] = list(keywords)
    finals, last_partial = [], ""
    async with websockets.connect(f"{URL}?sessionId=stream-{uuid.uuid4()}", max_size=None) as ws:
        await ws.send(json.dumps(handshake))
        ack = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if "sessionId" not in ack:
            raise RuntimeError(f"transcription session rejected: {ack}")

        async def sender():
            step = rate * 2 * chunk_ms // 1000
            for i in range(0, len(pcm16), step):
                await ws.send(pcm16[i:i + step])
                await asyncio.sleep(chunk_ms / 1000 / speed)
            await asyncio.sleep(0.5)
            await ws.send(json.dumps({"type": "endStream"}))

        task = asyncio.create_task(sender())
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), 30)
                if isinstance(msg, bytes):
                    continue
                ev = json.loads(msg)
                t = ev.get("type")
                if t == "transcript":
                    last_partial = ev.get("transcript", "")
                elif t == "speechComplete":
                    finals.append(ev.get("transcript", last_partial))
                    last_partial = ""
                elif t == "error":
                    raise RuntimeError(f"transcription error: {ev}")
                elif t in ("endStream", "streamEnd", "sessionEnd", "done"):
                    break
        except (asyncio.TimeoutError, websockets.ConnectionClosed):
            pass
        finally:
            task.cancel()
    # A trailing partial is kept only if it is new text (not a repeat of the last final).
    if last_partial and not (finals and finals[-1].strip().startswith(last_partial.strip())):
        finals.append(last_partial)
    return {"model": MODEL, "segments": finals, "text": " ".join(s.strip() for s in finals if s.strip())}


def transcribe_file(path: str, api_key: str = None, keywords=None) -> dict:
    """Convert any audio file to 16 kHz mono PCM with ffmpeg, then transcribe."""
    import subprocess
    api_key = api_key or os.environ["META_API_KEY"]
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         check=True, capture_output=True).stdout
    return asyncio.run(transcribe_pcm(pcm, api_key, 16000, keywords))


FIBRE_KEYWORDS = ["FibreSight", "OTDR", "ODF", "splice", "macrobend", "attenuator", "dBm", "Rx", "Tx", "1550", "1310",
                  "patch panel", "connector", "closure", "LOS", "pre-FEC"]
