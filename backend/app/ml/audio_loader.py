"""
EchoShield Universal Audio Ingestion & Preprocessing Engine.

Handles:
  - Formats: WAV, MP3, M4A, OGG, WebM, FLAC, AAC, raw PCM
  - Automatic downmixing to mono
  - Resampling to target rate (default 16,000 Hz)
  - Silent, quiet, and corrupt audio detection
  - Audio segmentation for long recordings (>6 seconds)
"""

import io
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def detect_format_magic(raw_bytes: bytes) -> str:
    """Detect common audio file format from magic bytes."""
    if len(raw_bytes) < 12:
        return "unknown"
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WAVE":
        return "wav"
    if raw_bytes[:3] == b"ID3" or raw_bytes[:2] in [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"]:
        return "mp3"
    if raw_bytes[:4] == b"OggS":
        return "ogg"
    if raw_bytes[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if len(raw_bytes) >= 8 and raw_bytes[4:8] in [b"ftyp", b"moov"]:
        return "m4a"
    if raw_bytes[:4] == b"fLaC":
        return "flac"
    return "unknown"


def load_audio(
    file_path: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
    target_sr: int = 16000
) -> Tuple[Optional[np.ndarray], int, Dict[str, Any]]:
    """
    Safely load audio from file path or bytes into a normalized mono float32 numpy array.
    Supports WAV, MP3, M4A, OGG, WebM, and FLAC.
    """
    raw_data = audio_bytes
    if not raw_data and file_path and os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()
        except Exception as e:
            logger.warning(f"Failed to read audio file {file_path}: {e}")

    info: Dict[str, Any] = {
        "format": "unknown",
        "duration_seconds": 0.0,
        "sample_rate": target_sr,
        "is_valid": False,
        "is_silent": True,
        "peak_amplitude": 0.0,
        "rms_energy": 0.0
    }

    if not raw_data or len(raw_data) < 44:
        return None, target_sr, info

    detected_format = detect_format_magic(raw_data)
    info["format"] = detected_format

    y: Optional[np.ndarray] = None

    # 1. Try PyAV first for WebM, M4A, and compressed containers
    if detected_format in ["webm", "m4a", "unknown"]:
        try:
            import av
            container = av.open(io.BytesIO(raw_data))
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=target_sr)
            frames = []
            for frame in container.decode(audio=0):
                resampled = resampler.resample(frame)
                for rf in resampled:
                    frames.append(rf.to_ndarray()[0])
            if frames:
                y = np.concatenate(frames).astype(np.float32)
        except Exception as e:
            logger.debug(f"PyAV decode fallback: {e}")

    # 2. Try soundfile (handles WAV, MP3, OGG, FLAC)
    if y is None:
        try:
            import soundfile as sf
            import librosa
            data, sr = sf.read(io.BytesIO(raw_data), dtype='float32')
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            y = data.astype(np.float32)
        except Exception:
            pass

    # 3. Try librosa with file path
    if y is None and file_path and os.path.exists(file_path):
        try:
            import librosa
            data, sr = librosa.load(file_path, sr=target_sr, mono=True)
            y = data.astype(np.float32)
        except Exception:
            pass

    # 4. Fallback: Parse raw linear 16-bit PCM (RIFF or headerless)
    if y is None:
        try:
            offset = 44 if raw_data[:4] == b"RIFF" else 0
            pcm = raw_data[offset:]
            if len(pcm) % 2 != 0:
                pcm = pcm[:-1]
            if len(pcm) >= 160:
                y = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            logger.warning(f"Raw PCM buffer fallback failed: {e}")

    if y is None or len(y) < int(target_sr * 0.15):  # Less than 150ms
        return None, target_sr, info

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    duration = float(len(y) / target_sr)

    info["duration_seconds"] = round(duration, 2)
    info["peak_amplitude"] = round(peak, 4)
    info["rms_energy"] = round(rms, 6)
    info["is_valid"] = True
    info["is_silent"] = bool(peak < 0.005 or rms < 0.001)

    return y, target_sr, info


def segment_audio(
    y: np.ndarray,
    sr: int = 16000,
    segment_duration: float = 3.0,
    hop_duration: float = 2.0
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Segment audio into overlapping windows for localized analysis.
    Returns list of (segment_array, start_time_sec, end_time_sec).
    """
    total_duration = len(y) / float(sr)
    if total_duration <= (segment_duration * 1.35):
        # Short audio: return single complete segment
        return [(y, 0.0, total_duration)]

    seg_samples = int(sr * segment_duration)
    hop_samples = int(sr * hop_duration)
    segments = []

    start_idx = 0
    while start_idx + int(sr * 0.5) < len(y):
        end_idx = min(start_idx + seg_samples, len(y))
        seg_y = y[start_idx:end_idx]

        # If last segment is shorter than seg_samples, pad with zeros or take up to end
        if len(seg_y) >= int(sr * 0.5):
            t_start = start_idx / float(sr)
            t_end = end_idx / float(sr)
            segments.append((seg_y, round(t_start, 2), round(t_end, 2)))

        if end_idx == len(y):
            break
        start_idx += hop_samples

    return segments

