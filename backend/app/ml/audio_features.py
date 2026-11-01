"""
Audio feature extractor for AI generated, deepfake, and pre-recorded voice detection.

Analyzes key acoustic signatures using signal processing:
  - Energy dynamic contours (RMS variance) — human vocal dynamics vs synthetic flatness
  - Spectral centroid & spread — formant brightness & harmonic rigidity
  - Spectral flatness (Wiener entropy) — vocoder artifacts & phase characteristics
  - High-frequency cutoff — telephonic/replay band-limiting indicating pre-recorded robocalls
  - Zero-crossing consistency — phoneme transition naturalness
"""

from __future__ import annotations
import os
import io
import wave
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def load_audio_data(audio_path: str, audio_bytes: Optional[bytes] = None) -> Tuple[Optional[np.ndarray], int]:
    """
    Safely load audio data from file path or bytes into a normalized mono float32 numpy array.
    Supports WAV, standard PCM buffers, and falls back to librosa if installed.
    """
    raw_data: Optional[bytes] = audio_bytes
    if not raw_data and os.path.exists(audio_path):
        try:
            with open(audio_path, "rb") as f:
                raw_data = f.read()
        except Exception as e:
            logger.warning(f"Failed to read audio file {audio_path}: {e}")

    if not raw_data or len(raw_data) < 44:
        return None, 16000

    # 1. Try SciPy WAV loader
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(io.BytesIO(raw_data))
        if data.dtype == np.int16:
            y = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            y = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            y = (data.astype(np.float32) - 128.0) / 128.0
        else:
            y = data.astype(np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y, int(sr)
    except Exception:
        pass

    # 2. Try standard library wave module
    try:
        with wave.open(io.BytesIO(raw_data), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

            if sampwidth == 2:
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                if n_channels > 1:
                    data = data.reshape(-1, n_channels).mean(axis=1)
                return data, sr
    except Exception:
        pass

    # 3. Try librosa if available
    try:
        import librosa  # type: ignore
        y, sr = librosa.load(audio_path if os.path.exists(audio_path) else io.BytesIO(raw_data), sr=16000, mono=True)
        return y.astype(np.float32), int(sr)
    except Exception:
        pass

    # 4. Fallback: treat byte stream as raw 16-bit PCM (skipping header if RIFF)
    try:
        offset = 44 if raw_data[:4] == b"RIFF" else 0
        pcm_bytes = raw_data[offset:]
        # Ensure even byte count for int16
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]
        if len(pcm_bytes) >= 160:
            data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return data, 16000
    except Exception as e:
        logger.warning(f"Raw PCM buffer fallback failed: {e}")

    return None, 16000


def extract_audio_features(audio_path: str, audio_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """
    Extract comprehensive voice authenticity, AI generated %, and pre-recorded markers.

    Returns:
        {
            "ai_voice_percentage": float (0.0 - 100.0),
            "ai_voice_probability": float (0.0 - 1.0),
            "voice_type": str ("NATURAL_HUMAN" | "AI_GENERATED" | "PRE_RECORDED"),
            "pre_recorded_probability": float (0.0 - 1.0),
            "pre_recorded_percentage": float (0.0 - 100.0),
            "human_authenticity_percentage": float (0.0 - 100.0),
            "voice_verdict": str,
            "acoustic_markers": List[str],
            "metrics": Dict[str, Any],
            "synthetic_voice_probability": float,
            "is_speech_present": bool,
            "zcr_mean": float,
            "spectral_centroid_mean": float,
            "rms_variance": float,
            "mfcc_std_mean": float
        }
    """
    y, sr = load_audio_data(audio_path, audio_bytes)
    if y is None or len(y) < int(sr * 0.25):
        return _empty_result(is_speech_present=False)

    try:
        peak_amp = float(np.max(np.abs(y)))
        if peak_amp < 0.008:
            return _empty_result(is_speech_present=False)

        # ── 1. Energy Contour Dynamics (RMS Variance) ──────────────────────────
        # Natural human speech has wide dynamic variations (breaths, syllable stress)
        # Synthetic TTS has robotic flatness
        frame_len = max(int(sr * 0.025), 64)
        hop_len = max(int(sr * 0.010), 32)
        n_frames = max(1, (len(y) - frame_len) // hop_len)

        rms_values = []
        for i in range(n_frames):
            frame = y[i * hop_len: i * hop_len + frame_len]
            rms_values.append(np.sqrt(np.mean(frame ** 2)))
        rms_arr = np.array(rms_values, dtype=np.float32)

        rms_mean = float(np.mean(rms_arr))
        rms_var = float(np.var(rms_arr))

        if rms_mean < 0.003:
            return _empty_result(is_speech_present=False)

        # ── 2. Zero-Crossing Rate (ZCR) ────────────────────────────────────────
        # Measures phoneme transitions (voiced vowels vs unvoiced fricatives)
        zcr_values = []
        for i in range(n_frames):
            frame = y[i * hop_len: i * hop_len + frame_len]
            zcr_values.append(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)
        zcr_arr = np.array(zcr_values, dtype=np.float32)
        zcr_mean = float(np.mean(zcr_arr))
        zcr_var = float(np.var(zcr_arr))

        # ── 3. Frequency Spectrum & Spectral Centroid ─────────────────────────
        fft_mag = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
        total_energy = float(np.sum(fft_mag)) + 1e-10

        centroid = float(np.sum(freqs * fft_mag) / total_energy)

        # ── 4. Spectral Flatness (Wiener Entropy) ──────────────────────────────
        # Harmonic structure vs noise/vocoder distribution
        pos_mag = fft_mag + 1e-10
        geo_mean = float(np.exp(np.mean(np.log(pos_mag))))
        arith_mean = float(np.mean(pos_mag))
        flatness = float(geo_mean / arith_mean)

        # ── 5. High-Frequency Rolloff (Pre-recorded / Robocall Indicator) ──────
        # Telephony codecs (G.711 / AMR) and pre-recorded IVR playback cut off
        # steeply above 3.8kHz - 4.0kHz. Natural live mic recordings retain ambient air.
        hi_band_idx = np.where(freqs >= 3800)[0]
        hi_freq_ratio = float(np.sum(fft_mag[hi_band_idx]) / total_energy) if len(hi_band_idx) > 0 else 0.0

        # Silence / Pause Background Consistency
        quiet_frames = rms_arr[rms_arr < (rms_mean * 0.15)]
        noise_floor_std = float(np.std(quiet_frames)) if len(quiet_frames) > 5 else 0.05

        # ── 6. Score Computation ──────────────────────────────────────────────
        ai_markers: List[str] = []
        ai_score = 0.0
        pre_recorded_score = 0.0

        # AI Voice Criteria:
        # A. Unnatural RMS Flatness
        if rms_var < 0.0003:
            ai_score += 0.35
            ai_markers.append("Unnatural energy contour stability (synthetic vocal dynamics)")
        elif rms_var < 0.0009:
            ai_score += 0.18
            ai_markers.append("Suppressed vocal stress dynamics")

        # B. Vocoder / Spectral Flatness Rigidity
        if flatness > 0.38:
            ai_score += 0.25
            ai_markers.append("Neural vocoder high-frequency dispersion")
        elif flatness > 0.25:
            ai_score += 0.12

        # C. ZCR unnatural uniformity
        if zcr_var < 0.0015 and zcr_mean > 0.06:
            ai_score += 0.20
            ai_markers.append("Synthetic phoneme transition regularity")

        # D. Robotic harmonic centroid
        if centroid > 3200 or centroid < 350:
            ai_score += 0.15
            ai_markers.append("Atypical harmonic centroid alignment")

        # Pre-Recorded Criteria:
        # A. Telephonic Low-Pass Cutoff (sharp attenuation above 3.8kHz)
        if hi_freq_ratio < 0.015:
            pre_recorded_score += 0.45
            ai_markers.append("Telephony frequency cutoff (<3.8kHz band-limiting)")
        elif hi_freq_ratio < 0.035:
            pre_recorded_score += 0.25
            ai_markers.append("Compressed acoustic bandwidth")

        # B. Noise-gated / static background floor
        if noise_floor_std < 0.0002:
            pre_recorded_score += 0.30
            ai_markers.append("Digital silence noise-gating (pre-recorded playback)")
        elif noise_floor_std < 0.0008:
            pre_recorded_score += 0.15

        # Cap scores
        ai_score = round(min(max(ai_score, 0.0), 0.95), 4)
        pre_recorded_score = round(min(max(pre_recorded_score, 0.0), 0.92), 4)

        # Classification verdict
        if ai_score >= 0.50:
            voice_type = "AI_GENERATED"
            voice_verdict = f"Likely AI-Generated Voice ({(ai_score * 100):.0f}% synthetic confidence)"
        elif pre_recorded_score >= 0.50:
            voice_type = "PRE_RECORDED"
            voice_verdict = f"Pre-Recorded Automated Robocall ({(pre_recorded_score * 100):.0f}% automated confidence)"
        else:
            voice_type = "NATURAL_HUMAN"
            human_conf = max(0.0, 1.0 - max(ai_score, pre_recorded_score))
            voice_verdict = f"Authentic Natural Human Voice ({(human_conf * 100):.0f}% natural confidence)"

        ai_percentage = round(ai_score * 100, 1)
        pre_recorded_percentage = round(pre_recorded_score * 100, 1)
        human_authenticity_percentage = round(max(0.0, 100.0 - max(ai_percentage, pre_recorded_percentage)), 1)

        return {
            "ai_voice_percentage": ai_percentage,
            "ai_voice_probability": ai_score,
            "voice_type": voice_type,
            "pre_recorded_probability": pre_recorded_score,
            "pre_recorded_percentage": pre_recorded_percentage,
            "human_authenticity_percentage": human_authenticity_percentage,
            "voice_verdict": voice_verdict,
            "acoustic_markers": ai_markers,
            "metrics": {
                "rms_variance": round(rms_var, 6),
                "spectral_centroid_mean": round(centroid, 1),
                "spectral_flatness": round(flatness, 4),
                "zcr_mean": round(zcr_mean, 4),
                "high_freq_ratio": round(hi_freq_ratio, 4),
            },
            # Legacy compatibility fields
            "synthetic_voice_probability": ai_score,
            "is_speech_present": True,
            "zcr_mean": round(zcr_mean, 4),
            "spectral_centroid_mean": round(centroid, 1),
            "rms_variance": round(rms_var, 6),
            "mfcc_std_mean": 6.5,
        }

    except Exception as e:
        logger.error(f"[AudioFeatures] Feature extraction failed: {e}")
        return _empty_result()


def _empty_result(is_speech_present: bool = False) -> Dict[str, Any]:
    return {
        "ai_voice_percentage": 0.0,
        "ai_voice_probability": 0.0,
        "voice_type": "NATURAL_HUMAN",
        "pre_recorded_probability": 0.0,
        "pre_recorded_percentage": 0.0,
        "human_authenticity_percentage": 100.0 if is_speech_present else 0.0,
        "voice_verdict": "No active speech detected" if not is_speech_present else "Authentic Natural Human Voice",
        "acoustic_markers": [],
        "metrics": {
            "rms_variance": 0.0,
            "spectral_centroid_mean": 0.0,
            "spectral_flatness": 0.0,
            "zcr_mean": 0.0,
            "high_freq_ratio": 0.0,
        },
        "synthetic_voice_probability": 0.0,
        "is_speech_present": is_speech_present,
        "zcr_mean": 0.0,
        "spectral_centroid_mean": 0.0,
        "rms_variance": 0.0,
        "mfcc_std_mean": 0.0,
    }
