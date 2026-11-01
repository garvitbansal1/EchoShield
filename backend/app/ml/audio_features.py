"""
Audio feature extractor and ML voice classification interface for EchoShield.

Integrates with the trained 91-dimensional acoustic classification engine
to detect:
  - AI-Generated / Cloned Voices (Class 0: AI_GENERATED)
  - Pre-Recorded Automated Robocalls & IVR (Class 1: ROBOCALL_IVR)
  - Authentic Natural Human Speech (Class 2: NATURAL_HUMAN)
"""

from __future__ import annotations
import os
import io
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from app.ml.audio_loader import load_audio
from app.ml.voice_classifier_service import classify_voice_audio, is_model_available

logger = logging.getLogger(__name__)


def load_audio_data(audio_path: str, audio_bytes: Optional[bytes] = None) -> Tuple[Optional[np.ndarray], int]:
    """
    Safely load audio data from file path or bytes into a normalized mono float32 numpy array.
    Supports WAV, MP3, M4A, OGG, WebM, FLAC, and raw PCM.
    """
    y, sr, info = load_audio(file_path=audio_path, audio_bytes=audio_bytes, target_sr=16000)
    return y, sr


def extract_audio_features(audio_path: str, audio_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """
    Extract comprehensive voice authenticity, AI generated %, and robocall/IVR markers
    using the trained and calibrated ML acoustic classifier.
    """
    # Quick silence/empty check for backwards compatibility
    y, sr = load_audio_data(audio_path, audio_bytes)
    if y is None or len(y) < int(sr * 0.25):
        return _empty_result(is_speech_present=False)

    peak_amp = float(np.max(np.abs(y)))
    rms_amp = float(np.sqrt(np.mean(y ** 2)))
    if peak_amp < 0.008 or rms_amp < 0.001:
        return _empty_result(is_speech_present=False)

    # Run ML classifier
    res = classify_voice_audio(file_path=audio_path, audio_bytes=audio_bytes)

    # If classification returned empty or silent
    if not res.get("metrics") and res.get("confidence", 0.0) == 0.0:
        return _empty_result(is_speech_present=False)

    # Provide backwards compatible fields
    zcr_mean = float(res.get("metrics", {}).get("zcr_mean", 0.045))
    centroid = float(res.get("metrics", {}).get("spectral_centroid_mean", 1850.0))
    rms_var = float(res.get("metrics", {}).get("rms_variance", 0.0025))

    return {
        "ai_voice_percentage": res["ai_voice_percentage"],
        "ai_voice_probability": res["ai_voice_probability"],
        "voice_type": res["voice_type"],
        "pre_recorded_probability": res["pre_recorded_probability"],
        "pre_recorded_percentage": res["pre_recorded_percentage"],
        "human_authenticity_percentage": res["human_authenticity_percentage"],
        "voice_verdict": res["voice_verdict"],
        "acoustic_markers": res["acoustic_markers"],
        "metrics": res["metrics"],
        "synthetic_voice_probability": res["ai_voice_probability"],
        "is_speech_present": True,
        "is_uncertain": res.get("is_uncertain", False),
        "model_verified": res.get("model_verified", True),
        "confidence": res.get("confidence", 0.0),
        "duration_seconds": res.get("duration_seconds", 0.0),
        "sample_rate": res.get("sample_rate", sr),
        # Legacy fields
        "zcr_mean": round(zcr_mean, 4),
        "spectral_centroid_mean": round(centroid, 1),
        "rms_variance": round(rms_var, 6),
        "mfcc_std_mean": 6.5,
    }


def _empty_result(is_speech_present: bool = False) -> Dict[str, Any]:
    return {
        "ai_voice_percentage": 0.0,
        "ai_voice_probability": 0.0,
        "voice_type": "NATURAL_HUMAN",
        "pre_recorded_probability": 0.0,
        "pre_recorded_percentage": 0.0,
        "human_authenticity_percentage": 100.0 if is_speech_present else 0.0,
        "voice_verdict": "No active speech detected" if not is_speech_present else "Authentic Natural Human Voice",
        "acoustic_markers": ["No active speech signal detected"],
        "metrics": {
            "rms_variance": 0.0,
            "spectral_centroid_mean": 0.0,
            "spectral_flatness": 0.0,
            "zcr_mean": 0.0,
            "high_freq_ratio": 0.0,
        },
        "synthetic_voice_probability": 0.0,
        "is_speech_present": is_speech_present,
        "is_uncertain": not is_speech_present,
        "model_verified": is_model_available(),
        "confidence": 0.0,
        "duration_seconds": 0.0,
        "sample_rate": 16000,
        "zcr_mean": 0.0,
        "spectral_centroid_mean": 0.0,
        "rms_variance": 0.0,
        "mfcc_std_mean": 0.0,
    }
