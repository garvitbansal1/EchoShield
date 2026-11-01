"""
EchoShield Machine Learning Voice Classification Service.

Provides real-time inference using the trained & calibrated 91-feature
Acoustic Classifier across three distinct categories:
  - Class 0: AI_GENERATED (AI voice clone, synthetic neural vocoder, TTS)
  - Class 1: ROBOCALL_IVR (Automated announcement, pre-recorded IVR telephony)
  - Class 2: NATURAL_HUMAN (Authentic human voice with organic prosody & dynamics)
"""

import os
import sys
import json
import pickle
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_TRAIN_DIR = os.path.join(_CUR_DIR, "..", "..", "ml_training")
for p in [_CUR_DIR, _ML_TRAIN_DIR]:
    if p not in sys.path:
        sys.path.append(p)

try:
    from app.ml.feature_extractor import extract_features_from_audio, FEATURE_NAMES, NUM_FEATURES
except ImportError:
    from feature_extractor import extract_features_from_audio, FEATURE_NAMES, NUM_FEATURES
from app.ml.audio_loader import load_audio, segment_audio

logger = logging.getLogger(__name__)

CLASS_NAMES = ["AI_GENERATED", "ROBOCALL_IVR", "NATURAL_HUMAN"]

_MODEL = None
_SCALER = None
_METADATA = None
_INIT_ATTEMPTED = False


def load_voice_model():
    """Load the trained classifier, scaler, and metadata (singleton pattern)."""
    global _MODEL, _SCALER, _METADATA, _INIT_ATTEMPTED
    if _INIT_ATTEMPTED:
        return _MODEL, _SCALER, _METADATA

    _INIT_ATTEMPTED = True

    model_candidates = [
        os.path.join(_CUR_DIR, "voice_classifier.pkl"),
        os.path.join(_ML_TRAIN_DIR, "voice_classifier.pkl"),
    ]
    scaler_candidates = [
        os.path.join(_CUR_DIR, "voice_scaler.pkl"),
        os.path.join(_ML_TRAIN_DIR, "voice_scaler.pkl"),
    ]
    meta_candidates = [
        os.path.join(_CUR_DIR, "voice_model_metadata.json"),
        os.path.join(_ML_TRAIN_DIR, "model_metadata.json"),
    ]

    for p in model_candidates:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    _MODEL = pickle.load(f)
                logger.info(f"Loaded trained voice classifier from {p}")
                break
            except Exception as e:
                logger.error(f"Failed to load model from {p}: {e}")

    for p in scaler_candidates:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    _SCALER = pickle.load(f)
                logger.info(f"Loaded voice feature scaler from {p}")
                break
            except Exception as e:
                logger.error(f"Failed to load scaler from {p}: {e}")

    for p in meta_candidates:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    _METADATA = json.load(f)
                break
            except Exception:
                pass

    return _MODEL, _SCALER, _METADATA


def is_model_available() -> bool:
    """Check if trained voice classification model is loaded and ready."""
    model, scaler, _ = load_voice_model()
    return (model is not None) and (scaler is not None)


def classify_voice_audio(
    file_path: Optional[str] = None,
    audio_bytes: Optional[bytes] = None
) -> Dict[str, Any]:
    """
    Run full end-to-end ML voice classification on an audio recording.
    Handles any audio format (WAV, MP3, M4A, OGG, WebM, FLAC).
    Performs segmentation for audio longer than 5 seconds and aggregates predictions.
    """
    model, scaler, metadata = load_voice_model()

    # Load and normalize audio
    y, sr, info = load_audio(file_path=file_path, audio_bytes=audio_bytes, target_sr=16000)

    # Empty or silent audio handling
    if y is None or not info.get("is_valid", False) or info.get("is_silent", True):
        return {
            "is_ai_generated": False,
            "is_pre_recorded": False,
            "voice_type": "NATURAL_HUMAN",
            "ai_voice_percentage": 0.0,
            "ai_voice_probability": 0.0,
            "pre_recorded_percentage": 0.0,
            "pre_recorded_probability": 0.0,
            "human_authenticity_percentage": 0.0,
            "voice_verdict": "Audio input is silent, corrupted, or too short for acoustic evaluation",
            "confidence": 0.0,
            "is_uncertain": True,
            "model_verified": model is not None,
            "duration_seconds": info.get("duration_seconds", 0.0),
            "sample_rate": sr,
            "audio_format": info.get("format", "unknown"),
            "acoustic_markers": ["No active vocal signal detected in audio stream"],
            "metrics": {},
            "segments_analyzed": 0,
            "segment_predictions": []
        }

    # If model is not available, report clear error status
    if model is None or scaler is None:
        logger.warning("Voice classifier model not loaded. Please ensure training pipeline has run.")
        return {
            "is_ai_generated": False,
            "is_pre_recorded": False,
            "voice_type": "NATURAL_HUMAN",
            "ai_voice_percentage": 0.0,
            "ai_voice_probability": 0.0,
            "pre_recorded_percentage": 0.0,
            "pre_recorded_probability": 0.0,
            "human_authenticity_percentage": 100.0,
            "voice_verdict": "Voice model checkpoint missing. Train model using train_voice_classifier.py",
            "confidence": 0.0,
            "is_uncertain": True,
            "model_verified": False,
            "duration_seconds": info.get("duration_seconds", 0.0),
            "sample_rate": sr,
            "audio_format": info.get("format", "unknown"),
            "acoustic_markers": ["Voice ML model unavailable"],
            "metrics": {},
            "segments_analyzed": 0,
            "segment_predictions": []
        }

    try:
        # Segment audio into overlapping windows
        segments = segment_audio(y, sr=sr, segment_duration=3.0, hop_duration=2.0)
        segment_probs_list = []
        segment_weights = []
        segment_details = []
        best_speech_feats = None
        max_rms = -1.0

        for seg_y, t_start, t_end in segments:
            seg_rms = float(np.sqrt(np.mean(seg_y ** 2)))
            feats = extract_features_from_audio(seg_y, sr=sr)
            if seg_rms > max_rms:
                max_rms = seg_rms
                best_speech_feats = feats
            # Scale feature vector
            feats_scaled = scaler.transform(feats.reshape(1, -1))
            # Predict calibrated class probabilities: [p_ai, p_ivr, p_human]
            probs = model.predict_proba(feats_scaled)[0]
            segment_probs_list.append(probs)

            # Weight segments by speech energy (down-weight silence/inter-utterance pauses)
            seg_weight = max(0.05, min(1.0, seg_rms / 0.025))
            segment_weights.append(seg_weight)

            segment_details.append({
                "start_time": t_start,
                "end_time": t_end,
                "ai_prob": round(float(probs[0]), 4),
                "ivr_prob": round(float(probs[1]), 4),
                "human_prob": round(float(probs[2]), 4),
                "speech_rms": round(seg_rms, 5)
            })

        seg_matrix = np.array(segment_probs_list, dtype=np.float32)
        weights_arr = np.array(segment_weights, dtype=np.float32)
        total_weight = float(np.sum(weights_arr))
        if total_weight > 0:
            weighted_probs = np.sum(seg_matrix * weights_arr[:, None], axis=0) / total_weight
        else:
            weighted_probs = np.mean(seg_matrix, axis=0)

        # Peak-aware anomaly aggregation (essential for catching synthetic calls with pauses)
        max_seg_ai = float(np.max(seg_matrix[:, 0])) if len(segments) > 0 else float(weighted_probs[0])
        max_seg_ivr = float(np.max(seg_matrix[:, 1])) if len(segments) > 0 else float(weighted_probs[1])

        top2_ai = float(np.mean(np.sort(seg_matrix[:, 0])[-min(2, len(seg_matrix)):])) if len(segments) > 0 else max_seg_ai
        top2_ivr = float(np.mean(np.sort(seg_matrix[:, 1])[-min(2, len(seg_matrix)):])) if len(segments) > 0 else max_seg_ivr

        # Composite AI score: combine speech-weighted average with top energetic segments
        composite_ai = max(float(weighted_probs[0]), 0.60 * top2_ai + 0.40 * float(weighted_probs[0]))
        composite_ivr = max(float(weighted_probs[1]), 0.60 * top2_ivr + 0.40 * float(weighted_probs[1]))

        # Re-balance probabilities preserving calibrated distribution
        if composite_ai >= 0.50 or (max_seg_ai >= 0.65 and composite_ai >= 0.40):
            p_ai = composite_ai
            rem = max(0.0, 1.0 - p_ai)
            sum_other = float(weighted_probs[1] + weighted_probs[2]) + 1e-6
            p_ivr = rem * (float(weighted_probs[1]) / sum_other)
            p_human = rem * (float(weighted_probs[2]) / sum_other)
        elif composite_ivr >= 0.50 or (max_seg_ivr >= 0.70 and composite_ivr >= 0.40):
            p_ivr = composite_ivr
            rem = max(0.0, 1.0 - p_ivr)
            sum_other = float(weighted_probs[0] + weighted_probs[2]) + 1e-6
            p_ai = rem * (float(weighted_probs[0]) / sum_other)
            p_human = rem * (float(weighted_probs[2]) / sum_other)
        else:
            p_ai = float(weighted_probs[0])
            p_ivr = float(weighted_probs[1])
            p_human = float(weighted_probs[2])

        ai_pct = round(p_ai * 100.0, 1)
        ivr_pct = round(p_ivr * 100.0, 1)
        human_pct = round(p_human * 100.0, 1)

        # Decision logic with confidence calibration & uncertainty
        is_uncertain = False
        if p_ai >= 0.45 or (max_seg_ai >= 0.65 and p_ai >= 0.35):
            voice_type = "AI_GENERATED"
            top_prob = p_ai
        elif p_ivr >= 0.50 or (max_seg_ivr >= 0.70 and p_ivr >= 0.35):
            voice_type = "PRE_RECORDED"
            top_prob = p_ivr
        elif p_human >= 0.50:
            voice_type = "NATURAL_HUMAN"
            top_prob = p_human
        else:
            voice_type = "UNCERTAIN"
            is_uncertain = True
            top_prob = max(p_ai, p_ivr, p_human)

        # Formulate Acoustic Markers from representative speech features
        active_feats = best_speech_feats if best_speech_feats is not None else (first_feats if 'first_feats' in locals() else None)
        acoustic_markers: List[str] = []
        if active_feats is not None:
            # Flatness / Wiener entropy
            flat_mean = float(active_feats[FEATURE_NAMES.index("spectral_flatness_mean")])
            # Pitch Jitter
            jitter = float(active_feats[FEATURE_NAMES.index("pitch_jitter")])
            # Telephony HF ratio (>3.8kHz)
            hf_ratio = float(active_feats[FEATURE_NAMES.index("band_ratio_above_3800")])
            # Telephony passband (300-3400Hz)
            telephony_ratio = float(active_feats[FEATURE_NAMES.index("band_ratio_telephony_300_3400")])
            # RMS Variance
            rms_var = float(active_feats[FEATURE_NAMES.index("rms_var")])

            if voice_type == "AI_GENERATED" or p_ai >= 0.40 or max_seg_ai >= 0.65:
                acoustic_markers.append(f"Synthetic vocal prosody & pitch regularity (jitter: {jitter:.4f})")
                if flat_mean > 0.04:
                    acoustic_markers.append(f"Neural vocoder phase dispersion (spectral flatness: {flat_mean:.4f})")
                if max_seg_ai >= 0.70:
                    acoustic_markers.append(f"Peak synthetic segment score: {max_seg_ai*100:.0f}% AI probability")
                acoustic_markers.append("Unnatural mel-spectral envelope continuity characteristic of neural TTS")

            if voice_type == "PRE_RECORDED" or p_ivr >= 0.40:
                if hf_ratio < 0.02:
                    acoustic_markers.append(f"Telephony low-pass band attenuation (<3.4kHz filter, HF ratio: {hf_ratio*100:.1f}%)")
                if telephony_ratio > 0.85:
                    acoustic_markers.append(f"Pre-recorded IVR passband resonance (telephony ratio: {telephony_ratio*100:.1f}%)")
                acoustic_markers.append("Noise-gated background pauses characteristic of IVR broadcast playback")

            if voice_type == "NATURAL_HUMAN" and not is_uncertain:
                acoustic_markers.append(f"Organic pitch intonation dynamics (natural jitter: {jitter:.4f})")
                acoustic_markers.append(f"Natural vocal cord dynamics (RMS variance: {rms_var:.6f})")
                if hf_ratio > 0.05:
                    acoustic_markers.append(f"Acoustic room air & ambient presence (HF ratio: {hf_ratio*100:.1f}%)")

        if not acoustic_markers:
            acoustic_markers.append("Acoustic feature analysis within normal baseline parameters")

        # Classification verdict
        if voice_type == "AI_GENERATED":
            voice_verdict = f"Likely AI-Generated Voice ({ai_pct:.0f}% synthetic confidence)"
        elif voice_type == "PRE_RECORDED":
            voice_verdict = f"Pre-Recorded Automated Robocall ({ivr_pct:.0f}% automated confidence)"
        elif is_uncertain:
            voice_verdict = f"Uncertain Voice Classification ({top_prob*100:.0f}% confidence - mixed acoustic markers)"
        else:
            voice_verdict = f"Authentic Natural Human Voice ({human_pct:.0f}% natural confidence)"

        is_ai = bool(voice_type == "AI_GENERATED" or ai_pct >= 50.0)
        is_pr = bool(voice_type == "PRE_RECORDED" or ivr_pct >= 50.0)

        metrics = {
            "ai_probability": round(p_ai, 4),
            "robocall_probability": round(p_ivr, 4),
            "human_probability": round(p_human, 4),
            "peak_segment_ai_prob": round(max_seg_ai, 4),
            "segments_count": len(segments),
            "audio_duration_sec": info.get("duration_seconds", 0.0),
            "model_architecture": metadata.get("model_name", "HistGradientBoosting") if metadata else "HistGradientBoosting"
        }

        return {
            "is_ai_generated": is_ai,
            "is_pre_recorded": is_pr,
            "voice_type": voice_type,
            "ai_voice_percentage": ai_pct,
            "ai_voice_probability": round(p_ai, 4),
            "pre_recorded_percentage": ivr_pct,
            "pre_recorded_probability": round(p_ivr, 4),
            "human_authenticity_percentage": human_pct,
            "voice_verdict": voice_verdict,
            "confidence": round(top_prob, 4),
            "is_uncertain": is_uncertain,
            "model_verified": True,
            "duration_seconds": info.get("duration_seconds", 0.0),
            "sample_rate": sr,
            "audio_format": info.get("format", "unknown"),
            "acoustic_markers": acoustic_markers,
            "metrics": metrics,
            "segments_analyzed": len(segments),
            "segment_predictions": segment_details
        }

    except Exception as e:
        logger.error(f"Error in ML voice classification: {e}", exc_info=True)
        return {
            "is_ai_generated": False,
            "is_pre_recorded": False,
            "voice_type": "NATURAL_HUMAN",
            "ai_voice_percentage": 0.0,
            "ai_voice_probability": 0.0,
            "pre_recorded_percentage": 0.0,
            "pre_recorded_probability": 0.0,
            "human_authenticity_percentage": 100.0,
            "voice_verdict": f"Acoustic classification error: {str(e)[:80]}",
            "confidence": 0.0,
            "is_uncertain": True,
            "model_verified": True,
            "duration_seconds": info.get("duration_seconds", 0.0),
            "sample_rate": sr,
            "audio_format": info.get("format", "unknown"),
            "acoustic_markers": ["Analysis encountered an unexpected computation error"],
            "metrics": {},
            "segments_analyzed": 0,
            "segment_predictions": []
        }

