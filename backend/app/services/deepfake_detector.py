import os
import logging
from typing import Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from app.ml.audio_features import extract_audio_features

logger = logging.getLogger(__name__)

# Thread pool for blocking operations
_executor = ThreadPoolExecutor(max_workers=2)

# Try PyTorch and librosa
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import librosa
    TORCH_AVAILABLE = True
    logger.info("PyTorch and librosa available for deepfake detection.")
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    logger.warning(f"PyTorch/librosa not available ({type(e).__name__}). Deepfake detection disabled.")


class SpectrogramCNN(nn.Module if TORCH_AVAILABLE else object):
    """
    Lightweight CNN architecture designed to analyze mel-spectrograms for
    synthetic voice artifacts when loaded with trained weights.
    """

    def __init__(self):
        if TORCH_AVAILABLE:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


_MODEL: Optional[Any] = None
_MODEL_VERIFIED: bool = False
_INIT_ATTEMPTED: bool = False


def _get_model_status() -> tuple[Optional[Any], bool]:
    """
    Load model only if a valid trained checkpoint is explicitly configured.
    Never run random/untrained weights for deepfake classification.
    """
    global _MODEL, _MODEL_VERIFIED, _INIT_ATTEMPTED
    if _INIT_ATTEMPTED:
        return _MODEL, _MODEL_VERIFIED

    _INIT_ATTEMPTED = True

    if not TORCH_AVAILABLE:
        _MODEL = None
        _MODEL_VERIFIED = False
        return None, False

    model_path = os.getenv("DEEPFAKE_MODEL_PATH", "").strip()
    if model_path and os.path.exists(model_path):
        try:
            model = SpectrogramCNN()
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            _MODEL = model
            _MODEL_VERIFIED = True
            logger.info(f"Loaded verified deepfake model from {model_path}")
            return _MODEL, True
        except Exception as e:
            logger.error(f"Failed to load trained deepfake model from {model_path}: {e}")
            _MODEL = None
            _MODEL_VERIFIED = False
            return None, False
    else:
        logger.info(
            "No trained DEEPFAKE_MODEL_PATH configured. "
            "Deepfake detector will explicitly report unverified status rather than classifying with random weights."
        )
        _MODEL = None
        _MODEL_VERIFIED = False
        return None, False


def _preprocess_audio(audio_path: str) -> Optional[np.ndarray]:
    """Extract mel-spectrogram with duration and size validation."""
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
        return None

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=10.0)
        if len(y) < int(sr * 0.25):  # Less than 250ms of audio
            return None
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_db
    except Exception as e:
        logger.warning(f"Audio preprocessing error: {e}")
        return None


def _spectrogram_to_tensor(mel_db: np.ndarray) -> "torch.Tensor":
    """
    Normalize decibel mel-spectrogram and resize using PyTorch interpolation.
    Does not require PIL or torchvision.
    """
    min_val, max_val = mel_db.min(), mel_db.max()
    if max_val - min_val > 1e-5:
        norm_mel = (mel_db - min_val) / (max_val - min_val)
    else:
        norm_mel = np.zeros_like(mel_db)

    # Shape: (1, 1, n_mels, time_frames)
    tensor = torch.from_numpy(norm_mel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    # Resize to fixed (1, 1, 128, 128)
    tensor = F.interpolate(tensor, size=(128, 128), mode="bilinear", align_corners=False)
    return tensor


def _deepfake_detection_sync(audio_path: str, audio_bytes: bytes) -> Dict[str, Any]:
    """Synchronous deepfake detection with verified checkpoint checking and acoustic feature analysis."""
    # Validate audio bytes/file first
    if not audio_bytes or len(audio_bytes) < 100 or not os.path.exists(audio_path):
        return {
            "is_deepfake": False,
            "confidence": 0.0,
            "status": "invalid_audio",
            "model_verified": False,
            "details": "Audio input is empty or too short for analysis",
            "ai_voice_percentage": 0.0,
            "ai_voice_probability": 0.0,
            "voice_type": "NATURAL_HUMAN",
            "pre_recorded_probability": 0.0,
            "pre_recorded_percentage": 0.0,
            "human_authenticity_percentage": 0.0,
            "voice_verdict": "Audio input is empty or too short for analysis",
            "acoustic_markers": [],
            "metrics": {},
        }

    # Extract acoustic feature signatures (RMS variance, spectral flatness, HF roll-off, etc.)
    features = extract_audio_features(audio_path=audio_path, audio_bytes=audio_bytes)

    model, is_verified = _get_model_status()

    # If no verified checkpoint exists, report unverified model status while exposing acoustic heuristics
    if not is_verified or model is None:
        return {
            "is_deepfake": False,
            "confidence": 0.0,
            "status": "unverified_no_checkpoint",
            "model_verified": False,
            "details": "No trained deepfake model checkpoint loaded. Set DEEPFAKE_MODEL_PATH for verified detection.",
            "ai_voice_percentage": features.get("ai_voice_percentage", 0.0),
            "ai_voice_probability": features.get("ai_voice_probability", 0.0),
            "voice_type": features.get("voice_type", "NATURAL_HUMAN"),
            "pre_recorded_probability": features.get("pre_recorded_probability", 0.0),
            "pre_recorded_percentage": features.get("pre_recorded_percentage", 0.0),
            "human_authenticity_percentage": features.get("human_authenticity_percentage", 100.0),
            "voice_verdict": features.get("voice_verdict", "Authentic Natural Human Voice"),
            "acoustic_markers": features.get("acoustic_markers", []),
            "metrics": features.get("metrics", {}),
        }

    try:
        mel_db = _preprocess_audio(audio_path)
        if mel_db is None:
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "status": "invalid_audio",
                "model_verified": True,
                "details": "Audio contains insufficient data or could not be loaded",
                "ai_voice_percentage": features.get("ai_voice_percentage", 0.0),
                "ai_voice_probability": features.get("ai_voice_probability", 0.0),
                "voice_type": features.get("voice_type", "NATURAL_HUMAN"),
                "pre_recorded_probability": features.get("pre_recorded_probability", 0.0),
                "pre_recorded_percentage": features.get("pre_recorded_percentage", 0.0),
                "human_authenticity_percentage": features.get("human_authenticity_percentage", 100.0),
                "voice_verdict": features.get("voice_verdict", "Authentic Natural Human Voice"),
                "acoustic_markers": features.get("acoustic_markers", []),
                "metrics": features.get("metrics", {}),
            }

        tensor = _spectrogram_to_tensor(mel_db)
        with torch.no_grad():
            raw_confidence = float(model(tensor).item())

        threshold = float(os.getenv("DEEPFAKE_THRESHOLD", "0.6"))
        is_deepfake = raw_confidence >= threshold

        # Harmonize CNN model score with acoustic feature analysis
        ai_prob = max(raw_confidence, features.get("ai_voice_probability", 0.0))
        ai_pct = round(ai_prob * 100, 1)

        voice_type = features.get("voice_type", "NATURAL_HUMAN")
        if is_deepfake or ai_prob >= 0.50:
            voice_type = "AI_GENERATED"
            voice_verdict = f"Likely AI-Generated Voice ({ai_pct:.0f}% synthetic confidence)"
        else:
            voice_verdict = features.get("voice_verdict", "Authentic Natural Human Voice")

        logger.info(f"Verified deepfake score: {raw_confidence:.4f}, threshold: {threshold}, is_deepfake: {is_deepfake}")

        return {
            "is_deepfake": is_deepfake,
            "confidence": round(raw_confidence, 4),
            "status": "verified",
            "model_verified": True,
            "details": f"Model inference completed with threshold {threshold}",
            "ai_voice_percentage": ai_pct,
            "ai_voice_probability": round(ai_prob, 4),
            "voice_type": voice_type,
            "pre_recorded_probability": features.get("pre_recorded_probability", 0.0),
            "pre_recorded_percentage": features.get("pre_recorded_percentage", 0.0),
            "human_authenticity_percentage": round(max(0.0, 100.0 - max(ai_pct, features.get("pre_recorded_percentage", 0.0))), 1),
            "voice_verdict": voice_verdict,
            "acoustic_markers": features.get("acoustic_markers", []),
            "metrics": features.get("metrics", {}),
        }
    except Exception as e:
        logger.error(f"Deepfake detection inference error: {e}")
        return {
            "is_deepfake": False,
            "confidence": 0.0,
            "status": "error",
            "model_verified": is_verified,
            "details": f"Inference failed: {str(e)}",
            "ai_voice_percentage": features.get("ai_voice_percentage", 0.0),
            "ai_voice_probability": features.get("ai_voice_probability", 0.0),
            "voice_type": features.get("voice_type", "NATURAL_HUMAN"),
            "pre_recorded_probability": features.get("pre_recorded_probability", 0.0),
            "pre_recorded_percentage": features.get("pre_recorded_percentage", 0.0),
            "human_authenticity_percentage": features.get("human_authenticity_percentage", 100.0),
            "voice_verdict": features.get("voice_verdict", "Authentic Natural Human Voice"),
            "acoustic_markers": features.get("acoustic_markers", []),
            "metrics": features.get("metrics", {}),
        }


async def detect_deepfake(audio_path: str, audio_bytes: bytes) -> Dict[str, Any]:
    """
    Analyze audio for deepfake/synthetic voice artifacts.
    Returns confidence score, boolean flag, and verification metadata.
    """
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                _deepfake_detection_sync,
                audio_path,
                audio_bytes,
            ),
            timeout=10.0,
        )
        return result
    except asyncio.TimeoutError:
        logger.error("Deepfake detection timed out")
        return {
            "is_deepfake": False,
            "confidence": 0.0,
            "status": "timeout",
            "model_verified": False,
            "details": "Detection timed out after 10s",
        }
    except Exception as e:
        logger.error(f"Deepfake detection error: {e}")
        return {
            "is_deepfake": False,
            "confidence": 0.0,
            "status": "error",
            "model_verified": False,
            "details": str(e),
        }


async def analyze_voice_authenticity(audio_path: str, audio_bytes: bytes = b"") -> Dict[str, Any]:
    """
    Complete analysis of voice authenticity.
    Determines whether audio is AI-generated (synthetic voice/clone),
    pre-recorded (automated robocall/IVR), or natural human voice,
    with exact percentage confidence levels.
    """
    if not audio_bytes and audio_path and os.path.exists(audio_path):
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
        except Exception:
            pass

    return await detect_deepfake(audio_path, audio_bytes)
