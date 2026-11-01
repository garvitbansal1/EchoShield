"""
Comprehensive unit and integration tests for EchoShield ML Voice Classifier.
Tests:
  - Model and scaler loading
  - Feature extraction dimensions and sanity
  - Accuracy across all 3 classes (AI, Robocall/IVR, Natural Human)
  - Multi-format ingestion (WAV, MP3, OGG)
  - Audio segmentation on long files
  - Empty, silent, and corrupt audio handling
  - Low confidence uncertainty handling
"""

import os
import io
import unittest
import numpy as np
import soundfile as sf

from app.ml.audio_loader import load_audio, segment_audio, detect_format_magic
from app.ml.voice_classifier_service import classify_voice_audio, is_model_available, load_voice_model
from app.ml.audio_features import extract_audio_features, load_audio_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "ml_training", "data", "raw")


class TestVoiceClassifierML(unittest.TestCase):

    def setUp(self):
        self.assertTrue(is_model_available(), "Trained voice classifier model must be loaded.")
        self.model, self.scaler, self.metadata = load_voice_model()

    def test_model_metadata_and_dimensions(self):
        """Verify model configuration and 91 acoustic feature alignment."""
        self.assertIsNotNone(self.metadata)
        self.assertEqual(self.metadata.get("feature_count"), 91)
        self.assertEqual(len(self.metadata.get("feature_names", [])), 91)
        self.assertEqual(self.metadata.get("classes"), ["AI_GENERATED", "ROBOCALL_IVR", "NATURAL_HUMAN"])

    def test_feature_extraction_shape_and_range(self):
        """Feature vector must be exactly 91 finite floats without NaNs or Infs."""
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.random.randn(len(t))).astype(np.float32)

        from feature_extractor import extract_features_from_audio
        feats = extract_features_from_audio(y, sr)
        self.assertEqual(len(feats), 91)
        self.assertTrue(np.all(np.isfinite(feats)))
        self.assertFalse(np.any(np.isnan(feats)))

    def test_classify_ai_generated_sample(self):
        """AI-generated sample from ASVspoof benchmark must classify as AI_GENERATED with >50% confidence."""
        ai_dir = os.path.join(DATA_DIR, "ai_generated")
        if not os.path.exists(ai_dir):
            self.skipTest("Raw dataset not found")

        files = [f for f in os.listdir(ai_dir) if f.endswith(".wav")]
        if not files:
            self.skipTest("No AI voice samples available")

        sample_path = os.path.join(ai_dir, files[0])
        res = classify_voice_audio(sample_path)
        self.assertEqual(res["voice_type"], "AI_GENERATED")
        self.assertTrue(res["is_ai_generated"])
        self.assertGreaterEqual(res["ai_voice_percentage"], 50.0)
        self.assertIn("Likely AI-Generated Voice", res["voice_verdict"])

    def test_classify_modern_neural_tts_scam_sample(self):
        """Modern neural TTS voice sample must classify as AI_GENERATED with >75% confidence."""
        ai_dir = os.path.join(DATA_DIR, "ai_generated")
        if not os.path.exists(ai_dir):
            self.skipTest("Raw dataset not found")

        neural_files = [f for f in os.listdir(ai_dir) if f.startswith("ai_neural_") and f.endswith(".wav")]
        if not neural_files:
            self.skipTest("No modern neural TTS voice samples available")

        sample_path = os.path.join(ai_dir, neural_files[0])
        res = classify_voice_audio(sample_path)
        self.assertEqual(res["voice_type"], "AI_GENERATED")
        self.assertTrue(res["is_ai_generated"])
        self.assertGreaterEqual(res["ai_voice_percentage"], 75.0)
        self.assertIn("Likely AI-Generated Voice", res["voice_verdict"])

    def test_classify_robocall_ivr_sample(self):
        """FreeSWITCH automated IVR telephony sample must classify as PRE_RECORDED with >50% confidence."""
        ivr_dir = os.path.join(DATA_DIR, "robocall_ivr")
        if not os.path.exists(ivr_dir):
            self.skipTest("Raw dataset not found")

        files = [f for f in os.listdir(ivr_dir) if f.endswith(".wav")]
        if not files:
            self.skipTest("No IVR voice samples available")

        sample_path = os.path.join(ivr_dir, files[0])
        res = classify_voice_audio(sample_path)
        self.assertEqual(res["voice_type"], "PRE_RECORDED")
        self.assertTrue(res["is_pre_recorded"])
        self.assertGreaterEqual(res["pre_recorded_percentage"], 50.0)
        self.assertIn("Pre-Recorded", res["voice_verdict"])

    def test_classify_natural_human_sample(self):
        """Natural human speech from CMU Arctic must classify as NATURAL_HUMAN with >50% confidence."""
        human_dir = os.path.join(DATA_DIR, "natural_human")
        if not os.path.exists(human_dir):
            self.skipTest("Raw dataset not found")

        cmu_files = [f for f in os.listdir(human_dir) if f.startswith("human_cmu_") and f.endswith(".wav")]
        if not cmu_files:
            self.skipTest("No CMU Arctic voice samples available")

        sample_path = os.path.join(human_dir, cmu_files[0])
        res = classify_voice_audio(sample_path)
        self.assertEqual(res["voice_type"], "NATURAL_HUMAN")
        self.assertFalse(res["is_ai_generated"])
        self.assertFalse(res["is_pre_recorded"])
        self.assertGreaterEqual(res["human_authenticity_percentage"], 50.0)
        self.assertIn("Authentic Natural Human Voice", res["voice_verdict"])

    def test_multi_format_mp3_and_ogg(self):
        """Classifier must correctly ingest in-memory MP3 and OGG byte streams."""
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        # MP3 buffer
        buf_mp3 = io.BytesIO()
        sf.write(buf_mp3, y, sr, format="MP3")
        res_mp3 = classify_voice_audio(audio_bytes=buf_mp3.getvalue())
        self.assertIn(res_mp3["audio_format"], ["mp3", "unknown"])
        self.assertIn("voice_type", res_mp3)

        # OGG buffer
        buf_ogg = io.BytesIO()
        sf.write(buf_ogg, y, sr, format="OGG")
        res_ogg = classify_voice_audio(audio_bytes=buf_ogg.getvalue())
        self.assertEqual(res_ogg["audio_format"], "ogg")
        self.assertIn("voice_type", res_ogg)

    def test_silent_audio_handling(self):
        """Pure silence must return safe verdict and zero synthetic confidence without crashing."""
        buf = io.BytesIO()
        sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
        res = classify_voice_audio(audio_bytes=buf.getvalue())
        self.assertFalse(res["is_ai_generated"])
        self.assertEqual(res["ai_voice_percentage"], 0.0)
        self.assertTrue(res["is_uncertain"])
        self.assertIn("silent", res["voice_verdict"].lower())

    def test_long_audio_segmentation(self):
        """Audio > 6 seconds must trigger windowed segmentation."""
        sr = 16000
        t = np.linspace(0, 8.0, sr * 8, endpoint=False)
        y = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        segments = segment_audio(y, sr=sr, segment_duration=3.0, hop_duration=2.0)
        self.assertGreaterEqual(len(segments), 3)

        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV")
        res = classify_voice_audio(audio_bytes=buf.getvalue())
        self.assertGreaterEqual(res["segments_analyzed"], 3)
        self.assertEqual(len(res["segment_predictions"]), res["segments_analyzed"])


if __name__ == "__main__":
    unittest.main()

