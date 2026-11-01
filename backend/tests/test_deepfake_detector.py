"""Tests for deepfake detector service."""
import unittest
import os
import sys
import tempfile
import numpy as np
import scipy.io.wavfile as wav
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Ensure backend is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app.services.deepfake_detector as df


class TestDeepfakeDetector(unittest.TestCase):

    def setUp(self):
        # Reset module state
        df._INIT_ATTEMPTED = False
        df._MODEL = None
        df._MODEL_VERIFIED = False
        if "DEEPFAKE_MODEL_PATH" in os.environ:
            del os.environ["DEEPFAKE_MODEL_PATH"]

    def test_unverified_model_without_checkpoint(self):
        """Without DEEPFAKE_MODEL_PATH, detector must report unverified status and 0.0 confidence."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav.write(tmp.name, 16000, (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 32767).astype(np.int16))
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                data = f.read()

            result = df._deepfake_detection_sync(tmp_path, data)
            self.assertFalse(result["is_deepfake"])
            self.assertEqual(result["confidence"], 0.0)
            self.assertFalse(result["model_verified"])
            self.assertEqual(result["status"], "unverified_no_checkpoint")
            self.assertIn("No trained deepfake model checkpoint loaded", result["details"])
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_invalid_and_empty_audio_handling(self):
        """Empty audio bytes or non-existent files must return invalid_audio status without crashing."""
        result_empty_bytes = df._deepfake_detection_sync("nonexistent.wav", b"")
        self.assertFalse(result_empty_bytes["is_deepfake"])
        self.assertEqual(result_empty_bytes["status"], "invalid_audio")

        result_short = df._deepfake_detection_sync("nonexistent.wav", b"too_short")
        self.assertFalse(result_short["is_deepfake"])
        self.assertEqual(result_short["status"], "invalid_audio")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_verified_checkpoint_loading_and_inference(self):
        """When a valid checkpoint is provided, SpectrogramCNN must load and infer without torchvision/PIL errors."""
        model = df.SpectrogramCNN()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as ckpt:
            torch.save(model.state_dict(), ckpt.name)
            ckpt_path = ckpt.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
            wav.write(audio_tmp.name, 16000, (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 32767).astype(np.int16))
            audio_path = audio_tmp.name

        try:
            os.environ["DEEPFAKE_MODEL_PATH"] = ckpt_path
            df._INIT_ATTEMPTED = False

            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            result = df._deepfake_detection_sync(audio_path, audio_bytes)
            self.assertTrue(result["model_verified"])
            self.assertEqual(result["status"], "verified")
            self.assertIsInstance(result["confidence"], float)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
            self.assertIsInstance(result["is_deepfake"], bool)
        finally:
            if os.path.exists(ckpt_path):
                os.unlink(ckpt_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)


if __name__ == "__main__":
    unittest.main()
