"""Tests for REST API endpoints and error validation."""
import unittest
import os
import sys
import io

# Ensure backend is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.core.security import create_access_token, get_current_user
from app.core.mock_db import MockUser, clear_mock_db
import app.core.database as database


class TestAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize mock database models
        database.User = MockUser
        database.CallLog = database.MockCallLog
        cls.client = TestClient(app)

    def setUp(self):
        clear_mock_db()

    def _get_auth_headers(self):
        user = MockUser(user_id="test_user_1", email="test@example.com", username="testuser")
        import asyncio
        asyncio.run(user.insert())
        token = create_access_token({"sub": "test_user_1"})
        return {"Authorization": f"Bearer {token}"}

    def test_health_check(self):
        """GET /health must return ok status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("service"), "EchoShield")

    def test_analyze_transcript_benign(self):
        """POST /analyze with benign text must return LOW threat level."""
        headers = self._get_auth_headers()
        payload = {"transcript": "Hello, I am calling to confirm our lunch reservation tomorrow."}
        response = self.client.post("/analyze", json=payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["threat_level"], "LOW")
        self.assertEqual(data["overall_threat_score"], 0.0)
        self.assertFalse(data["alert_required"])
        self.assertIn("threat_factors", data)

    def test_analyze_transcript_scam(self):
        """POST /analyze with scam text must return CRITICAL threat level and alert required."""
        headers = self._get_auth_headers()
        payload = {
            "transcript": "This is the IRS. A warrant has been issued for your arrest. Go to a bitcoin atm and transfer funds now."
        }
        response = self.client.post("/analyze", json=payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["threat_level"], ["HIGH", "CRITICAL"])
        self.assertGreaterEqual(data["overall_threat_score"], 0.80)
        self.assertTrue(data["alert_required"])
        self.assertGreater(len(data["urgency_phrases_found"]), 0)

    def test_stream_audio_empty_rejected(self):
        """POST /stream-audio with 0-byte or empty file must return 400 Bad Request."""
        headers = self._get_auth_headers()
        files = {"audio": ("empty.wav", b"", "audio/wav")}
        response = self.client.post("/stream-audio", files=files, headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Empty or invalid audio data", response.json()["detail"])

    def test_analyze_audio_empty_rejected(self):
        """POST /analyze-audio with 0-byte or empty file must return 400 Bad Request."""
        headers = self._get_auth_headers()
        files = {"audio": ("empty.wav", b"", "audio/wav")}
        response = self.client.post("/analyze-audio", files=files, headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Empty or invalid audio data", response.json()["detail"])

    def test_get_call_logs_shows_past_history(self):
        """GET /call-logs should return seeded past calls history for user."""
        headers = self._get_auth_headers()
        response = self.client.get("/call-logs", headers=headers)
        self.assertEqual(response.status_code, 200)
        logs = response.json()
        self.assertIsInstance(logs, list)
        self.assertGreaterEqual(len(logs), 1)
        # Verify first log has expected fields
        first = logs[0]
        self.assertIn("caller_number", first)
        self.assertIn("threat_level", first)
        self.assertIn("transcript", first)
        self.assertIn("negotiator_strategy", first)
        self.assertIn("overall_threat_score", first)

    def test_get_auth_call_history_shows_past_history(self):
        """GET /auth/call-history should return past calls history for user."""
        headers = self._get_auth_headers()
        response = self.client.get("/auth/call-history", headers=headers)
        self.assertEqual(response.status_code, 200)
        logs = response.json()
        self.assertIsInstance(logs, list)
        self.assertGreaterEqual(len(logs), 1)

    def test_analyze_persists_to_call_history(self):
        """POST /analyze should add the new analyzed call to history."""
        headers = self._get_auth_headers()
        initial_res = self.client.get("/call-logs", headers=headers)
        initial_count = len(initial_res.json())

        # Analyze a new call
        payload = {"transcript": "This is your credit card company. Confirm your password now."}
        analyze_res = self.client.post("/analyze", json=payload, headers=headers)
        self.assertEqual(analyze_res.status_code, 200)

        # History should now have 1 more call
        updated_res = self.client.get("/call-logs", headers=headers)
        updated_logs = updated_res.json()
        self.assertEqual(len(updated_logs), initial_count + 1)
        self.assertEqual(updated_logs[0]["transcript"], payload["transcript"])

    def test_detect_voice_empty_rejected(self):
        """POST /detect-voice with empty audio must return 400 Bad Request."""
        files = {"audio": ("empty.wav", b"", "audio/wav")}
        response = self.client.post("/detect-voice", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Empty or invalid audio data", response.json()["detail"])

    def test_detect_voice_tone_analysis(self):
        """POST /detect-voice with synthetic tone returns voice classification and AI percentage."""
        import numpy as np
        import scipy.io.wavfile as wav
        import io

        buf = io.BytesIO()
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)
        wav.write(buf, sr, tone)
        buf.seek(0)

        files = {"audio": ("tone.wav", buf.read(), "audio/wav")}
        response = self.client.post("/detect-voice", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("ai_voice_percentage", data)
        self.assertIn("voice_type", data)
        self.assertIn("pre_recorded_percentage", data)
        self.assertIn("human_authenticity_percentage", data)
        self.assertIn("voice_verdict", data)
        self.assertIn("acoustic_markers", data)
        self.assertIsInstance(data["ai_voice_percentage"], (int, float))
        self.assertGreaterEqual(data["ai_voice_percentage"], 0.0)
        self.assertLessEqual(data["ai_voice_percentage"], 100.0)


if __name__ == "__main__":
    unittest.main()
