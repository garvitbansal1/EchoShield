"""Seed initial past call logs for users so call history is immediately populated."""
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

SAMPLE_PAST_CALLS = [
    {
        "caller_number": "+1 (800) 555-0199",
        "hours_ago": 2.25,
        "duration_seconds": 245,
        "threat_level": "CRITICAL",
        "overall_threat_score": 0.95,
        "is_deepfake": False,
        "deepfake_confidence": 0.08,
        "urgency_detected": True,
        "urgency_score": 0.98,
        "urgency_phrases_found": ["federal warrant", "immediate wire transfer", "arrest", "do not hang up"],
        "transcript": "This is Officer James from the Internal Revenue Service. A federal warrant has been issued under your Social Security Number for unpaid back taxes. You must resolve this immediately via certified wire transfer or federal marshals will be dispatched to your address within 45 minutes. Do not disconnect this line.",
        "negotiator_strategy": "Scammer is using extreme legal intimidation and artificial urgency. Feign compliance and ask for badge number and supervisor extension, then claim you need to locate tax documents in the attic to stall for 15+ minutes.",
        "voice_type": "NATURAL_HUMAN",
        "ai_voice_percentage": 4.2,
        "pre_recorded_probability": 0.05,
        "voice_verdict": "Authentic Natural Human Voice (Live Human Scammer)",
        "acoustic_markers": ["Dynamic speech prosody", "Variable fundamental frequency", "Acoustic background room noise"],
    },
    {
        "caller_number": "+1 (312) 555-0142",
        "hours_ago": 26.5,
        "duration_seconds": 180,
        "threat_level": "CRITICAL",
        "overall_threat_score": 0.92,
        "is_deepfake": True,
        "deepfake_confidence": 0.89,
        "urgency_detected": True,
        "urgency_score": 0.85,
        "urgency_phrases_found": ["huge trouble", "car accident", "send money via Zelle", "bail before 5 PM"],
        "transcript": "Grandpa, it's me! I'm in huge trouble. I got into a car accident and they're holding me in jail right now. I need $3,500 for bail before 5 PM or they're transferring me to county jail. Please don't tell mom and dad, just send the money via Zelle to this public defender.",
        "negotiator_strategy": "High-confidence synthetic voice clone detected. Ask a personal verification question only the real grandchild would know (e.g., family pet's name or childhood street). Do not send funds.",
        "voice_type": "AI_GENERATED",
        "ai_voice_percentage": 91.5,
        "pre_recorded_probability": 0.12,
        "voice_verdict": "Likely AI-Generated Voice (92% synthetic voice clone)",
        "acoustic_markers": ["Unnatural energy consistency", "Spectral phase discontinuities", "Synthetic harmonic alignment"],
    },
    {
        "caller_number": "+1 (888) 555-0177",
        "hours_ago": 54.0,
        "duration_seconds": 310,
        "threat_level": "HIGH",
        "overall_threat_score": 0.78,
        "is_deepfake": False,
        "deepfake_confidence": 0.12,
        "urgency_detected": True,
        "urgency_score": 0.82,
        "urgency_phrases_found": ["suspicious transactions", "read back passcode", "fraud prevention"],
        "transcript": "Security alert from Chase Fraud Prevention. We have flagged two suspicious transactions of $899 and $1,250 on your debit card ending in 4192. To stop these pending charges, please read back the 6-digit one-time passcode we just sent to your mobile device.",
        "negotiator_strategy": "Classic OTP harvesting scam. Banks will never ask for your verification code over the phone. Tell the caller you will call back using the verified number printed on the back of your card and hang up.",
        "voice_type": "NATURAL_HUMAN",
        "ai_voice_percentage": 8.0,
        "pre_recorded_probability": 0.10,
        "voice_verdict": "Authentic Natural Human Voice",
        "acoustic_markers": ["Organic pitch modulation", "Natural phoneme durations"],
    },
    {
        "caller_number": "+1 (844) 555-0123",
        "hours_ago": 78.0,
        "duration_seconds": 420,
        "threat_level": "MEDIUM",
        "overall_threat_score": 0.54,
        "is_deepfake": False,
        "deepfake_confidence": 0.05,
        "urgency_detected": False,
        "urgency_score": 0.45,
        "urgency_phrases_found": ["cancellation refund", "connect via AnyDesk"],
        "transcript": "EchoShield notification: automated message from Amazon Customer Support. An order for an iPhone 15 Pro Max for the amount of $1,199 has been ordered on your account. If you did not make this purchase, press 1 now to speak with a representative and connect via AnyDesk to receive a cancellation refund.",
        "negotiator_strategy": "Remote access phishing scam. Act confused about computer controls, give deliberate typos when asked to type URLs, and pretend the browser is frozen to waste agent bandwidth.",
        "voice_type": "PRE_RECORDED",
        "ai_voice_percentage": 22.0,
        "pre_recorded_probability": 0.88,
        "voice_verdict": "Pre-Recorded Automated Robocall (88% automated confidence)",
        "acoustic_markers": ["Telephony 3.8kHz lowpass attenuation", "Noise-gated silence windows", "Pre-rendered broadcast rhythm"],
    },
    {
        "caller_number": "+1 (415) 555-0188",
        "hours_ago": 102.0,
        "duration_seconds": 95,
        "threat_level": "LOW",
        "overall_threat_score": 0.04,
        "is_deepfake": False,
        "deepfake_confidence": 0.02,
        "urgency_detected": False,
        "urgency_score": 0.0,
        "urgency_phrases_found": [],
        "transcript": "Good morning, this is St. Jude Medical Clinic calling to confirm your annual physical appointment scheduled for Thursday at 10:00 AM with Dr. Bennett. Please remember to arrive 15 minutes early and bring your current insurance card. Press 1 to confirm, or 2 to reschedule.",
        "negotiator_strategy": "Standard legitimate medical confirmation call. No scam patterns or synthetic speech indicators detected. Safe to respond or confirm.",
        "voice_type": "PRE_RECORDED",
        "ai_voice_percentage": 15.0,
        "pre_recorded_probability": 0.82,
        "voice_verdict": "Pre-Recorded Automated Robocall (IVR System)",
        "acoustic_markers": ["Bandlimited telephony compression", "Fixed prompt pauses"],
    },
    {
        "caller_number": "+1 (650) 555-0164",
        "hours_ago": 145.0,
        "duration_seconds": 60,
        "threat_level": "LOW",
        "overall_threat_score": 0.02,
        "is_deepfake": False,
        "deepfake_confidence": 0.01,
        "urgency_detected": False,
        "urgency_score": 0.0,
        "urgency_phrases_found": [],
        "transcript": "Hello, this is FedEx Delivery Services. Your package with tracking ending in 8921 is scheduled for delivery today by 4:30 PM. A signature is not required. Thank you for using FedEx.",
        "negotiator_strategy": "Legitimate delivery notification. Zero threat factors identified.",
        "voice_type": "PRE_RECORDED",
        "ai_voice_percentage": 12.0,
        "pre_recorded_probability": 0.79,
        "voice_verdict": "Pre-Recorded Automated Robocall (Delivery Notification)",
        "acoustic_markers": ["Telephony bandpass cutoff", "Studio compressed audio track"],
    },
]


async def ensure_seed_logs(call_log_cls, user_id: str):
    """Ensure user has sample past calls history if they have none."""
    if not user_id or not call_log_cls:
        return
    try:
        existing = await call_log_cls.find({"user_id": str(user_id)}).limit(1).to_list()
        if existing:
            return

        now = datetime.now(timezone.utc)
        for sample in SAMPLE_PAST_CALLS:
            start_dt = now - timedelta(hours=sample["hours_ago"])
            end_dt = start_dt + timedelta(seconds=sample["duration_seconds"])

            threat_level = sample["threat_level"]
            try:
                from app.models.call_log import ThreatLevel
                threat_level = ThreatLevel(threat_level)
            except Exception:
                pass

            log = call_log_cls(
                user_id=str(user_id),
                caller_number=sample["caller_number"],
                call_start=start_dt,
                call_end=end_dt,
                transcript=sample["transcript"],
                threat_level=threat_level,
                deepfake_confidence=sample["deepfake_confidence"],
                is_deepfake=sample["is_deepfake"],
                urgency_score=sample["urgency_score"],
                urgency_detected=sample["urgency_detected"],
                overall_threat_score=sample["overall_threat_score"],
                urgency_phrases_found=sample["urgency_phrases_found"],
                negotiator_strategy=sample["negotiator_strategy"],
                alert_sent=sample["threat_level"] in ["HIGH", "CRITICAL"],
                ai_voice_percentage=sample.get("ai_voice_percentage", 0.0),
                voice_type=sample.get("voice_type", "NATURAL_HUMAN"),
                pre_recorded_probability=sample.get("pre_recorded_probability", 0.0),
                voice_verdict=sample.get("voice_verdict", None),
                acoustic_markers=sample.get("acoustic_markers", []),
            )
            await log.insert()
        logger.info(f"Seeded past calls history for user {user_id}")
    except Exception as e:
        logger.warning(f"Could not seed past calls for user {user_id}: {e}")

