import io
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional
from app.core.security import get_current_user
from app.models.call_log import ThreatResult
from app.services.transcription import transcribe_audio
from app.services.deepfake_detector import detect_deepfake
from app.services.scam_detector import detect_scam_tactics
from app.services.threat_classifier import classify_threat
from app.services.negotiator import generate_negotiator_strategy
from app.ml.audio_features import extract_audio_features
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/stream-audio", response_model=ThreatResult)
async def stream_audio_chunk(
    audio: UploadFile = File(...),
    call_id: Optional[str] = Form(None),
    chunk_index: int = Form(0),
    caller_number: Optional[str] = Form(None),
    transcript: Optional[str] = Form(None),
    current_user = Depends(get_current_user),
):
    """Receive audio chunk, transcribe, analyze, and return threat result."""
    from app.core.database import CallLog
    audio_bytes = await audio.read()

    # Validate input: audio cannot be empty or trivially small
    if not audio_bytes or len(audio_bytes) < 44:
        raise HTTPException(status_code=400, detail="Empty or invalid audio data received")

    # Save chunk temporarily for processing
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Transcription (use client-supplied live speech recognition if available or run Whisper)
        if not transcript or not transcript.strip():
            transcript = await transcribe_audio(tmp_path)
        if not transcript:
            transcript = "[Audio chunk received]"

        # Deepfake detection
        deepfake_result = await detect_deepfake(tmp_path, audio_bytes)

        # Audio acoustic feature extraction
        audio_features = extract_audio_features(tmp_path)

        # Scam tactic detection
        scam_result = await detect_scam_tactics(transcript)

        # Threat classification
        threat = classify_threat(
            deepfake_confidence=deepfake_result["confidence"],
            is_deepfake=deepfake_result["is_deepfake"],
            urgency_score=scam_result["urgency_score"],
            urgency_detected=scam_result["urgency_detected"],
            deepfake_verified=deepfake_result.get("model_verified", False),
            tactics_detected=scam_result.get("tactics_detected"),
        )

        # Negotiator strategy
        strategy = await generate_negotiator_strategy(transcript, threat["threat_level"])

        # Upsert call log
        if call_id:
            log = await CallLog.get(call_id)
        else:
            log = None

        if not log:
            log = CallLog(
                user_id=str(current_user.id),
                caller_number=caller_number,
            )

        log.transcript = ((log.transcript or "") + " " + transcript).strip()
        log.deepfake_confidence = deepfake_result["confidence"]
        log.is_deepfake = deepfake_result["is_deepfake"]
        log.urgency_score = scam_result["urgency_score"]
        log.urgency_detected = scam_result["urgency_detected"]
        log.urgency_phrases_found = scam_result["phrases_found"]
        log.overall_threat_score = threat["overall_score"]
        log.threat_level = threat["threat_level"]
        log.negotiator_strategy = strategy
        log.audio_chunks_received = getattr(log, 'audio_chunks_received', 0) + 1
        log.alert_sent = threat["alert_required"]

        try:
            await log.save()
        except Exception as e:
            logger.warning(f"Failed to save call log: {e}")

        return ThreatResult(
            call_log_id=str(log.id),
            transcript=transcript,
            is_deepfake=deepfake_result["is_deepfake"],
            deepfake_confidence=deepfake_result["confidence"],
            urgency_detected=scam_result["urgency_detected"],
            urgency_score=scam_result["urgency_score"],
            urgency_phrases_found=scam_result["phrases_found"],
            overall_threat_score=threat["overall_score"],
            threat_level=threat["threat_level"],
            negotiator_strategy=strategy,
            alert_required=threat["alert_required"],
        )
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@router.post("/analyze-audio", response_model=ThreatResult)
async def analyze_uploaded_audio(
    audio: UploadFile = File(...),
    caller_number: Optional[str] = Form(None),
    transcript: Optional[str] = Form(None),
    current_user = Depends(get_current_user),
):
    """Analyze a complete uploaded audio recording."""
    from app.core.database import CallLog
    audio_bytes = await audio.read()

    # Validate input: audio cannot be empty or trivially small
    if not audio_bytes or len(audio_bytes) < 44:
        raise HTTPException(status_code=400, detail="Empty or invalid audio data received")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        if not transcript or not transcript.strip():
            transcript = await transcribe_audio(tmp_path)
        if not transcript:
            transcript = "[Audio recording received - acoustic analysis processed]"
        deepfake_result = await detect_deepfake(tmp_path, audio_bytes)
        audio_features = extract_audio_features(tmp_path)
        scam_result = await detect_scam_tactics(transcript)
        threat = classify_threat(
            deepfake_confidence=deepfake_result["confidence"],
            is_deepfake=deepfake_result["is_deepfake"],
            urgency_score=scam_result["urgency_score"],
            urgency_detected=scam_result["urgency_detected"],
            deepfake_verified=deepfake_result.get("model_verified", False),
            tactics_detected=scam_result.get("tactics_detected"),
        )
        strategy = await generate_negotiator_strategy(transcript, threat["threat_level"])

        log = CallLog(
            user_id=str(current_user.id),
            caller_number=caller_number,
            transcript=transcript,
            deepfake_confidence=deepfake_result["confidence"],
            is_deepfake=deepfake_result["is_deepfake"],
            urgency_score=scam_result["urgency_score"],
            urgency_detected=scam_result["urgency_detected"],
            urgency_phrases_found=scam_result["phrases_found"],
            overall_threat_score=threat["overall_score"],
            threat_level=threat["threat_level"],
            negotiator_strategy=strategy,
            audio_chunks_received=1,
            alert_sent=threat["alert_required"],
        )
        try:
            await log.insert()
        except Exception as e:
            logger.warning(f"Failed to insert call log: {e}")

        return ThreatResult(
            call_log_id=str(log.id),
            transcript=transcript,
            is_deepfake=deepfake_result["is_deepfake"],
            deepfake_confidence=deepfake_result["confidence"],
            urgency_detected=scam_result["urgency_detected"],
            urgency_score=scam_result["urgency_score"],
            urgency_phrases_found=scam_result["phrases_found"],
            overall_threat_score=threat["overall_score"],
            threat_level=threat["threat_level"],
            negotiator_strategy=strategy,
            alert_required=threat["alert_required"],
        )
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
