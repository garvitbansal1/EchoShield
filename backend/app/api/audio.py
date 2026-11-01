import io
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional
from app.core.security import get_current_user, get_current_user_optional
from app.models.call_log import ThreatResult, VoiceDetectionResult
from app.services.transcription import transcribe_audio
from app.services.deepfake_detector import detect_deepfake
from app.services.scam_detector import detect_scam_tactics
from app.services.threat_classifier import classify_threat
from app.services.negotiator import generate_negotiator_strategy
from app.ml.audio_features import extract_audio_features, load_audio_data
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

    # Save chunk temporarily for processing preserving true extension
    original_name = audio.filename or "chunk.wav"
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in [".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac", ".aac"]:
        ext = ".wav"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
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

        # Acoustic feature extraction and deepfake harmonization
        ai_voice_pct = deepfake_result.get("ai_voice_percentage", audio_features.get("ai_voice_percentage", 0.0))
        voice_type = deepfake_result.get("voice_type", audio_features.get("voice_type", "NATURAL_HUMAN"))
        pre_rec_prob = deepfake_result.get("pre_recorded_probability", audio_features.get("pre_recorded_probability", 0.0))
        pre_rec_pct = deepfake_result.get("pre_recorded_percentage", audio_features.get("pre_recorded_percentage", 0.0))
        human_auth_pct = deepfake_result.get("human_authenticity_percentage", audio_features.get("human_authenticity_percentage", 100.0))
        voice_verdict = deepfake_result.get("voice_verdict", audio_features.get("voice_verdict", "Authentic Natural Human Voice"))
        acoustic_markers = deepfake_result.get("acoustic_markers", audio_features.get("acoustic_markers", []))

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
        log.ai_voice_percentage = ai_voice_pct
        log.voice_type = voice_type
        log.pre_recorded_probability = pre_rec_prob
        log.voice_verdict = voice_verdict
        log.acoustic_markers = acoustic_markers

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
            tactics_detected=scam_result.get("tactics_detected", []),
            threat_factors=scam_result.get("threat_factors", []),
            deepfake_verified=deepfake_result.get("model_verified", False),
            ai_voice_percentage=ai_voice_pct,
            pre_recorded_percentage=pre_rec_pct,
            human_authenticity_percentage=human_auth_pct,
            voice_type=voice_type,
            pre_recorded_probability=pre_rec_prob,
            voice_verdict=voice_verdict,
            acoustic_markers=acoustic_markers,
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

    original_name = audio.filename or "upload.wav"
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in [".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac", ".aac"]:
        ext = ".wav"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
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

        # Acoustic feature extraction and deepfake harmonization
        ai_voice_pct = deepfake_result.get("ai_voice_percentage", audio_features.get("ai_voice_percentage", 0.0))
        voice_type = deepfake_result.get("voice_type", audio_features.get("voice_type", "NATURAL_HUMAN"))
        pre_rec_prob = deepfake_result.get("pre_recorded_probability", audio_features.get("pre_recorded_probability", 0.0))
        pre_rec_pct = deepfake_result.get("pre_recorded_percentage", audio_features.get("pre_recorded_percentage", 0.0))
        human_auth_pct = deepfake_result.get("human_authenticity_percentage", audio_features.get("human_authenticity_percentage", 100.0))
        voice_verdict = deepfake_result.get("voice_verdict", audio_features.get("voice_verdict", "Authentic Natural Human Voice"))
        acoustic_markers = deepfake_result.get("acoustic_markers", audio_features.get("acoustic_markers", []))

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
            ai_voice_percentage=ai_voice_pct,
            pre_recorded_percentage=pre_rec_pct,
            human_authenticity_percentage=human_auth_pct,
            voice_type=voice_type,
            pre_recorded_probability=pre_rec_prob,
            voice_verdict=voice_verdict,
            acoustic_markers=acoustic_markers,
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
            tactics_detected=scam_result.get("tactics_detected", []),
            threat_factors=scam_result.get("threat_factors", []),
            deepfake_verified=deepfake_result.get("model_verified", False),
            ai_voice_percentage=ai_voice_pct,
            pre_recorded_percentage=pre_rec_pct,
            human_authenticity_percentage=human_auth_pct,
            voice_type=voice_type,
            pre_recorded_probability=pre_rec_prob,
            voice_verdict=voice_verdict,
            acoustic_markers=acoustic_markers,
        )
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@router.post("/detect-voice", response_model=VoiceDetectionResult)
async def detect_voice_authenticity_endpoint(
    audio: UploadFile = File(...),
    current_user = Depends(get_current_user_optional),
):
    """
    Test whether audio is AI-generated (voice clone/TTS synthetic),
    pre-recorded (automated robocall/IVR), or authentic natural human voice.
    Returns explicit % levels of AI voice, pre-recorded voice, and natural authenticity.
    """
    audio_bytes = await audio.read()
    if not audio_bytes or len(audio_bytes) < 44:
        raise HTTPException(status_code=400, detail="Empty or invalid audio data received")

    original_name = audio.filename or "upload.wav"
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in [".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac", ".aac"]:
        ext = ".wav"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        samples, sr = load_audio_data(audio_path=tmp_path, audio_bytes=audio_bytes)
        duration = (len(samples) / float(sr)) if (samples is not None and sr > 0) else 0.0

        deepfake_result = await detect_deepfake(tmp_path, audio_bytes)

        ai_pct = float(deepfake_result.get("ai_voice_percentage", 0.0))
        pr_pct = float(deepfake_result.get("pre_recorded_percentage", 0.0))
        human_pct = float(deepfake_result.get("human_authenticity_percentage", max(0.0, 100.0 - max(ai_pct, pr_pct))))
        voice_type = str(deepfake_result.get("voice_type", "NATURAL_HUMAN"))
        verdict = str(deepfake_result.get("voice_verdict", "Authentic Natural Human Voice"))
        markers = list(deepfake_result.get("acoustic_markers", []))
        is_ai = voice_type == "AI_GENERATED" or ai_pct >= 50.0 or bool(deepfake_result.get("is_deepfake", False))
        is_pr = voice_type == "PRE_RECORDED" or pr_pct >= 50.0

        return VoiceDetectionResult(
            is_ai_generated=is_ai,
            is_pre_recorded=is_pr,
            voice_type=voice_type,
            ai_voice_percentage=round(ai_pct, 1),
            pre_recorded_percentage=round(pr_pct, 1),
            human_authenticity_percentage=round(human_pct, 1),
            voice_verdict=verdict,
            acoustic_markers=markers,
            deepfake_confidence=float(deepfake_result.get("confidence", 0.0)),
            deepfake_verified=bool(deepfake_result.get("model_verified", False)),
            duration_seconds=round(duration, 2),
            sample_rate=sr if sr > 0 else 16000,
            details=deepfake_result.get("metrics", {}),
        )
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
