from fastapi import APIRouter, Depends, HTTPException, Body
from app.core.security import get_current_user
from app.services.scam_detector import detect_scam_tactics
from app.services.threat_classifier import classify_threat
from app.services.negotiator import generate_negotiator_strategy
from typing import Optional
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=dict)
async def analyze_transcript(
    request: dict = Body(...),
    current_user = Depends(get_current_user),
):
    """Analyze a transcript string directly and return threat assessment."""
    import asyncio
    
    transcript = request.get("transcript", "")
    call_id = request.get("call_id")
    
    if not transcript or len(transcript) < 5:
        raise HTTPException(status_code=400, detail="Transcript too short")
    
    logger.info(f"Analyzing transcript: {transcript[:100]}...")
    
    # Run analysis tasks concurrently
    scam_task = detect_scam_tactics(transcript)
    strategy_task = None  # Will create after threat classification
    
    scam_result = await asyncio.wait_for(scam_task, timeout=65.0)
    logger.info(f"Scam detection complete: {scam_result['urgency_score']}")
    
    threat = classify_threat(
        deepfake_confidence=0.0,
        is_deepfake=False,
        urgency_score=scam_result["urgency_score"],
        urgency_detected=scam_result["urgency_detected"],
        deepfake_verified=False,
        tactics_detected=scam_result.get("tactics_detected"),
    )
    logger.info(f"Threat classification: {threat['threat_level']}")
    
    strategy = await asyncio.wait_for(
        generate_negotiator_strategy(transcript, threat["threat_level"]),
        timeout=15.0
    )
    logger.info(f"Negotiator strategy generated")

    # Serialize threat_level enum to string
    threat_level_str = threat["threat_level"].value if hasattr(threat["threat_level"], 'value') else str(threat["threat_level"])

    # Persist call to database so it appears in past calls history
    from app.core.database import CallLog
    try:
        new_log = CallLog(
            user_id=str(current_user.id),
            caller_number=request.get("caller_number", "+1 (555) 019-8834"),
            transcript=transcript,
            threat_level=threat["threat_level"],
            deepfake_confidence=0.0,
            is_deepfake=False,
            urgency_score=scam_result["urgency_score"],
            urgency_detected=scam_result["urgency_detected"],
            overall_threat_score=threat["overall_score"],
            urgency_phrases_found=scam_result.get("phrases_found", []),
            negotiator_strategy=strategy,
            alert_sent=threat["alert_required"],
        )
        await new_log.insert()
        call_log_id = str(new_log.id)
    except Exception as e:
        logger.warning(f"Failed to save call log: {e}")
        call_log_id = call_id or str(uuid4())

    logger.info(f"Analysis complete. Returning response.")

    return {
        "call_log_id": call_log_id,
        "transcript": transcript,
        "is_deepfake": False,
        "deepfake_confidence": 0.0,
        "urgency_detected": scam_result["urgency_detected"],
        "urgency_score": scam_result["urgency_score"],
        "urgency_phrases_found": scam_result["phrases_found"],
        "tactics_detected": scam_result.get("tactics_detected", []),
        "overall_threat_score": threat["overall_score"],
        "threat_level": threat_level_str,
        "threat_factors": threat.get("threat_factors", []),
        "negotiator_strategy": strategy,
        "alert_required": threat["alert_required"],
    }


@router.get("/call-logs")
async def get_call_logs(
    limit: int = 20,
    current_user = Depends(get_current_user),
):
    """Get call logs for current user."""
    from app.core.database import CallLog
    from app.core.seed_data import ensure_seed_logs
    
    await ensure_seed_logs(CallLog, str(current_user.id))
    logs = await CallLog.find({"user_id": str(current_user.id)}).sort([("call_start", -1)]).limit(limit).to_list()
    return [
        {
            "id": str(log.id),
            "call_start": log.call_start.isoformat() if hasattr(log.call_start, 'isoformat') else str(log.call_start),
            "call_end": log.call_end.isoformat() if getattr(log, 'call_end', None) and hasattr(log.call_end, 'isoformat') else getattr(log, 'call_end', None),
            "threat_level": log.threat_level.value if hasattr(log.threat_level, 'value') else str(log.threat_level),
            "overall_threat_score": log.overall_threat_score,
            "is_deepfake": log.is_deepfake,
            "deepfake_confidence": getattr(log, 'deepfake_confidence', 0.0),
            "urgency_detected": log.urgency_detected,
            "urgency_score": getattr(log, 'urgency_score', 0.0),
            "urgency_phrases_found": getattr(log, 'urgency_phrases_found', []),
            "transcript": log.transcript,
            "negotiator_strategy": log.negotiator_strategy,
            "caller_number": log.caller_number,
            "ai_voice_percentage": getattr(log, 'ai_voice_percentage', 0.0),
            "voice_type": getattr(log, 'voice_type', 'NATURAL_HUMAN'),
            "pre_recorded_probability": getattr(log, 'pre_recorded_probability', 0.0),
            "voice_verdict": getattr(log, 'voice_verdict', None),
            "acoustic_markers": getattr(log, 'acoustic_markers', []),
        }
        for log in logs
    ]


@router.get("/stats")
async def get_stats(current_user = Depends(get_current_user)):
    """Get user statistics."""
    from app.core.database import CallLog
    from app.core.seed_data import ensure_seed_logs
    
    await ensure_seed_logs(CallLog, str(current_user.id))
    all_logs = await CallLog.find({"user_id": str(current_user.id)}).to_list()
    total_calls = len(all_logs)
    # threat_level may be a ThreatLevel enum or a plain string — normalise to str
    scam_calls = sum(1 for l in all_logs if str(getattr(l.threat_level, 'value', l.threat_level)) in ["HIGH", "CRITICAL"])
    deepfake_calls = sum(1 for l in all_logs if l.is_deepfake)
    avg_threat = sum(l.overall_threat_score for l in all_logs) / total_calls if total_calls else 0

    threat_breakdown = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for log in all_logs:
        level_str = str(getattr(log.threat_level, 'value', log.threat_level))
        threat_breakdown[level_str] = threat_breakdown.get(level_str, 0) + 1

    return {
        "total_calls_analyzed": total_calls,
        "scam_calls_detected": scam_calls,
        "deepfake_calls_detected": deepfake_calls,
        "average_threat_score": round(avg_threat, 3),
        "threat_breakdown": threat_breakdown,
    }
