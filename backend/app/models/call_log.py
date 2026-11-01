try:
    from beanie import Document
except ImportError:
    class Document:  # type: ignore
        pass
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ThreatLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CallLog(Document):
    user_id: str
    call_start: datetime = Field(default_factory=datetime.utcnow)
    call_end: Optional[datetime] = None
    caller_number: Optional[str] = None
    transcript: Optional[str] = None
    threat_level: ThreatLevel = ThreatLevel.LOW
    deepfake_confidence: float = 0.0
    is_deepfake: bool = False
    urgency_score: float = 0.0
    urgency_detected: bool = False
    overall_threat_score: float = 0.0
    urgency_phrases_found: List[str] = []
    negotiator_strategy: Optional[str] = None
    audio_chunks_received: int = 0
    alert_sent: bool = False
    ai_voice_percentage: float = 0.0
    voice_type: str = "NATURAL_HUMAN"
    pre_recorded_probability: float = 0.0
    voice_verdict: Optional[str] = None
    acoustic_markers: List[str] = Field(default_factory=list)

    class Settings:
        name = "call_logs"
        indexes = ["user_id", "call_start", "threat_level"]


class ThreatResult(BaseModel):
    call_log_id: str
    transcript: str
    is_deepfake: bool
    deepfake_confidence: float
    urgency_detected: bool
    urgency_score: float
    urgency_phrases_found: List[str]
    overall_threat_score: float
    threat_level: ThreatLevel
    negotiator_strategy: str
    alert_required: bool
    tactics_detected: List[str] = Field(default_factory=list)
    threat_factors: List[str] = Field(default_factory=list)
    deepfake_verified: bool = False
    ai_voice_percentage: float = 0.0
    voice_type: str = "NATURAL_HUMAN"
    pre_recorded_probability: float = 0.0
    voice_verdict: Optional[str] = None
    acoustic_markers: List[str] = Field(default_factory=list)


class VoiceDetectionResult(BaseModel):
    is_ai_generated: bool
    is_pre_recorded: bool
    voice_type: str
    ai_voice_percentage: float
    pre_recorded_percentage: float
    human_authenticity_percentage: float
    voice_verdict: str
    acoustic_markers: List[str] = Field(default_factory=list)
    deepfake_confidence: float = 0.0
    deepfake_verified: bool = False
    duration_seconds: float = 0.0
    sample_rate: int = 16000
    details: dict = Field(default_factory=dict)


class AudioChunkRequest(BaseModel):
    call_id: Optional[str] = None
    chunk_index: int = 0
    sample_rate: int = 16000


class AnalyzeTextRequest(BaseModel):
    transcript: str
    call_id: Optional[str] = None
