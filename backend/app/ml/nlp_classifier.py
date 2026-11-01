"""
NLP-based scam classifier.

Provides explainable scam detection by analyzing conversational intent, coercive tactics,
and financial demands. Decouples general sentiment analysis from scam classification so that
benign negative sentiment (e.g. sadness, frustration) is never falsely flagged as a scam.
"""

from __future__ import annotations
import os
import logging
import re
from typing import Dict, Any, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.ml.trained_scam_model import predict_scam as predict_trained_scam

logger = logging.getLogger(__name__)

# Thread pool for blocking operations
_executor = ThreadPoolExecutor(max_workers=1)

# ─── Model Configuration ───────────────────────────────────────────────────────
_pipeline = None
_MODEL_LOADED: bool = False
_IS_SCAM_SPECIFIC_MODEL: bool = False
_SCAM_MODEL_PATH = os.getenv("SCAM_MODEL_PATH", "").strip()
_SCAM_MODEL_NAME = os.getenv("SCAM_MODEL_NAME", "").strip()
_USE_TRANSFORMERS = os.getenv("USE_TRANSFORMERS_NLP", "false").lower() in ["true", "1", "yes"]


def _load_model() -> bool:
    """
    Lazy-load NLP text classification pipeline if explicitly configured.
    Avoids spontaneous 250MB+ model downloads in resource-constrained or offline environments.
    """
    global _pipeline, _MODEL_LOADED, _IS_SCAM_SPECIFIC_MODEL
    if _MODEL_LOADED:
        return _pipeline is not None

    _MODEL_LOADED = True

    # Only load if explicit path is provided or USE_TRANSFORMERS_NLP is set
    model_identifier = _SCAM_MODEL_PATH or _SCAM_MODEL_NAME
    if not model_identifier and not _USE_TRANSFORMERS:
        logger.info("[NLP] No external scam model checkpoint specified. Running calibrated contextual engine.")
        _pipeline = None
        return False

    target_model = model_identifier or "distilbert-base-uncased-finetuned-sst-2-english"

    try:
        from transformers import pipeline  # type: ignore

        _pipeline = pipeline(
            "text-classification",
            model=target_model,
            truncation=True,
            max_length=512,
        )
        model_name_lower = target_model.lower()
        if any(term in model_name_lower for term in ["scam", "spam", "fraud", "phishing"]):
            _IS_SCAM_SPECIFIC_MODEL = True
            logger.info(f"[NLP] Loaded dedicated scam/spam classifier: {target_model}")
        else:
            _IS_SCAM_SPECIFIC_MODEL = False
            logger.info(
                f"[NLP] Loaded general classifier ({target_model}). "
                "Negative sentiment will NOT be treated as proof of scam."
            )
        return True
    except Exception as e:
        logger.info(f"[NLP] Transformer model '{target_model}' could not be loaded: {e}. Using calibrated contextual engine.")
        _pipeline = None
        return False


# ─── Contextual Intent Signal Patterns ─────────────────────────────────────────

_TACTIC_CATEGORIES = {
    "FINANCIAL_DEMAND": [
        r"\b(gift\s+card|apple\s+gift|google\s+play|target\s+card|steam\s+card|itunes\s+card)\b",
        r"\b(bitcoin\s+atm|crypto\s+atm|cryptocurrency\s+deposit|bitcoin\s+kiosk|crypto\s+wallet)\b",
        r"\b(wire\s+transfer|send\s+money\s+immediately|transfer\s+funds?\s+now|zelle\s+payment|western\s+union|moneygram)\b",
        r"\b(withdraw\s+cash\s+and\s+deposit|cash\s+in\s+an\s+envelope|safe\s+locker\s+deposit)\b",
    ],
    "COERCION_AND_THREATS": [
        r"\b(arrest\s+warrant|warrant\s+out\s+for\s+your\s+arrest|you\s+will\s+be\s+arrested)\b",
        r"\b(social\s+security\s+(number\s+has\s+been\s+)?suspended|ssn\s+has\s+been\s+flagged)\b",
        r"\b(account\s+will\s+be\s+(blocked|suspended|closed|terminated|frozen))\b",
        r"\b(police\s+(officer|department)\s+will\s+be\s+dispatched|sheriff\s+at\s+your\s+door)\b",
        r"\bdon'?t\s+hang\s+up\b",
        r"\bfinal\s+(notice|warning)\b",
        r"\bimmediate\s+(action|payment)\s+required\b",
    ],
    "IMPERSONATION_AUTHORITY": [
        r"\b(irs|internal\s+revenue\s+service)\b",
        r"\b(social\s+security\s+administration)\b",
        r"\b(microsoft\s+technical\s+support|windows\s+defender\s+(alert|warning)|apple\s+support)\b",
        r"\b(bank\s+fraud\s+department|security\s+division\s+of\s+your\s+bank)\b",
        r"\b(federal\s+agent|investigator\s+badge|customs\s+and\s+border\s+protection)\b",
    ],
    "REMOTE_ACCESS_EXPLOIT": [
        r"\bdownload\s+(anydesk|teamviewer|ultraviewer|quick\s+assist)\b",
        r"\b(give\s+me|grant)\s+remote\s+access\b",
        r"\binstall\s+(the\s+support\s+software|anydesk|teamviewer)\b",
        r"\bvirus\s+detected\s+on\s+your\s+computer\b",
    ],
    "CREDENTIAL_HARVESTING": [
        r"\b(provide|verify|give\s+me)\s+your\s+(full\s+)?(credit\s+card|card\s+number|cvv|routing\s+number|pin)\b",
        r"\b(one-time\s+password|otp|verification\s+code)\b",
        r"\bprovide\s+your\s+(social\s+security\s+number|full\s+ssn)\b",
    ],
    "SECRECY_ISOLATION": [
        r"\bdo\s+not\s+tell\s+(anyone|your\s+family|your\s+spouse|the\s+bank\s+teller)\b",
        r"\bkeep\s+this\s+(strictly\s+)?confidential\b",
        r"\bfederal\s+gag\s+order\b",
    ],
    "PRIZE_ADVANCE_FEE": [
        r"\bcongratulations\s+you\s+(have\s+)?(won|been\s+selected)\b",
        r"\b(lottery\s+winner|unclaimed\s+grant|inheritance\s+funds?)\b",
        r"\bpay\s+a\s+(small\s+)?processing\s+fee\s+to\s+release\b",
    ],
}

# Compile regular expressions for efficiency
_COMPILED_CATEGORIES = {
    cat: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for cat, patterns in _TACTIC_CATEGORIES.items()
}


def _analyze_contextual_scam_signals(text: str) -> tuple[float, List[str], Dict[str, List[str]]]:
    """
    Calibrated contextual signal analyzer.
    Evaluates co-occurrence of predatory tactics rather than raw keyword hits or sentiment.
    """
    text_clean = text.lower().strip()
    triggered_categories: Dict[str, List[str]] = {}
    matched_phrases: List[str] = []

    for cat_name, compiled_patterns in _COMPILED_CATEGORIES.items():
        cat_matches = []
        for pattern in compiled_patterns:
            matches = pattern.findall(text_clean)
            if matches:
                for match in matches:
                    phrase = match if isinstance(match, str) else " ".join(filter(None, match))
                    if phrase and phrase not in cat_matches:
                        cat_matches.append(phrase)
                        if phrase not in matched_phrases:
                            matched_phrases.append(phrase)
        if cat_matches:
            triggered_categories[cat_name] = cat_matches

    num_categories = len(triggered_categories)

    # ── Contextual Risk Synthesis ─────────────────────────────────────────────
    # If no recognized scam tactics match, probability is virtually zero
    if num_categories == 0:
        return 0.0, [], {}

    has_financial = "FINANCIAL_DEMAND" in triggered_categories
    has_coercion = "COERCION_AND_THREATS" in triggered_categories
    has_impersonation = "IMPERSONATION_AUTHORITY" in triggered_categories
    has_remote = "REMOTE_ACCESS_EXPLOIT" in triggered_categories
    has_harvesting = "CREDENTIAL_HARVESTING" in triggered_categories
    has_secrecy = "SECRECY_ISOLATION" in triggered_categories
    has_prize = "PRIZE_ADVANCE_FEE" in triggered_categories

    score = 0.0

    # Critical single indicators
    if has_remote:
        score = max(score, 0.85)  # Tech support remote access scams
    if has_financial:
        score = max(score, 0.70)  # Gift card / Bitcoin ATM demands
    if has_coercion:
        score = max(score, 0.60)  # Arrest warrants, SSN suspension
    if has_harvesting:
        score = max(score, 0.65)  # OTP, PIN, SSN harvesting
    if has_impersonation:
        score = max(score, 0.45)  # IRS, SSA impersonation
    if has_prize:
        score = max(score, 0.40)  # Fake lottery / prize claims

    # Synergistic co-occurrence escalations (hallmark of sophisticated scams)
    if (has_financial or has_harvesting) and (has_coercion or has_impersonation):
        score = max(score, 0.95)
    if has_remote and (has_impersonation or has_coercion):
        score = max(score, 0.95)
    if has_prize and (has_financial or has_harvesting):
        score = max(score, 0.92)
    if has_secrecy and (has_financial or has_coercion or has_impersonation):
        score = min(score + 0.15, 0.98)

    # Multi-category bonus
    if num_categories >= 3:
        score = max(score, 0.90)

    return round(min(score, 0.99), 4), matched_phrases, triggered_categories


def _classify_with_model(transcript: str) -> Dict[str, Any]:
    """Synchronous classification blending model inference and contextual rules."""
    model_loaded = _load_model()
    contextual_score, phrases, categories = _analyze_contextual_scam_signals(transcript)
    
        # Newly trained EchoShield ML model
    try:
        trained_result = predict_trained_scam(transcript)

        trained_probability = trained_result["scam_probability"]

        # Combine trained ML and contextual rule-based scores
        if contextual_score > 0:
            # High-confidence contextual rules must not be diluted downwards
            final_prob = max(contextual_score, round(
                0.60 * trained_probability +
                0.40 * contextual_score,
                4
            ))
        else:
            # If no contextual scam tactics or coercion exist, low background score (<0.35)
            # must not flag benign conversation.
            final_prob = trained_probability if trained_probability >= 0.35 else 0.0

        logger.info(
            f"[NLP] Trained ML={trained_probability:.3f}, "
            f"Contextual={contextual_score:.3f}, "
            f"Combined={final_prob:.3f}"
        )

        return {
            "scam_probability": final_prob,
            "ml_available": True,
            "label": "SCAM" if final_prob >= 0.50 else "SAFE",
            "confidence": max(final_prob, round(1.0 - final_prob, 4)),
            "sentiment": None,
            "categories": list(categories.keys()),
            "matched_signals": phrases[:10],
            "trained_model_probability": trained_probability,
            "contextual_probability": contextual_score,
        }

    except Exception as e:
        logger.warning(
            f"[NLP] Trained model unavailable: {e}. "
            "Using existing detection engine."
        )

    sentiment_label: Optional[str] = None
    sentiment_conf: float = 0.0

    if model_loaded and _pipeline is not None:
        try:
            result = _pipeline(transcript[:512])[0]
            label = result["label"].upper()
            score = float(result["score"])
            sentiment_label = label
            sentiment_conf = score

            if _IS_SCAM_SPECIFIC_MODEL:
                # Dedicated scam classifier: label indicates SCAM / SPAM directly
                is_scam_label = label in ["SCAM", "SPAM", "FRAUD", "LABEL_1"]
                model_scam_prob = score if is_scam_label else 1.0 - score
                final_prob = round(0.6 * model_scam_prob + 0.4 * contextual_score, 4)
                logger.info(f"[NLP] Dedicated model score: {model_scam_prob:.3f}, contextual: {contextual_score:.3f}")
            else:
                # General sentiment model (e.g. SST-2)
                # CRITICAL: Negative sentiment ALONE is NEVER proof of a scam!
                # Sadness, illness, or normal complaints have contextual_score == 0.0 -> final_prob = 0.0.
                if contextual_score > 0.35 and label == "NEGATIVE":
                    # Negative tone in a coercive context indicates acute intimidation
                    final_prob = round(min(contextual_score + 0.05 * score, 0.98), 4)
                else:
                    final_prob = contextual_score
                logger.info(
                    f"[NLP] Contextual engine: score={final_prob:.3f}, "
                    f"sentiment={sentiment_label} ({sentiment_conf:.2f}), "
                    f"categories={list(categories.keys())}"
                )

            return {
                "scam_probability": final_prob,
                "ml_available": True,
                "label": "SCAM" if final_prob >= 0.50 else "SAFE",
                "confidence": max(final_prob, round(1.0 - final_prob, 4)),
                "sentiment": sentiment_label,
                "categories": list(categories.keys()),
                "matched_signals": phrases[:10],
            }
        except Exception as e:
            logger.error(f"[NLP] Model inference failed: {e}")

    # Fallback / Contextual Engine
    final_prob = contextual_score
    return {
        "scam_probability": final_prob,
        "ml_available": False,
        "label": "SCAM" if final_prob >= 0.50 else "SAFE",
        "confidence": max(final_prob, round(1.0 - final_prob, 4)),
        "sentiment": None,
        "categories": list(categories.keys()),
        "matched_signals": phrases[:10],
    }


async def classify_scam(transcript: str) -> Dict[str, Any]:
    """
    Classify transcript for scam probability.

    Returns:
        {
            "scam_probability": float,   # 0.0 – 1.0
            "ml_available": bool,
            "label": str,               # "SCAM" | "SAFE"
            "confidence": float,
        }
    """
    if not transcript or len(transcript.strip()) < 4:
        return {
            "scam_probability": 0.0,
            "ml_available": False,
            "label": "SAFE",
            "confidence": 1.0,
        }

    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _classify_with_model, transcript),
            timeout=15.0,
        )
        return result
    except asyncio.TimeoutError:
        logger.warning("[NLP] Classification timeout, falling back to contextual rules")
        score, phrases, cats = _analyze_contextual_scam_signals(transcript)
        return {
            "scam_probability": score,
            "ml_available": False,
            "label": "SCAM" if score >= 0.50 else "SAFE",
            "confidence": max(score, round(1.0 - score, 4)),
        }
    except Exception as e:
        logger.error(f"[NLP] Classification error: {e}")
        score, phrases, cats = _analyze_contextual_scam_signals(transcript)
        return {
            "scam_probability": score,
            "ml_available": False,
            "label": "SCAM" if score >= 0.50 else "SAFE",
            "confidence": max(score, round(1.0 - score, 4)),
        }

