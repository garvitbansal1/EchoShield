import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Locate the trained model files
CUR_DIR = Path(__file__).resolve().parent
BACKEND_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = BACKEND_DIR / "ml_training"

MODEL_CANDIDATES = [
    CUR_DIR / "scam_classifier.pkl",
    TRAINING_DIR / "scam_classifier.pkl",
]
VECTORIZER_CANDIDATES = [
    CUR_DIR / "tfidf_vectorizer.pkl",
    TRAINING_DIR / "tfidf_vectorizer.pkl",
]

_model = None
_vectorizer = None


def load_trained_model():
    global _model, _vectorizer

    if _model is None or _vectorizer is None:
        model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
        vec_path = next((p for p in VECTORIZER_CANDIDATES if p.exists()), None)

        if not model_path:
            raise FileNotFoundError(f"Scam model not found in candidates: {MODEL_CANDIDATES}")
        if not vec_path:
            raise FileNotFoundError(f"Vectorizer not found in candidates: {VECTORIZER_CANDIDATES}")

        _model = joblib.load(model_path)
        _vectorizer = joblib.load(vec_path)

        logger.info(f"EchoShield trained scam model loaded successfully from {model_path}")

    return _model, _vectorizer


def predict_scam(text: str):
    if not text or not text.strip():
        return {
            "prediction": "SAFE",
            "scam_probability": 0.0,
            "model_available": True
        }

    model, vectorizer = load_trained_model()

    features = vectorizer.transform([text])

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    return {
        "prediction": "SCAM" if prediction == 1 else "SAFE",
        "scam_probability": round(float(probabilities[1]), 4),
        "model_available": True
    }