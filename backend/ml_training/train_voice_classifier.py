"""
EchoShield Voice Classification Model Training & Evaluation Pipeline.

Phases 3 & 4:
  1. Loads normalized audio samples from raw dataset folders:
     - Class 0: AI_GENERATED (200 samples)
     - Class 1: ROBOCALL_IVR (200 samples)
     - Class 2: NATURAL_HUMAN (200 samples)
  2. Extracts 91-dimensional acoustic feature vectors per sample.
  3. Prepares stratified 70% Train / 15% Validation / 15% Test splits (preventing leakage).
  4. Robust scaling with RobustScaler.
  5. Trains and compares 3 candidate models:
     - Model A: HistGradientBoostingClassifier
     - Model B: RandomForestClassifier
     - Model C: MLPClassifier (Neural Network)
  6. Selects top performer and calibrates probability outputs with CalibratedClassifierCV.
  7. Comprehensive evaluation on held-out Test set:
     - Accuracy, Precision, Recall, F1 (macro and per-class)
     - Full 3x3 Confusion Matrix
     - Uncertainty thresholding
  8. Serializes artifacts to disk for backend production serving.
"""

import os
import sys
import json
import time
import pickle
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_features_from_audio, load_audio_normalized, FEATURE_NAMES, NUM_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
FEATURES_CACHE = os.path.join(BASE_DIR, "data", "extracted_features.npz")
MODEL_OUT = os.path.join(BASE_DIR, "voice_classifier.pkl")
SCALER_OUT = os.path.join(BASE_DIR, "voice_scaler.pkl")
METADATA_OUT = os.path.join(BASE_DIR, "model_metadata.json")

CLASS_NAMES = ["AI_GENERATED", "ROBOCALL_IVR", "NATURAL_HUMAN"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def extract_or_load_dataset():
    """Extract features for all raw dataset files or load from npz cache if present."""
    if os.path.exists(FEATURES_CACHE):
        logger.info(f"Loading cached extracted features from {FEATURES_CACHE}...")
        data = np.load(FEATURES_CACHE, allow_pickle=True)
        return data["X"], data["y"], data["filenames"]

    logger.info("Extracting acoustic features from raw audio datasets...")
    X_list = []
    y_list = []
    fname_list = []

    dirs = [
        ("ai_generated", "AI_GENERATED"),
        ("robocall_ivr", "ROBOCALL_IVR"),
        ("natural_human", "NATURAL_HUMAN"),
    ]

    for sub_dir, class_label in dirs:
        dir_path = os.path.join(DATA_RAW, sub_dir)
        if not os.path.exists(dir_path):
            logger.warning(f"Directory {dir_path} not found!")
            continue

        files = sorted([f for f in os.listdir(dir_path) if f.endswith(".wav")])
        class_idx = CLASS_TO_IDX[class_label]
        logger.info(f"Extracting features for {len(files)} files in {class_label}...")

        success = 0
        for i, fname in enumerate(files):
            p = os.path.join(dir_path, fname)
            try:
                y, sr = load_audio_normalized(p)
                if y is not None and len(y) >= int(sr * 0.25):
                    feats = extract_features_from_audio(y, sr)
                    if np.any(np.abs(feats) > 1e-6):
                        X_list.append(feats)
                        y_list.append(class_idx)
                        fname_list.append(fname)
                        success += 1
            except Exception as e:
                logger.warning(f"Error extracting {fname}: {e}")

        logger.info(f"Extracted {success}/{len(files)} valid samples for {class_label}.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    filenames = np.array(fname_list)

    # Save cache
    os.makedirs(os.path.dirname(FEATURES_CACHE), exist_ok=True)
    np.savez_compressed(FEATURES_CACHE, X=X, y=y, filenames=filenames)
    logger.info(f"Feature extraction complete. Total samples: {len(y)}. Cached to {FEATURES_CACHE}.")
    return X, y, filenames


def train_and_evaluate():
    X, y, filenames = extract_or_load_dataset()
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Check class distribution
    for c_idx, c_name in enumerate(CLASS_NAMES):
        count = int(np.sum(y == c_idx))
        pct = (count / len(y)) * 100
        logger.info(f"  Class {c_idx} ({c_name}): {count} samples ({pct:.1f}%)")

    # 1. Stratified Train / Val / Test Split: 70% Train, 15% Val, 15% Test
    # Random seed = 42 for reproducibility
    X_train_val, X_test, y_train_val, y_test, f_train_val, f_test = train_test_split(
        X, y, filenames, test_size=0.15, stratify=y, random_state=42
    )

    # 15% / 85% = ~0.17647 of train_val to get exactly 15% of total as val
    X_train, X_val, y_train, y_val, f_train, f_val = train_test_split(
        X_train_val, y_train_val, f_train_val, test_size=0.17647, stratify=y_train_val, random_state=42
    )

    logger.info(f"Split sizes: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

    # 2. Scaling: Fit RobustScaler on training data only (preventing leakage)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model Comparison
    models = {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=6,
            min_samples_leaf=5,
            l2_regularization=1.0,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "MLP_NeuralNet": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=0.01,
            max_iter=300,
            early_stopping=True,
            random_state=42
        )
    }

    comparison_results = {}
    best_model_name = None
    best_val_f1 = -1.0
    best_model = None

    logger.info("--- Comparing Candidate Models on Validation Set ---")
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train_scaled, y_train)
        train_time = time.time() - t0

        val_preds = clf.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_preds)
        val_p, val_r, val_f1, _ = precision_recall_fscore_support(y_val, val_preds, average="macro", zero_division=0)

        comparison_results[name] = {
            "val_accuracy": round(float(val_acc), 4),
            "val_precision_macro": round(float(val_p), 4),
            "val_recall_macro": round(float(val_r), 4),
            "val_f1_macro": round(float(val_f1), 4),
            "train_time_sec": round(float(train_time), 3)
        }
        logger.info(f"Model [{name}]: Val Accuracy={val_acc*100:.2f}%, Val Macro F1={val_f1*100:.2f}%, Train Time={train_time:.2f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_name = name
            best_model = clf

    logger.info(f"Best Candidate Model Selected: {best_model_name} (Val F1: {best_val_f1*100:.2f}%)")

    # 4. Calibrate Probabilities with Platt Scaling (CalibratedClassifierCV with 5-fold CV)
    logger.info("Training and calibrating final model using 5-fold cross-validation...")
    # Scale combined train+val for final cross-validated calibration
    X_train_val_scaled = scaler.transform(X_train_val)
    
    # Clone best model architecture for CV calibration
    if best_model_name == "HistGradientBoosting":
        final_base = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=6,
            min_samples_leaf=5,
            l2_regularization=1.0,
            random_state=42
        )
    elif best_model_name == "RandomForest":
        final_base = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    else:
        final_base = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=0.01,
            max_iter=300,
            early_stopping=True,
            random_state=42
        )

    calibrated_clf = CalibratedClassifierCV(estimator=final_base, method="sigmoid", cv=5)
    calibrated_clf.fit(X_train_val_scaled, y_train_val)

    # 5. Held-Out Test Set Evaluation
    logger.info("--- Evaluating Calibrated Model on Held-Out Test Set (Unseen) ---")
    test_probs = calibrated_clf.predict_proba(X_test_scaled)
    test_preds = np.argmax(test_probs, axis=1)

    test_acc = accuracy_score(y_test, test_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, test_preds, average="macro", zero_division=0)
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(y_test, test_preds, average=None, zero_division=0)
    cm = confusion_matrix(y_test, test_preds)

    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    logger.info(f"Test Macro Precision: {p_macro*100:.2f}%")
    logger.info(f"Test Macro Recall:    {r_macro*100:.2f}%")
    logger.info(f"Test Macro F1-Score:  {f1_macro*100:.2f}%")

    per_class_metrics = {}
    for i, c_name in enumerate(CLASS_NAMES):
        per_class_metrics[c_name] = {
            "precision": round(float(p_class[i]), 4),
            "recall": round(float(r_class[i]), 4),
            "f1_score": round(float(f1_class[i]), 4),
            "support": int(support_class[i])
        }
        logger.info(f"  Class [{c_name}]: P={p_class[i]*100:.1f}%, R={r_class[i]*100:.1f}%, F1={f1_class[i]*100:.1f}% (Support: {support_class[i]})")

    logger.info(f"Confusion Matrix (Rows=True, Cols=Predicted):\n{cm}")

    # False Positive Analysis
    # Class 2 (NATURAL_HUMAN) misclassified as AI (Class 0)
    human_as_ai = int(cm[2, 0])
    # Class 1 (ROBOCALL_IVR) misclassified as AI (Class 0)
    ivr_as_ai = int(cm[1, 0])
    # Class 0 (AI_GENERATED) misclassified as Human (Class 2)
    ai_as_human = int(cm[0, 2])

    logger.info("Critical False Positive Check:")
    logger.info(f"  - Natural Human falsely flagged as AI: {human_as_ai} cases")
    logger.info(f"  - Robocall/IVR falsely flagged as AI:  {ivr_as_ai} cases")
    logger.info(f"  - AI Clone falsely classified as Human: {ai_as_human} cases")

    # Uncertainty Analysis: Predictions with confidence < 0.55
    max_probs = np.max(test_probs, axis=1)
    uncertain_mask = max_probs < 0.55
    uncertain_count = int(np.sum(uncertain_mask))
    logger.info(f"Uncertainty flag (confidence < 0.55): {uncertain_count}/{len(y_test)} test cases")

    # 6. Save Artifacts
    logger.info(f"Saving trained model pipeline to {MODEL_OUT} and scaler to {SCALER_OUT}...")
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(calibrated_clf, f)

    with open(SCALER_OUT, "wb") as f:
        pickle.dump(scaler, f)

    # Copy to backend/app/ml/ for production serving
    app_ml_dir = os.path.join(BASE_DIR, "..", "app", "ml")
    os.makedirs(app_ml_dir, exist_ok=True)
    with open(os.path.join(app_ml_dir, "voice_classifier.pkl"), "wb") as f:
        pickle.dump(calibrated_clf, f)
    with open(os.path.join(app_ml_dir, "voice_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "model_name": best_model_name,
        "calibrated": True,
        "calibration_method": "Platt Scaling (Sigmoid)",
        "feature_count": NUM_FEATURES,
        "feature_names": FEATURE_NAMES,
        "classes": CLASS_NAMES,
        "test_metrics": {
            "accuracy": round(float(test_acc), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class_metrics,
            "false_positives": {
                "human_falsely_flagged_as_ai": human_as_ai,
                "ivr_falsely_flagged_as_ai": ivr_as_ai,
                "ai_falsely_passed_as_human": ai_as_human
            },
            "uncertain_cases_test": uncertain_count
        },
        "model_comparison": comparison_results,
        "dataset_summary": {
            "total_samples": len(y),
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "test_samples": len(y_test),
            "class_distribution": {
                c_name: int(np.sum(y == i)) for i, c_name in enumerate(CLASS_NAMES)
            }
        },
        "training_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    with open(METADATA_OUT, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(app_ml_dir, "voice_model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model metadata successfully saved to {METADATA_OUT}.")
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
