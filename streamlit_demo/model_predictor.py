from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "model_artifacts"


def artifacts_available(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> bool:
    artifact_dir = Path(artifact_dir)
    return (
        (artifact_dir / "classifier.joblib").exists()
        and (artifact_dir / "feature_columns.json").exists()
    )


def load_artifacts(artifact_dir: Path = DEFAULT_ARTIFACT_DIR):
    artifact_dir = Path(artifact_dir)
    classifier = joblib.load(artifact_dir / "classifier.joblib")

    scaler_path = artifact_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    feature_columns = json.loads((artifact_dir / "feature_columns.json").read_text(encoding="utf-8"))

    class_names_path = artifact_dir / "class_names.json"
    if class_names_path.exists():
        class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    elif hasattr(classifier, "classes_"):
        class_names = list(classifier.classes_)
    else:
        class_names = [str(index) for index in range(getattr(classifier, "n_classes_", 0))]

    return classifier, scaler, feature_columns, class_names


def predict_segments(features: pd.DataFrame, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> pd.DataFrame:
    classifier, scaler, feature_columns, class_names = load_artifacts(artifact_dir)
    missing = [column for column in feature_columns if column not in features.columns]
    if missing:
        raise ValueError("Feature table is missing model columns: " + ", ".join(missing))

    x = features[feature_columns].copy()
    x = scaler.transform(x) if scaler is not None else x

    predictions = classifier.predict(x)
    result = features[["segment_id", "start_seconds", "end_seconds"]].copy()
    result["predicted_genre"] = predictions

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(x)
        for index, class_name in enumerate(class_names):
            result[f"prob_{class_name}"] = probabilities[:, index]
        result["confidence"] = np.max(probabilities, axis=1)

    return result


def aggregate_genre_prediction(segment_predictions: pd.DataFrame) -> pd.DataFrame:
    probability_columns = [
        column for column in segment_predictions.columns if column.startswith("prob_")
    ]
    if probability_columns:
        scores = segment_predictions[probability_columns].mean().sort_values(ascending=False)
        return pd.DataFrame(
            {
                "genre": [column.replace("prob_", "", 1) for column in scores.index],
                "score": scores.values,
            }
        )

    counts = segment_predictions["predicted_genre"].value_counts(normalize=True)
    return pd.DataFrame({"genre": counts.index, "score": counts.values})

