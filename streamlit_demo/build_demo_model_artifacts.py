from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


NON_MODEL_COLUMNS = [
    "filename",
    "genre_from_filename",
    "track_id",
    "segment_id",
    "length",
]


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "processed" / "processed_features_3_sec.csv"


def prepare_training_data(data_csv: Path):
    df = pd.read_csv(data_csv)
    if "label" not in df.columns:
        raise ValueError("Training CSV must contain a label column.")

    existing_non_model = [column for column in NON_MODEL_COLUMNS if column in df.columns]
    x = df.drop(columns=existing_non_model + ["label"]).select_dtypes(include="number")
    y = df["label"]
    if x.empty:
        raise ValueError("No numeric feature columns remain after dropping metadata columns.")

    return x, y, existing_non_model, df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Streamlit demo SVM-RBF inference artifact from the project GTZAN feature CSV."
    )
    parser.add_argument("--data-csv", type=Path, default=default_data_path())
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "model_artifacts",
    )
    args = parser.parse_args()

    data_csv = args.data_csv.resolve()
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    x, y, dropped_columns, df = prepare_training_data(data_csv)
    pipeline = build_pipeline()
    pipeline.fit(x, y)

    joblib.dump(pipeline, artifact_dir / "classifier.joblib")
    write_json(artifact_dir / "feature_columns.json", list(x.columns))
    write_json(artifact_dir / "class_names.json", sorted(y.unique()))
    write_json(
        artifact_dir / "artifact_summary.json",
        {
            "purpose": "Inference artifact for the Streamlit course demo.",
            "warning": "This artifact is trained from the repository feature CSV for demonstration. Replace it with the final team model when available.",
            "source_csv": str(data_csv),
            "row_count": int(len(df)),
            "feature_count": int(x.shape[1]),
            "class_names": sorted(y.unique()),
            "dropped_columns": dropped_columns,
            "model": "Pipeline(StandardScaler, SVC-RBF)",
            "params": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": True,
                "random_state": 42,
            },
        },
    )

    print(f"Saved demo model artifacts to: {artifact_dir}")
    print(f"Feature count: {x.shape[1]}")
    print("Classes:", ", ".join(sorted(y.unique())))


if __name__ == "__main__":
    main()
