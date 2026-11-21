"""
Export a fitted preprocessing-and-random-forest pipeline for inference.
"""

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Data.clean_data import RANDOM_STATE, build_preprocessor

DATA_PATH = PROJECT_ROOT / "Data" / "water_potability.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_PATH = (
    PROJECT_ROOT / "models" / "water_potability_pipeline.joblib"
)
TARGET_COLUMN = "Potability"


def _artifact_version(path: Path) -> int:
    """Version number from a folder name like 'water-potability-rf:v3'."""
    try:
        return int(path.parent.name.rsplit(":v", 1)[1])
    except (IndexError, ValueError):
        return -1


def load_random_forest():
    """Find the trained random forest, in this order:

    1. RF_MODEL_PATH environment variable, if set
    2. the newest local download in artifacts/water-potability-rf:v*/model.joblib
    3. the best random_forest_baseline run on W&B (downloaded automatically)
    """
    env_path = os.environ.get("RF_MODEL_PATH")
    if env_path:
        if not Path(env_path).is_file():
            raise FileNotFoundError(f"RF_MODEL_PATH does not exist: {env_path}")
        print(f"Using random forest from RF_MODEL_PATH: {env_path}")
        return joblib.load(env_path)

    local_models = sorted(
        ARTIFACTS_DIR.glob("water-potability-rf:v*/model.joblib"),
        key=_artifact_version,
    )
    if local_models:
        print(f"Using local random forest: {local_models[-1]}")
        return joblib.load(local_models[-1])

    print("No local random forest found; downloading the best one from W&B...")
    from models.test_best_model import fetch_best_model_from_wandb

    model, _, _, _ = fetch_best_model_from_wandb()
    return model


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset is missing target column {TARGET_COLUMN!r}")

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    preprocessor = build_preprocessor(numeric_features)
    preprocessor.set_output(transform="pandas")
    preprocessor.fit(X_train)

    classifier = load_random_forest()
    model_features = getattr(classifier, "feature_names_in_", None)
    if model_features is None:
        raise ValueError(
            "The random forest does not expose its input feature names."
        )

    processed_features = preprocessor.get_feature_names_out().tolist()
    missing_features = [
        feature for feature in model_features if feature not in processed_features
    ]
    if missing_features:
        raise ValueError(
            "The preprocessing pipeline does not produce model input features: "
            f"{missing_features}"
        )

    feature_selector = ColumnTransformer(
        [("model_features", "passthrough", list(model_features))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    feature_selector.set_output(transform="pandas")
    feature_selector.fit(preprocessor.transform(X_train))

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("feature_selector", feature_selector),
            ("classifier", classifier),
        ]
    )

    expected_predictions = classifier.predict(
        feature_selector.transform(preprocessor.transform(X_test))
    )
    actual_predictions = pipeline.predict(X_test)
    if not (actual_predictions == expected_predictions).all():
        raise RuntimeError("Exported pipeline predictions differ from the classifier.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PATH)
    print(f"Saved preprocessing and random forest pipeline to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
