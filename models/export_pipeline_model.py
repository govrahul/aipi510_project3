"""
Export a fitted preprocessing-and-random-forest pipeline for inference.
"""

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
MODEL_PATH = PROJECT_ROOT / "artifacts" / "water-potability-rf:v0" / "model.joblib"
OUTPUT_PATH = (
    PROJECT_ROOT / "models" / "water_potability_pipeline.joblib"
)
TARGET_COLUMN = "Potability"


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Random forest model not found: {MODEL_PATH}")

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

    classifier = joblib.load(MODEL_PATH)
    model_features = getattr(classifier, "feature_names_in_", None)
    if model_features is None:
        raise ValueError(
            f"The random forest at {MODEL_PATH} does not expose its input feature names."
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

    joblib.dump(pipeline, OUTPUT_PATH)
    print(f"Saved preprocessing and random forest pipeline to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
