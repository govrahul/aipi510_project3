"""Prediction helpers for the Streamlit app (no Streamlit imports, so it can be tested alone)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "water_potability_pipeline.joblib"


@dataclass(frozen=True)
class Feature:
    name: str          # column name in the raw dataset
    label: str
    unit: str
    default: float
    min_value: float
    max_value: float
    step: float
    guideline: str     # short, commonly cited reference range
    low: float | None = None   # flag values below this
    high: float | None = None  # flag values above this


# The five home-testable features the random forest was trained on
FEATURES: list[Feature] = [
    Feature("ph", "pH", "", 7.0, 0.0, 14.0, 0.1,
            "Common drinking-water guideline: 6.5–8.5", low=6.5, high=8.5),
    Feature("Hardness", "Hardness", "mg/L", 197.0, 0.0, 1000.0, 1.0,
            "No health-based limit; above ~180 mg/L is considered very hard"),
    Feature("Chloramines", "Chloramines", "ppm", 7.0, 0.0, 20.0, 0.1,
            "Commonly cited limit: up to 4 ppm (mg/L)", high=4.0),
    Feature("Sulfate", "Sulfate", "mg/L", 333.0, 0.0, 1000.0, 1.0,
            "Common secondary guideline: up to 250 mg/L", high=250.0),
    Feature("Conductivity", "Conductivity", "µS/cm", 422.0, 0.0, 2000.0, 1.0,
            "Commonly cited guideline: up to 400 µS/cm", high=400.0),
]
FEATURE_NAMES = [f.name for f in FEATURES]


def model_path() -> Path:
    return Path(os.environ.get("PIPELINE_MODEL_PATH", DEFAULT_MODEL_PATH))


def load_pipeline(path: Path | None = None):
    path = Path(path) if path else model_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    return joblib.load(path)


def outside_guideline(feature: Feature, value: float | None) -> bool:
    if value is None:
        return False
    return (feature.low is not None and value < feature.low) or (
        feature.high is not None and value > feature.high
    )


def build_input_frame(pipeline, rows: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with every raw column the pipeline expects.

    Columns the user didn't provide (or left blank) are NaN; the pipeline's
    imputer fills them with the training-set median.
    """
    expected = list(getattr(pipeline, "feature_names_in_", FEATURE_NAMES))
    frame = pd.DataFrame(index=rows.index)
    for column in expected:
        if column in rows.columns:
            frame[column] = pd.to_numeric(rows[column], errors="coerce")
        else:
            frame[column] = np.nan
    return frame.astype(float)


def predict(pipeline, rows: pd.DataFrame) -> pd.DataFrame:
    """Predict potability for each row. Returns prediction (0/1) and P(potable)."""
    frame = build_input_frame(pipeline, rows)
    classes = list(pipeline.classes_)
    proba = pipeline.predict_proba(frame)[:, classes.index(1)]
    return pd.DataFrame(
        {
            "prediction": pipeline.predict(frame).astype(int),
            "probability_potable": proba,
        },
        index=rows.index,
    )


def predict_one(pipeline, values: dict[str, float | None]) -> tuple[int, float]:
    row = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in values.items()}])
    result = predict(pipeline, row).iloc[0]
    return int(result["prediction"]), float(result["probability_potable"])


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Match uploaded column names to feature names case-insensitively.

    Returns the renamed frame and the list of feature columns that are missing.
    """
    lookup = {name.lower(): name for name in FEATURE_NAMES}
    renamed = df.rename(
        columns={c: lookup[c.strip().lower()] for c in df.columns if c.strip().lower() in lookup}
    )
    missing = [name for name in FEATURE_NAMES if name not in renamed.columns]
    return renamed, missing


def template_csv() -> bytes:
    example = pd.DataFrame(
        [
            {f.name: f.default for f in FEATURES},
            {"ph": 6.8, "Hardness": 150, "Chloramines": 3.5, "Sulfate": 240, "Conductivity": 380},
        ]
    )
    return example.to_csv(index=False).encode()
