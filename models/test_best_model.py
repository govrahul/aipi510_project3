from pathlib import Path

import joblib
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, f1_score

WANDB_ENTITY = "water-potability-510"
WANDB_PROJECT = "water-potability-510-project3"
MODEL_RUN_NAME = "random_forest_baseline"
TARGET_COLUMN = "Potability"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_PATH = PROJECT_ROOT / "Data" / "test.csv"


def fetch_best_model_from_wandb():
    """Fetch the highest-validation-F1 model artifact from the named W&B run."""
    print(f"Finding W&B run {MODEL_RUN_NAME!r}...")
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    matching_runs = [run for run in runs if run.name == MODEL_RUN_NAME]

    if not matching_runs:
        raise ValueError(
            f"No run named {MODEL_RUN_NAME!r} found in "
            f"{WANDB_ENTITY}/{WANDB_PROJECT}."
        )

    runs_with_scores = [
        (run, run.summary.get("val_f1"))
        for run in matching_runs
        if run.summary.get("val_f1") is not None
    ]
    if not runs_with_scores:
        raise ValueError(
            f"Run {MODEL_RUN_NAME!r} has no 'val_f1' metric to identify the best run."
        )

    best_run, val_f1 = max(runs_with_scores, key=lambda item: float(item[1]))
    print(f"Selected run: {best_run.name} (ID: {best_run.id}, val_f1: {float(val_f1):.4f})")

    model_artifacts = [
        artifact
        for artifact in best_run.logged_artifacts()
        if artifact.type.lower() == "model"
    ]
    if not model_artifacts:
        raise ValueError(f"Run {best_run.name!r} has no logged model artifact.")
    if len(model_artifacts) > 1:
        raise ValueError(
            f"Run {best_run.name!r} has multiple model artifacts; "
            "cannot choose one unambiguously."
        )

    model_artifact = model_artifacts[0]
    artifact_dir = Path(model_artifact.download())
    model_files = sorted(
        path
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".joblib", ".pkl"}
    )
    if not model_files:
        raise FileNotFoundError(
            f"No .joblib or .pkl model file found in artifact {model_artifact.name!r}."
        )
    if len(model_files) > 1:
        raise ValueError(
            f"Artifact {model_artifact.name!r} contains multiple model files; "
            "cannot choose one unambiguously."
        )

    model = joblib.load(model_files[0])
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None:
        model_features = best_run.config.get("data", {}).get("features")
    if model_features is None:
        raise ValueError(
            "The model artifact does not expose feature names and the W&B run "
            "config has no data.features entry."
        )
    feature_columns = list(model_features)
    if not feature_columns:
        raise ValueError("The selected model has no input feature columns.")

    print(f"Downloaded model artifact: {model_artifact.name}")
    return model, feature_columns, best_run, model_artifact


def main():
    if not TEST_DATA_PATH.is_file():
        raise FileNotFoundError(f"Test dataset not found: {TEST_DATA_PATH}")

    test_df = pd.read_csv(TEST_DATA_PATH)
    model, feature_columns, best_run, model_artifact = fetch_best_model_from_wandb()

    required_columns = [*feature_columns, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in test_df]
    if missing_columns:
        raise ValueError(
            f"Test dataset is missing columns required for evaluation: {missing_columns}"
        )

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    print("Evaluating model on the test dataset...")
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="binary")

    print("\n" + "=" * 40)
    print(" FINAL TEST RESULTS ")
    print("=" * 40)
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test F1 Score : {test_f1:.4f}")
    print("=" * 40)

    evaluation_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        job_type="evaluation",
        name=f"evaluate-{best_run.name}",
        config={
            "source_run_id": best_run.id,
            "source_run_name": best_run.name,
            "source_model_artifact": model_artifact.name,
        },
    )
    try:
        evaluation_run.use_artifact(model_artifact)
        evaluation_run.log(
            {"test_accuracy": test_accuracy, "test_f1": test_f1}
        )
        print(f"Evaluation run logged to W&B: {evaluation_run.url}")
    finally:
        evaluation_run.finish()


if __name__ == "__main__":
    main()
