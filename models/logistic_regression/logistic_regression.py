"""Train and evaluate a logistic-regression model using Google Cloud Storage and W&B.

From the repository root:
    python models/logistic_regression/logistic_regression.py
    python models/logistic_regression/logistic_regression.py --config path/to/config.yaml
"""
import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import yaml
from google.cloud import storage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=script_dir / "config.yaml",
        help="Path to the experiment configuration (defaults to this script's config.yaml).",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    seed = config["seed"]

    client = storage.Client(project=config["gcs"]["project"])
    bucket = client.bucket(config["gcs"]["bucket"])
    train_csv = script_dir / "train.csv"
    bucket.blob(config["gcs"]["file"]).download_to_filename(str(train_csv))

    df = pd.read_csv(train_csv)
    print("Data:", df.shape)

    features = config["data"]["features"]
    target = config["data"]["target"]
    X = df[features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config["split"]["val_size"],
        random_state=seed,
        stratify=y,
    )
    print("Train:", X_train.shape, " Validation:", X_val.shape)

    model_config = dict(config["model"])
    model_config.setdefault("random_state", seed)
    run = wandb.init(
        entity=config["wandb"]["entity"],
        project=config["wandb"]["project"],
        name=config["wandb"]["run_name"],
        config=config,
    )

    try:
        model = LogisticRegression(**model_config)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "train_f1": f1_score(y_train, train_pred),
            "val_accuracy": accuracy_score(y_val, val_pred),
            "val_f1": f1_score(y_val, val_pred),
            "val_roc_auc": roc_auc_score(y_val, val_proba),
        }
        wandb.log(metrics)

        for name, value in metrics.items():
            print(f"{name:15} {value:.3f}")
        print()
        print(classification_report(y_val, val_pred, target_names=["not potable", "potable"]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_val,
            val_pred,
            ax=axes[0],
            colorbar=False,
            display_labels=["not potable", "potable"],
        )
        axes[0].set_title("Confusion matrix (validation)")

        coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values()
        coefficients.plot.barh(ax=axes[1])
        axes[1].set_title("Feature coefficients")

        plt.tight_layout()
        plots_path = script_dir / "plots.png"
        fig.savefig(plots_path)
        wandb.log({"plots": wandb.Image(fig)})
        plt.close(fig)

        model_path = script_dir / "model.joblib"
        joblib.dump(model, model_path)

        artifact = wandb.Artifact("water-potability-logreg", type="model")
        artifact.add_file(str(model_path))
        artifact.add_file(str(config_path))
        run.log_artifact(artifact)
    finally:
        run.finish()

    print(f"Saved {script_dir / 'model.joblib'} and {script_dir / 'plots.png'}")


if __name__ == "__main__":
    main()
