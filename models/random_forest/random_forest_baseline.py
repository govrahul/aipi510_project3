"""Train a random forest on train.csv from Google Cloud Storage and log it to Weights & Biases.

Same steps as water_potability_wandb.ipynb, as a script.

Before running (one time):
    gcloud auth application-default login    # lets the script read the GCS bucket
    wandb login                               # or set the WANDB_API_KEY environment variable

Run:
    python train.py                       # uses config.yaml
    python train.py --config my_config.yaml
"""
import argparse

import joblib
import matplotlib

matplotlib.use("Agg")   # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import yaml
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split

# 1. Which config file to use
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

# 2. Load the settings
with open(args.config) as f:
    config = yaml.safe_load(f)

SEED = config["seed"]

# 3. Download train.csv from Google Cloud Storage
client = storage.Client(project=config["gcs"]["project"])
bucket = client.bucket(config["gcs"]["bucket"])
bucket.blob(config["gcs"]["file"]).download_to_filename("train.csv")

df = pd.read_csv("train.csv")
print("Data:", df.shape)

# 4. Split into train and validation
X = df[config["data"]["features"]]
y = df[config["data"]["target"]]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=config["split"]["val_size"], random_state=SEED, stratify=y
)
print("Train:", X_train.shape, " Validation:", X_val.shape)

# 5. Start a W&B run (the whole config is saved with it)
run = wandb.init(
    entity=config["wandb"]["entity"],
    project=config["wandb"]["project"],
    name=config["wandb"]["run_name"],
    config=config,
)

# 6. Train the random forest
model = RandomForestClassifier(**config["model"], random_state=SEED)
model.fit(X_train, y_train)

# 7. Evaluate and log metrics
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

# 8. Plots: confusion matrix and feature importance
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ConfusionMatrixDisplay.from_predictions(y_val, val_pred, ax=axes[0], colorbar=False,
                                        display_labels=["not potable", "potable"])
axes[0].set_title("Confusion matrix (validation)")

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importance.plot.barh(ax=axes[1])
axes[1].set_title("Feature importance")

plt.tight_layout()
fig.savefig("plots.png")
wandb.log({"plots": wandb.Image(fig)})

# 9. Save the model and upload it to W&B
joblib.dump(model, "model.joblib")

artifact = wandb.Artifact("water-potability-rf", type="model")
artifact.add_file("model.joblib")
artifact.add_file(args.config)
run.log_artifact(artifact)

run.finish()
print("Saved model.joblib and plots.png")
