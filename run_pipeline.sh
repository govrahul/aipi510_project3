#!/usr/bin/env bash
# Runs one step of the pipeline, or all of them in order.
# Usage: ./run_pipeline.sh [download|clean|upload|rf|logreg|test|all]
# (clean downloads the data automatically if it isn't already in Data/)
set -euo pipefail
cd /app

KAGGLE_DATASET="uom190346a/water-quality-and-potability"
RAW_CSV="Data/water_potability.csv"

download() {
  if [ -f "$RAW_CSV" ]; then
    echo "Found $RAW_CSV, skipping Kaggle download."
    return
  fi
  if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
    echo "ERROR: $RAW_CSV not found and KAGGLE_USERNAME / KAGGLE_KEY are not set."
    echo "Either add your Kaggle credentials to .env, or download the CSV from"
    echo "https://www.kaggle.com/datasets/$KAGGLE_DATASET and put it in Data/."
    exit 1
  fi
  echo "Downloading $KAGGLE_DATASET from Kaggle..."
  tmp=$(mktemp -d)
  kaggle datasets download -d "$KAGGLE_DATASET" -p "$tmp" --unzip
  # Move the CSV into place (handles the file having a different name)
  csv=$(find "$tmp" -name '*.csv' | head -n 1)
  if [ -z "$csv" ]; then
    echo "ERROR: no CSV found in the Kaggle download."; exit 1
  fi
  mv "$csv" "$RAW_CSV"
  rm -rf "$tmp"
  echo "Saved $RAW_CSV"
}

clean() {
  download
  python Data/clean_data.py
}
upload() { python Data/add_data_to_cloud.py; }
rf()     { python models/random_forest/random_forest_baseline.py; }
logreg() { python models/logistic_regression/logistic_regression.py; }
test()   { python models/test_best_model.py; }

# W&B can take a minute to show a just-finished run's metrics,
# so in the full pipeline retry the test step a few times
test_with_retry() {
  for attempt in 1 2 3 4 5; do
    if python models/test_best_model.py; then
      return 0
    fi
    if [ "$attempt" -lt 5 ]; then
      echo "W&B may still be processing the new runs. Retrying in 30s ($attempt/5)..."
      sleep 30
    fi
  done
  echo "ERROR: test step failed after 5 attempts."
  exit 1
}

case "${1:-all}" in
  download) download ;;
  clean)  clean ;;
  upload) upload ;;
  rf)     rf ;;
  logreg) logreg ;;
  test)   test ;;
  all)    clean; upload; rf; logreg; test_with_retry ;;
  *) echo "Unknown step '$1'. Use: download | clean | upload | rf | logreg | test | all"; exit 1 ;;
esac
