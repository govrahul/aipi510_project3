# AIPI 510 Project 3: Water Potability

Water potability (safety of its consumption) is a very common problem around the world. This project creates a model alongside an API endpoint and frontend to predict water quality based on testable features.

All required packages are in `requirements.txt` (NOTE: before installing google-cloud-storage please follow the steps under "Cloud Storage")

## Running with Docker (recommended)

The whole pipeline (download data → clean → upload to cloud → random forest → logistic regression → test best model) runs in one Docker container. **Each user provides their own Kaggle, Weights & Biases, and Google Cloud credentials**; no keys are included in this repo or the image.

### 1. Install Docker
Install [Docker Desktop](https://www.docker.com/products/docker-desktop/), open it, and leave it running. Check it works:
```bash
docker --version
```

### 2. Get the code
```bash
git clone https://github.com/govrahul/aipi510_project3.git
cd aipi510_project3
```

### 3. Get your Kaggle API key
At kaggle.com: **Settings → API → Create New Token**. Note your username and key. The container uses these to download the dataset automatically.

### 4. Get your Weights & Biases API key
At wandb.ai: **profile icon → User Settings → API Keys → Create new API key**. Copy it right away (it's only shown once). Your **entity** is your W&B username, or your team name if logging to a team.

### 5. Set up Google Cloud
At [console.cloud.google.com](https://console.cloud.google.com):
1. Create or choose a project and note its **Project ID**.
2. **Cloud Storage → Buckets → Create**: pick a globally unique bucket name.
3. **IAM & Admin → Service Accounts → Create**: give it the **Storage Object Admin** role.
4. Open the service account → **Keys → Add Key → Create new key → JSON**.
5. Move the downloaded key to `~/keys/gcp-key.json`:
   - **Mac/Linux:** `mkdir -p ~/keys && mv ~/Downloads/<key-file>.json ~/keys/gcp-key.json`
   - **Windows (PowerShell):** `mkdir $HOME\keys; Move-Item $HOME\Downloads\<key-file>.json $HOME\keys\gcp-key.json`

### 6. Create your `.env` file
- **Mac/Linux:** `cp .env.example .env` then `nano .env`
- **Windows:** `Copy-Item .env.example .env` then `notepad .env`

Fill in your own values (no quotes or spaces), and leave `GOOGLE_APPLICATION_CREDENTIALS` as is:
```
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KEY=your-kaggle-key
WANDB_API_KEY=your-wandb-key
WANDB_ENTITY=your-wandb-username-or-team
WANDB_PROJECT=water-potability-510-project3
GCP_PROJECT=your-gcp-project-id
GCS_BUCKET=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-key.json
```
`.env` is git-ignored and excluded from the Docker image. Never commit it.

### 7. Build the image
```bash
docker build -t water-potability .
```

### 8. Run the full pipeline
**Mac/Linux:**
```bash
docker run --rm --env-file .env \
  -v "$(pwd)/Data:/app/Data" \
  -v "$HOME/keys/gcp-key.json:/secrets/gcp-key.json:ro" \
  water-potability
```
**Windows (PowerShell, one line):**
```powershell
docker run --rm --env-file .env -v "${PWD}\Data:/app/Data" -v "$HOME\keys\gcp-key.json:/secrets/gcp-key.json:ro" water-potability
```
The run ends by printing **FINAL TEST RESULTS**. Metrics, plots, and models are logged to your W&B project; data is uploaded to your bucket; CSVs are saved to your local `Data/` folder.

### 9. (Optional) Run a single step
Add a step name to the end of the `docker run` command:

| Step | What it does |
| :--- | :--- |
| `download` | Downloads the dataset from Kaggle (skipped if `Data/water_potability.csv` exists) |
| `clean` | Downloads if needed, then creates `Data/train.csv` and `Data/test.csv` |
| `upload` | Uploads the CSVs to your bucket |
| `rf` | Trains the random forest baseline |
| `logreg` | Trains the logistic regression |
| `test` | Evaluates the best random forest run on the test set |

### Troubleshooting
- **"Cannot connect to the Docker daemon":** Docker Desktop isn't running.
- **`CERTIFICATE_VERIFY_FAILED` during `docker build`:** some networks interfere with Docker's secure connections. Try another network (e.g. a phone hotspot) or disconnect from VPN, then rebuild and run.
- **Google 403 / permission denied:** check that the service account has **Storage Object Admin** and that `GCS_BUCKET` matches your bucket name exactly.
- **Kaggle 401 Unauthorized:** check `KAGGLE_USERNAME` and `KAGGLE_KEY`, or create a new token.
- **Test step says a run has no `val_f1`:** W&B can take a minute to process new runs. The full pipeline retries automatically; when running `test` on its own, wait a minute and run it again.
- **Changed `.env`?** Just run again. Rebuild (step 7) only after changing code or `requirements.txt`.

## Running locally (without Docker)

The sections below describe running each script directly with Python.

## Data

The data was sourced from Kaggle: https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability. NOTE: this data is synthetically generated. All models and any inference done should have this in mind.

To clean the data and split it into a training and test set, put the raw CSV into the Data/ folder and run:

```bash
python Data/clean_data.py
```

To reproduce the exploratory data analysis plots and cleaned dataset outputs from the notebook, run:

```bash
python Data/cleaning_and_EDA.py
```

The script writes three plots, `water_potability_clean.csv`, and `feature_metadata.json` to `Data/eda_output/`.

## Cloud Storage

All data was added to a Google Cloud database for storage. To run scripts involving GCP, follow these steps:

1. Create a bucket in your Google Cloud project
2. Create a service account with a **Storage Object Admin** role and generate a key
3. Setup local authentication:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-key.json"
pip install google-cloud-storage
```

To push data to cloud, first clean the data (above) and then run:
```bash
python Data/add_data_to_cloud.py
```

## Modeling

All modeling can be run with:

```bash
python models/logistic_regression/logistic_regression.py
python models/random_forest/random_forest_baseline.py
```

Their configurations are in their respective `config.yaml` files. Both files utilize Weights & Biases for model logging which requires an API key. Instructions for this, as well as local setup with it, can be done with the command `wandb login`.

To export the random forest together with the preprocessing steps for inference, run:

```bash
python models/export_pipeline_model.py
```

This writes `models/water_potability_pipeline.joblib`.

To evaluate our best model, the random forest, run:

```bash
python models/test_best_model.py
```

The model evaluation results are shown below:

| Model | Validation F1 | Validation Accuracy |
| :---: | :---: | :---: |
| Random Forest | .51507 | .66221 |
| Logistic Regression | .42358 | .49618 |

The best model was the **random forest** model, and its test metrics were F1 = 0.5219 and accuracy = 0.6509.

---
Developers: Rahul Govil and Daniel Yaari
