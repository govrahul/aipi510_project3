# AIPI 510 Project 3: Water Potability

Water potability (safety of its consumption) is a very common problem around the world. This project creates a model alongside an API endpoint and frontend to predict water quality based on testable features.

All required packages are in `requirements.txt` (NOTE: before installing google-cloud-storage please follow the steps under "Cloud Storage")

## Data

The data was sourced from Kaggle: https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability. NOTE: this data is synthetically generated. All models and any inference done should have this in mind.

To clean the data and split it into a training and test set, put the raw CSV into the Data/ folder and run:

```bash
python Data/clean_data.py
```

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

---
Developers: Rahul Govil and Daniel Yaari