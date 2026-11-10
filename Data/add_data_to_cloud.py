from google.cloud import storage
import os

BUCKET_NAME = 'aipi510-water-bucket'
DATA_DIR = 'Data'
FILES_TO_UPLOAD = ['water_potability.csv', 'train.csv', 'test.csv']

def upload_files_to_gcs(bucket_name, data_dir, files):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for file_name in files:
        local_path = os.path.join(data_dir, file_name)

        if not os.path.exists(local_path):
            print(f"File {local_path} does not exist.")
            continue

        blob = bucket.blob(file_name)
        blob.upload_from_filename(local_path)

        print(f"File {file_name} uploaded to {bucket_name}.")

if __name__ == "__main__":
    upload_files_to_gcs(BUCKET_NAME, DATA_DIR, FILES_TO_UPLOAD)