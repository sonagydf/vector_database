from google.cloud import storage
import os

bucket_name = "your-bucket-name"
index_file = "faiss_index.bin"
remote_path = "faiss/faiss_index.bin"

client = storage.Client()
bucket = client.bucket(bucket_name)

def upload_index():
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(index_file)

def download_index():
    blob = bucket.blob(remote_path)
    if blob.exists():
        blob.download_to_filename(index_file)
