import faiss
from google.cloud import storage
def save_index():
    index_file = "faiss_index.bin"  # Define the index file path/name
    faiss.write_index(index, index_file)
    client = storage.Client()
    bucket = client.get_bucket("your-bucket-name")
    blob = bucket.blob("faiss_index.bin")
    blob.upload_from_filename(index_file)