from google.cloud import firestore
import os

# Set this before import or use GOOGLE_APPLICATION_CREDENTIALS env var
os.environ["GOOGLE_CLOUD_PROJECT"] = "your-project-id"

db = firestore.Client()

def store_metadata(vector_id: int, text: str, metadata: dict):
    doc_ref = db.collection("vectors").document(str(vector_id))
    doc_ref.set({
        "text": text,
        "metadata": metadata
    })

def fetch_metadata(vector_id: int):
    doc = db.collection("vectors").document(str(vector_id)).get()
    return doc.to_dict() if doc.exists else None
