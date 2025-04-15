from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, storage, firestore
import os

load_dotenv('.env.local')


cred = credentials.Certificate("firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
})

db = firestore.client()
storage_bucket = storage.bucket()