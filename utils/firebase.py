import base64
import json
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, storage, firestore
import os

load_dotenv('.env.local')

cred_json = base64.b64decode(os.environ["FIREBASE_CREDENTIALS"]).decode("utf-8")
cred_dict = json.loads(cred_json)

cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
})

db = firestore.client()
storage_bucket = storage.bucket()