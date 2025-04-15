from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from firebase_admin import firestore
from deepface import DeepFace
from PIL import Image
import numpy as np
import uuid
import io
from utils.firebase import db, storage_bucket
from utils.pinecone import face_index


def detect_faces(image_bytes: bytes, user_id: str, filename: str, content_type: str):
    image = np.array(Image.open(io.BytesIO(image_bytes)))

    try:
        detections = run_face_detection(image)
    except ValueError as e:
        if "Face could not be detected" in str(e):
            return {"results": [], "unnamed_faces": None}
        else:
            raise HTTPException(status_code=500, detail="Face detection failed")

    image_id = upload_image_to_firebase(user_id, image_bytes, filename, content_type)

    results = []
    for detection in detections:
        result = store_face_and_match_results(detection, user_id, image_id)
        results.append(result)

    return {
        "results": results,
        "unnamed_faces": [r for r in results if r.get("need_naming")],
        "image_id": image_id
    }

def run_face_detection(image):
    return DeepFace.represent(
        img_path=image,
        model_name="Facenet",
        enforce_detection=True,
        detector_backend="retinaface"
    )

def upload_image_to_firebase(user_id: str, image_bytes: bytes, filename: str, content_type: str):
    blob = storage_bucket.blob(f"users/{user_id}/images/{filename}")
    blob.upload_from_string(image_bytes, content_type=content_type)

    image_id = str(uuid.uuid4())
    blob_url = f"https://firebasestorage.googleapis.com/v0/b/{storage_bucket.name}/o/{blob.name.replace('/', '%2F')}?alt=media"

    db.collection(f"users/{user_id}/images").document(image_id).set({
        "id": image_id,
        "storage_path": blob.name,
        "url": blob_url,
        "uploaded_at": firestore.SERVER_TIMESTAMP,
        "faces": []
    })

    return image_id

def match_face(embedding, user_id):
    query = face_index.query(vector=embedding, top_k=5, include_metadata=True)
    for match in query["matches"]:
        if match["score"] > 0.45:
            face_doc = db.collection(f"users/{user_id}/faces").document(match["id"]).get()
            if face_doc.exists:
                return {
                    "status": "matched",
                    "face_id": match["id"],
                    "name": face_doc.to_dict().get("name"),
                    "confidence": match["score"]
                }
    return None

def store_face_and_match_results(detection, user_id: str, image_id: str):
    face_id = str(uuid.uuid4())
    embedding = detection["embedding"]
    face_coords = detection["facial_area"]

    match_result = match_face(embedding, user_id)
    if match_result:
        return match_result

    
    db.collection(f"users/{user_id}/faces").document(face_id).set({
        "id": face_id,
        "embedding": embedding,
        "name": None,
        "face_coordinates": {"x": face_coords["x"],
                    "y": face_coords["y"],
                    "w": face_coords["w"],
                    "h": face_coords["h"]},
        "image_refs": [image_id],
        "need_naming": True
    })

    face_index.upsert(vectors=[(face_id, embedding)])

    db.collection(f"users/{user_id}/images").document(image_id).update({
        "faces": firestore.ArrayUnion([face_id])
    })

    return {
        "status": "new_face",
        "face_id": face_id,
        "need_naming": True
    }

def name_face(user_id: str, face_id: str, name: str):
    face_ref = db.collection(f"users/{user_id}/faces").document(face_id)

    if not face_ref.get().exists:
        raise HTTPException(status_code=404, detail="Face not found")

    face_ref.update({
        "name": name.strip(),
        "need_naming": False,
        "last_named": firestore.SERVER_TIMESTAMP
    })

    return {"status": "success", "face_id": face_id}

def get_face_crop(user_id: str, face_id: str): 

    face_data = db.collection(f"users/{user_id}/faces").document(face_id).get().to_dict()
    if not face_data:
        raise HTTPException(status_code=404, detail="Face not found")

    image_id = face_data["image_refs"][0]
    image_data = db.collection(f"users/{user_id}/images").document(image_id).get().to_dict()

    blob = storage_bucket.blob(image_data["storage_path"])
    image_bytes = blob.download_as_bytes()

    img = Image.open(io.BytesIO(image_bytes))
    box = face_data["face_coordinates"]
    cropped = img.crop((box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]))

    byte_arr = io.BytesIO()
    cropped.save(byte_arr, format='JPEG')
    return StreamingResponse(io.BytesIO(byte_arr.getvalue()), media_type="image/jpeg")

def get_person_images(user_id: str, name: str):    
    face_query = db.collection(f"users/{user_id}/faces").where("name", "==", name.strip()).stream()

    face_ids = [face.id for face in face_query]
    if not face_ids:
        raise HTTPException(status_code=404, detail="Person not found")

    images = []
    seen_images = set()

    for face_id in face_ids:
        face_doc = db.collection(f"users/{user_id}/faces").document(face_id).get()        
        for image_id in face_doc.to_dict().get("image_refs", []):
            if image_id not in seen_images:
                image_doc = db.collection(f"users/{user_id}/images").document(image_id).get()
                images.append(image_doc.to_dict())
                seen_images.add(image_id)

    return {"images": images}


def get_user_faces(user_id: str):   
    faces_ref = db.collection(f"users/{user_id}/faces")
    docs = faces_ref.stream()

    return {
        "faces": [
            {"face_id": doc.id, "name": doc.to_dict().get("name")}
            for doc in docs
        ]
    }
