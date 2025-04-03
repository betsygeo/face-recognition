import firebase_admin
from fastapi import FastAPI, UploadFile, File, HTTPException
from firebase_admin import credentials, storage, firestore, initialize_app
from pydantic import BaseModel
import requests
import io
from deepface import DeepFace
import numpy as np
import uuid
import os
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone
from PIL import Image
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your Next.js frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
load_dotenv('.env.local')


cred = credentials.Certificate("firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
})

db = firestore.client()
storage_bucket = storage.bucket()

# initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)    
index_name = "face-recognition"
index = pc.Index(index_name)

class NameFaceRequest(BaseModel):
    name: str
   
@app.post("/upload-image/{user_id}")
async def upload_image(user_id: str, file: UploadFile = File(...)):
    try:
        #read and process image
        image_bytes = await file.read()
        image = np.array(Image.open(io.BytesIO(image_bytes)))
        
        results = []
       #detect faces
        try:
            detections = DeepFace.represent(
                img_path=image,
                model_name="Facenet",
                enforce_detection=True,
                detector_backend="retinaface"
            )
        except ValueError as e:
            if "Face could not be detected" in str(e):
       
                return {
                "results": results,
                "unnamed_faces": None,
                
            }
            else:
                raise HTTPException(status_code=500, detail="Face detection failed")
            
            # don't forget to catch this in your front end

        #store in storage for cropping
        blob = storage_bucket.blob(f"users/{user_id}/images/{file.filename}")
        blob.upload_from_string(image_bytes, content_type=file.content_type)
        
        #image metadata
        image_id = str(uuid.uuid4())
        image_ref = db.collection(f"users/{user_id}/images").document(image_id)
        image_ref.set({
            "id": image_id,
            "storage_path": blob.name,
            "url": f"https://firebasestorage.googleapis.com/v0/b/{storage_bucket.name}/o/{blob.name.replace('/', '%2F')}?alt=media",

            "uploaded_at": firestore.SERVER_TIMESTAMP,
            "faces": []
        })

        #results of face detection
      
        for detection in detections:
            face_id = str(uuid.uuid4())
            embedding = detection["embedding"]
            face_coords = detection["facial_area"] #duh
            
            #face matching
            query_res = index.query(vector=embedding, top_k=5, include_metadata=True)
            matched = False
            
            for match in query_res["matches"]:
                if match["score"] > 0.45:
                    matched_id = match["id"]
                    face_doc = db.collection(f"users/{user_id}/faces").document(matched_id).get()
                    
                    if face_doc.exists:
                        results.append({
                            "status": "matched",
                            "face_id": matched_id,
                            "name": face_doc.to_dict().get("name"),
                            "confidence": match["score"]
                        })
                        matched = True
                        break  
            
            if matched:
                continue  

            #new face!
            face_ref = db.collection(f"users/{user_id}/faces").document(face_id)
            face_ref.set({
                "id": face_id,
                "embedding": embedding,
                "name": None,
                "face_coordinates": { # for cropping
                    "x": face_coords["x"],
                    "y": face_coords["y"],
                    "w": face_coords["w"],
                    "h": face_coords["h"]
                },
                "image_refs": [image_id],
                "need_naming": True
            })
            
            index.upsert(vectors=[(face_id, embedding)]) # to pinecone
            results.append({
                "status": "new_face",
                "face_id": face_id,
                "need_naming": True
            })
            
            # faces found in image 
            image_ref.update({
                "faces": firestore.ArrayUnion([face_id])
            })

        return {
            "results": results,
            "unnamed_faces": [r for r in results if r.get("need_naming")],
            "image_id": image_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/name-face/{user_id}/{face_id}")
async def name_face(user_id: str, face_id: str, request: NameFaceRequest):
    """Assign a name to an unnamed face"""
    face_ref = db.collection(f"users/{user_id}/faces").document(face_id)

    if not face_ref.get().exists:
        raise HTTPException(status_code=404, detail="Face not found")

    face_ref.update({
        "name": request.name.strip(),
        "need_naming": False,
        "last_named": firestore.SERVER_TIMESTAMP
    })

    return {"status": "success", "face_id": face_id}

@app.get("/face-crop/{user_id}/{face_id}")
async def get_face_crop(user_id: str, face_id: str):
    """Get cropped image of specific face"""
    try:
        face_data = db.collection(f"users/{user_id}/faces").document(face_id).get().to_dict()
        if not face_data:
            raise HTTPException(status_code=404, detail="Face not found")
        
        image_ref = db.collection(f"users/{user_id}/images").document(face_data["image_refs"][0])
        image_data = image_ref.get().to_dict()
        
        blob = storage_bucket.blob(image_data["storage_path"])
        image_bytes = blob.download_as_bytes()
        
        img = Image.open(io.BytesIO(image_bytes))
        box = face_data["face_coordinates"]
        cropped = img.crop((box["x"], box["y"], box["x"]+box["w"], box["y"]+box["h"]))
        
        byte_arr = io.BytesIO()
        cropped.save(byte_arr, format='JPEG')
        return StreamingResponse(io.BytesIO(byte_arr.getvalue()), media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/person-images/{user_id}/{name}")
async def get_person_images(user_id: str, name: str):
    """Get all images containing a named person"""
    faces = db.collection(f"users/{user_id}/faces").where("name", "==", name.strip()).stream()
    face_ids = [face.id for face in faces]
    
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
                print(image_doc.to_dict())
                seen_images.add(image_id)
                
    return {"images": images}

@app.get("/user-faces/{user_id}")
async def get_user_faces(user_id: str):
    
    faces_ref = db.collection(f"users/{user_id}/faces")
    docs = faces_ref.stream()
    
    return {
        "faces": [
            {"face_id": doc.id, "name": doc.to_dict().get("name")}
            for doc in docs
        ]
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)