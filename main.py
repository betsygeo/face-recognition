from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from models import NameFaceRequest
from services import face_service, embeddings_service

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-image/{user_id}")
async def upload_image(user_id: str, file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        return face_service.detect_faces(image_bytes, user_id, file.filename, file.content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/name-face/{user_id}/{face_id}")
async def name_face(user_id: str, face_id: str, request: NameFaceRequest):
    """Assign a name to an unnamed face"""
    try:
        return face_service.name_face(user_id,face_id,request.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/face-crop/{user_id}/{face_id}")
async def get_face_crop(user_id: str, face_id: str):
    """Get cropped image of specific face"""
    try:
        return face_service.get_face_crop(user_id,face_id)        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/person-images/{user_id}/{name}")
async def get_person_images(user_id: str, name: str):
    """Get all images containing a named person"""
    try:
        return face_service.get_person_images(user_id,name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/user-faces/{user_id}")
async def get_user_faces(user_id: str):
    
    try:
        return face_service.get_user_faces(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/image-embed/{user_id}")
async def upsert_image_embedding(user_id: str, file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        return embeddings_service.image_embedding(user_id, image_bytes)
    except Exception as e:  
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-embed/{user_id}/{text}")
async def upsert_text_embedding(user_id: str, text: str):
    try:
        return embeddings_service.text_embedding(user_id, text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))












if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)