from utils.firebase import db
from utils.pinecone import image_index
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import uuid
import io

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def image_embedding(user_id: str, file_bytes:bytes):
           
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)[0].tolist()

    vector_id = str(uuid.uuid4())
    image_index.upsert(vectors=[(vector_id, embedding, {"user_id": user_id, "type": "image"})])

    return {"status": "success", "vector_id": vector_id}  

def text_embedding(user_id:str,text: str):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = clip_model.get_text_features(**inputs)[0].tolist()

    vector_id = str(uuid.uuid4())
    image_index.upsert(vectors=[(vector_id, embedding, {"user_id": user_id, "type": "text", "text": text})])

    query_result = image_index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True,
        filter={"user_id": {"$eq": user_id}, "type": {"$eq": "image"}}
    )

    matches = [
        {
            "id": match["id"],
            "score": match["score"],
            "metadata": match.get("metadata", {})
        }
        for match in query_result.get("matches", [])
    ]

    return {"status": "success", "vector_id": vector_id, "matches": matches}