import pinecone
from pinecone import Pinecone
import os


# initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)    
face_index = pc.Index("face-recognition")
image_index = pc.Index("image-embeddings")