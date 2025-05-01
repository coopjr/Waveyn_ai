# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

from config import Settings
from index_manager import HnswIndex

app = FastAPI(title="Advisor Semantic Search")

# Load embedding model (all-MiniLM-L6-v2)&#8203;:contentReference[oaicite:15]{index=15}
model = SentenceTransformer(Settings.MODEL_NAME)
vector_index = HnswIndex()

class Advisor(BaseModel):
    id: int
    name: str
    bio: Optional[str] = ""
    skills: Optional[List[str]] = []

class Query(BaseModel):
    query: str
    k: Optional[int] = Settings.TOP_K_DEFAULT

class TextIn(BaseModel):
    text: str

@app.post("/advisor", summary="Add or update an advisor by ID")
def add_update_advisor(advisor: Advisor):
    # Concatenate advisor fields
    text = " ".join([advisor.name, advisor.bio, " ".join(advisor.skills)])
    # Compute embedding
    embedding = model.encode(text)
    vector = np.array(embedding, dtype=np.float32)
    try:
        vector_index.add_or_update(advisor.id, vector)
        return {"status": "success", "id": advisor.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/advisor/{advisor_id}", summary="Delete an advisor vector by ID")
def delete_advisor(advisor_id: int):
    try:
        vector_index.delete(advisor_id)
        return {"status": "deleted", "id": advisor_id}
    except KeyError:
        raise HTTPException(status_code=404, detail="ID not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", summary="Top-K semantic similarity search")
def search_advisors(query: Query):
    embedding = model.encode(query.query)
    results = vector_index.search(np.array(embedding, dtype=np.float32), query.k)
    return {"query": query.query, "results": results}
    
@app.post("/embed", summary="Embed a string and return its vector")
def embed_text(text_in: TextIn):
    embedding = model.encode(text_in.text)
    return {"vector": embedding.tolist()}
