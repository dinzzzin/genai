import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    raise RuntimeError("Ensure 'en_core_web_md' is installed via spacy download.") from e

# --- FIX INDENTATION HERE ---
class TextGenerationRequest(BaseModel):
    start_word: str
    num_words: int = 10

class EmbeddingResponse(BaseModel):
    word: str
    has_vector: bool
    vector_dimension: int
    embedding: List[float]

@app.get("/")
def root():
    return {"status": "API is running"}

@app.get("/embedding/{word}", response_model=EmbeddingResponse)
def get_embedding(word: str):
    doc = nlp(word)
    if not doc or not doc[0].has_vector:
        raise HTTPException(status_code=404, detail=f"No vector found for '{word}'")
    
    token = doc[0]
    return EmbeddingResponse(
        word=token.text,
        has_vector=token.has_vector,
        vector_dimension=len(token.vector),
        embedding=token.vector.tolist()
    )