from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import os
from contextlib import asynccontextmanager

from main import ingest, INDEX_PATH, META_PATH
from vector_store_manager import FAISSVectorStore
from rag_pipeline import run_rag_pipeline

# Global vector store instance
store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the FAISS index if it exists
    global store
    if Path(INDEX_PATH).exists():
        print("Loading existing FAISS index...")
        store = FAISSVectorStore.load(INDEX_PATH, META_PATH)
    else:
        print("No FAISS index found. Ready to /ingest.")
    yield
    # Shutdown logic (if any) could go here

app = FastAPI(
    title="RAG API", 
    description="API for the End-to-End RAG Application using Gemini and FAISS",
    lifespan=lifespan
)

# ── API Models ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    evaluate: bool = False

class IngestRequest(BaseModel):
    raw_dir: str = "data/raw"
    strategy: str = "semantic"

class SourceModel(BaseModel):
    ref: int
    chunk_id: str
    source: str
    score: float
    preview: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    model_used: str
    query_type: str
    guardrail_triggered: bool
    evaluation: str | None = None

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "RAG API is running. Check /docs for interactive documentation."}

@app.post("/ingest")
def api_ingest(req: IngestRequest):
    """
    Triggers the ingestion pipeline: ETL -> Chunking -> Embedding -> FAISS
    """
    global store
    try:
        store = ingest(raw_dir=req.raw_dir, strategy=req.strategy)
        return {"message": "Ingestion successful. Knowledge base updated and loaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def api_query(req: QueryRequest):
    """
    Queries the RAG knowledge base.
    """
    global store
    if store is None:
        raise HTTPException(
            status_code=400, 
            detail="Knowledge base not initialized. Please call /ingest first."
        )
    
    try:
        response = run_rag_pipeline(
            query=req.query, 
            store=store, 
            top_k=req.top_k, 
            run_evaluation=req.evaluate
        )
        
        return QueryResponse(
            answer=response.answer,
            sources=response.sources,
            model_used=response.model_used,
            query_type=response.query_type,
            guardrail_triggered=response.guardrail_triggered,
            evaluation=response.evaluation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
