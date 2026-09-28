# ==============================================================================
# Step 8 & 9 – Embeddings + Vector Database (FAISS open-source)
# Uses: gemini-embedding-2 (multimodal) or gemini-embedding-001 (text-only)
# Vector DB: FAISS (open-source, local)  ←  matches diagram's "FAISS" option
# ==============================================================================
import json
import os
import pickle
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from chunking import Chunk

# FAISS import with helpful error
try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False


# ── Embedding model config ────────────────────────────────────────────────────

EMBEDDING_MODEL = "gemini-embedding-2"   # multimodal, 3072-dim default
EMBEDDING_DIM   = 768                    # recommended truncated size (saves storage)
BATCH_SIZE      = 100                    # API batch limit


def _get_client() -> genai.Client:
    return genai.Client()


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Generate embeddings for a list of texts using Gemini embedding model.
    task_type options: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY
    """
    if not _FAISS_OK:
        raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    client = _get_client()
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        batch_vecs = [emb.values for emb in response.embeddings]
        all_embeddings.extend(batch_vecs)
        print(f"  Embedded batch {i // BATCH_SIZE + 1} / {(len(texts) - 1) // BATCH_SIZE + 1}")

    return all_embeddings


# ── FAISS Vector Store ────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    Lightweight FAISS wrapper that stores:
      - A flat L2 index for similarity search
      - A metadata list mapping vector ID → Chunk
    """

    INDEX_FILE    = "vector_store/faiss.index"
    METADATA_FILE = "vector_store/metadata.pkl"

    def __init__(self):
        if not _FAISS_OK:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")
        self.index: faiss.Index | None = None
        self.metadata: list[Chunk] = []   # parallel list to index rows

    # ── Build & Persist ───────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Create a new FAISS index from chunks + pre-computed embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        vecs = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vecs)                   # cosine similarity via L2 on normalised vecs

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)   # Inner Product = cosine after normalisation
        self.index.add(vecs)
        self.metadata = list(chunks)
        print(f"  ✓ FAISS index built: {self.index.ntotal} vectors ({EMBEDDING_DIM}-dim)")

    def save(self, index_path: str = INDEX_FILE, meta_path: str = METADATA_FILE) -> None:
        """Persist index and metadata to disk."""
        os.makedirs(Path(index_path).parent, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"  ✓ Saved FAISS index → {index_path}")
        print(f"  ✓ Saved metadata    → {meta_path}")

    @classmethod
    def load(cls, index_path: str = INDEX_FILE, meta_path: str = METADATA_FILE) -> "FAISSVectorStore":
        """Load a persisted FAISS store from disk."""
        store = cls()
        store.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            store.metadata = pickle.load(f)
        print(f"  ✓ Loaded FAISS index: {store.index.ntotal} vectors")
        return store

    # ── Search / Retrieval ────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embed the query and retrieve top-k most similar chunks.
        Returns list of dicts with 'chunk' and 'score'.
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Vector store is empty. Build or load it first.")

        [q_vec] = embed_texts([query], task_type="RETRIEVAL_QUERY")
        q_arr = np.array([q_vec], dtype="float32")
        faiss.normalize_L2(q_arr)

        scores, indices = self.index.search(q_arr, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "chunk": self.metadata[idx],
                "score": float(score),
            })
        return results

    def incremental_add(self, new_chunks: list[Chunk], new_embeddings: list[list[float]]) -> None:
        """Add new vectors without rebuilding the entire index (incremental indexing)."""
        vecs = np.array(new_embeddings, dtype="float32")
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.metadata.extend(new_chunks)
        print(f"  ✓ Incremental add: {len(new_chunks)} new vectors (total: {self.index.ntotal})")
