# ==============================================================================
# Steps 11–19: Query → Retrieval → Generation Pipeline
#
# 11. User Query
# 12. Input Guardrails       (block malicious / unsafe queries)
# 13. Query Understanding & Routing  (simple → fast model, complex → capable model)
# 14. Retrieval Strategy     (keyword + semantic hybrid; top-K fetch)
# 15. Reranking              (rescore by relevance)
# 16. LLM Generation         (Gemini with retrieved context)
# 17. Data Postprocessing    (format, add references)
# 18. Output Guardrails      (mask PII, check toxicity, remove harmful content)
# 19. Evaluator              (LLM-as-a-judge)
# ==============================================================================
import re
from dataclasses import dataclass

from google import genai

from chunking import Chunk
from vector_store_manager import FAISSVectorStore


# ── Model routing config ──────────────────────────────────────────────────────

FAST_MODEL    = "gemini-3.8-flash"          # simple / short queries
CAPABLE_MODEL = "gemini-3.8-flash"          # complex queries (swap to gemini-3.1-pro-preview for max quality)
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview"]


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]          # [{chunk_id, source, score}]
    model_used: str
    query_type: str
    guardrail_triggered: bool
    evaluation: str | None = None


# ── Step 12: Input Guardrails ─────────────────────────────────────────────────

_BLOCKED_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"jailbreak",
    r"system prompt",
    r"you are now",
    r"DAN mode",
    r"<script",
    r"exec\s*\(",
    r"__import__",
]
_BLOCKED_RE = re.compile("|".join(_BLOCKED_PATTERNS), re.IGNORECASE)


def _input_guardrail(query: str) -> tuple[bool, str]:
    """Return (is_safe, reason). Blocks prompt injections and abuse."""
    if _BLOCKED_RE.search(query):
        return False, "Query flagged by security filter."
    if len(query.strip()) < 3:
        return False, "Query too short."
    if len(query) > 2000:
        return False, "Query too long (max 2000 chars)."
    return True, ""


# ── Step 13: Query Understanding & Routing ────────────────────────────────────

def _classify_query(query: str) -> str:
    """Simple heuristic routing: 'simple' or 'complex'."""
    complex_signals = ["explain", "compare", "summarise", "summarize", "analyze",
                       "difference between", "pros and cons", "how does", "why does"]
    q_lower = query.lower()
    if any(sig in q_lower for sig in complex_signals) or len(query.split()) > 15:
        return "complex"
    return "simple"


# ── Step 14 & 15: Hybrid Retrieval + Reranking ───────────────────────────────

def _keyword_score(query: str, chunk: Chunk) -> float:
    """BM25-lite: count query term overlaps (keyword retrieval signal)."""
    query_terms = set(re.findall(r"\w+", query.lower()))
    chunk_terms = re.findall(r"\w+", chunk.text.lower())
    if not chunk_terms:
        return 0.0
    hits = sum(1 for t in chunk_terms if t in query_terms)
    return hits / len(chunk_terms)


def _hybrid_retrieve(query: str, store: FAISSVectorStore,
                     top_k: int = 10, alpha: float = 0.7) -> list[dict]:
    """
    Hybrid retrieval: alpha * semantic_score + (1-alpha) * keyword_score
    Then rerank and return top-K.
    """
    semantic_results = store.search(query, top_k=top_k * 2)   # fetch more for reranking

    for r in semantic_results:
        kw = _keyword_score(query, r["chunk"])
        r["hybrid_score"] = alpha * r["score"] + (1 - alpha) * kw

    # Step 15: Rerank by hybrid score
    reranked = sorted(semantic_results, key=lambda x: x["hybrid_score"], reverse=True)
    return reranked[:top_k]


# ── Step 16: LLM Generation ───────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a helpful, accurate assistant that answers questions
ONLY using the provided context documents. If the answer is not in the context,
say "I don't have enough information to answer that."

Always:
- Be concise and factual.
- Cite the source document name at the end of relevant statements.
- Do not hallucinate facts not present in the context."""


def _build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, r in enumerate(chunks, 1):
        chunk = r["chunk"]
        context_blocks.append(
            f"[Document {i} | Source: {chunk.source} | ID: {chunk.chunk_id}]\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"


def _generate_answer(prompt: str, model: str) -> tuple[str, str]:
    client = genai.Client()
    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
    last_err = None
    for m in models_to_try:
        try:
            interaction = client.interactions.create(
                model=m,
                input=prompt,
                system_instruction=_SYSTEM_PROMPT,
            )
            return interaction.output_text or "", m
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("Failed to generate answer from all available models.")


# ── Step 17: Data Postprocessing ─────────────────────────────────────────────

def _format_response(answer: str, top_chunks: list[dict]) -> tuple[str, list[dict]]:
    """Add numbered references and structure the source list."""
    sources = []
    for i, r in enumerate(top_chunks, 1):
        c = r["chunk"]
        sources.append({
            "ref": i,
            "chunk_id": c.chunk_id,
            "source": c.source,
            "score": round(r.get("hybrid_score", r["score"]), 4),
            "preview": c.text[:120] + "…" if len(c.text) > 120 else c.text,
        })

    ref_block = "\n".join(
        f"[{s['ref']}] {s['source']} (score: {s['score']})" for s in sources
    )
    formatted = f"{answer.strip()}\n\n**References:**\n{ref_block}"
    return formatted, sources


# ── Step 18: Output Guardrails ────────────────────────────────────────────────

_PII_PATTERNS = {
    "email":   re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone":   re.compile(r"\b(\+?\d[\d\s\-().]{7,}\d)\b"),
    "ssn":     re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "cc":      re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
}

_HARMFUL_WORDS = {"hack", "exploit", "malware", "bomb", "weapon", "illegal"}


def _output_guardrail(text: str) -> tuple[str, bool]:
    """Mask PII and flag harmful content in the generated answer."""
    triggered = False
    for label, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            text = pattern.sub(f"[REDACTED-{label.upper()}]", text)
            triggered = True
    lower = text.lower()
    if any(w in lower for w in _HARMFUL_WORDS):
        triggered = True
        text = "[Response withheld by output safety filter.]"
    return text, triggered


# ── Step 19: Evaluator (LLM-as-a-judge) ──────────────────────────────────────

_EVAL_PROMPT_TEMPLATE = """You are an evaluation judge. Score the answer on a scale 1-5 for:
1. Relevance   – Does it answer the question?
2. Faithfulness – Is it grounded in the provided context?
3. Coherence    – Is it well-structured?

Question: {query}
Answer: {answer}
Context used: {context_preview}

Respond with: Relevance: X/5, Faithfulness: X/5, Coherence: X/5, Summary: <1 sentence>"""


def _evaluate(query: str, answer: str, top_chunks: list[dict]) -> str:
    context_preview = " | ".join(r["chunk"].text[:80] for r in top_chunks[:3])
    prompt = _EVAL_PROMPT_TEMPLATE.format(
        query=query, answer=answer, context_preview=context_preview
    )
    client = genai.Client()
    models_to_try = [FAST_MODEL] + [m for m in FALLBACK_MODELS if m != FAST_MODEL]
    for m in models_to_try:
        try:
            result = client.interactions.create(model=m, input=prompt)
            return result.output_text or "Evaluation unavailable."
        except Exception:
            continue
    return "Evaluation unavailable."


# ── Main RAG Pipeline ─────────────────────────────────────────────────────────

def run_rag_pipeline(
    query: str,
    store: FAISSVectorStore,
    top_k: int = 5,
    run_evaluation: bool = False,
) -> RAGResponse:
    """
    Full RAG pipeline: Steps 11–19.
    Returns a RAGResponse with answer, sources, model info, and optional eval.
    """

    # Step 12: Input guardrails
    is_safe, reason = _input_guardrail(query)
    if not is_safe:
        return RAGResponse(
            answer=f"⚠️ Query blocked: {reason}",
            sources=[], model_used="none",
            query_type="blocked",
            guardrail_triggered=True,
        )

    # Step 13: Query routing
    query_type = _classify_query(query)
    target_model = CAPABLE_MODEL if query_type == "complex" else FAST_MODEL

    # Steps 14 & 15: Hybrid retrieval + reranking
    top_chunks = _hybrid_retrieve(query, store, top_k=top_k)
    if not top_chunks:
        return RAGResponse(
            answer="No relevant documents found in the knowledge base.",
            sources=[], model_used=target_model,
            query_type=query_type,
            guardrail_triggered=False,
        )

    # Step 16: LLM Generation
    prompt = _build_prompt(query, top_chunks)
    raw_answer, model_used = _generate_answer(prompt, target_model)

    # Step 17: Postprocessing
    formatted_answer, sources = _format_response(raw_answer, top_chunks)

    # Step 18: Output guardrails
    safe_answer, output_flagged = _output_guardrail(formatted_answer)

    # Step 19: Optional evaluation
    evaluation = None
    if run_evaluation:
        evaluation = _evaluate(query, safe_answer, top_chunks)

    return RAGResponse(
        answer=safe_answer,
        sources=sources,
        model_used=model_used,
        query_type=query_type,
        guardrail_triggered=output_flagged,
        evaluation=evaluation,
    )
