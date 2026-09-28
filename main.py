# ==============================================================================
# main.py – End-to-End RAG Orchestrator
# Runs all steps from the diagram: 1 → 10 (ingest) then 11–19 (query loop)
# ==============================================================================
import os
import sys
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()   # loads GEMINI_API_KEY from .env

# ── Rich console for nice output ───────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    from rich import print as rprint
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    console = None

def _print(msg: str, style: str = ""):
    if _RICH and console:
        console.print(msg, style=style)
    else:
        print(msg)

def _panel(title: str, content: str):
    if _RICH and console:
        console.print(Panel(content, title=title, border_style="blue"))
    else:
        print(f"\n{'='*60}\n{title}\n{'='*60}\n{content}\n")


# ── Import project modules ────────────────────────────────────────────────────

from data_processing import process_directory, extract_text_from_file, save_cleaned_data
from chunking import chunk_text, save_chunks, load_chunks_from_dir
from vector_store_manager import FAISSVectorStore, embed_texts
from rag_pipeline import run_rag_pipeline

INDEX_PATH = "vector_store/faiss.index"
META_PATH  = "vector_store/metadata.pkl"


# ==============================================================================
# Phase 1 – Ingest (Steps 1–10)
# ==============================================================================

def ingest(
    raw_dir: str = "data/raw",
    chunks_dir: str = "data/chunks",
    cleaned_dir: str = "data/cleaned",
    strategy: str = "semantic",
):
    """
    Full ingestion pipeline:
      Step 5  – ETL: extract & clean text from raw documents
      Step 6  – Save cleaned JSON
      Step 7  – Chunk documents
      Step 8  – Generate embeddings
      Step 9  – Build FAISS vector index
      Step 10 – Persist knowledge base (ready for query)
    """
    _print("\n[bold cyan]═══ RAG Ingestion Pipeline ═══[/bold cyan]" if _RICH else "\n=== RAG Ingestion Pipeline ===")

    # ── Step 5 & 6: Data Processing ──────────────────────────────────────────
    _print("\n[bold]Step 5-6: ETL – Extracting & cleaning documents…[/bold]" if _RICH else "\nStep 5-6: ETL")
    cleaned_paths = process_directory(raw_dir, cleaned_dir)
    _print(f"  [green]✓ {len(cleaned_paths)} document(s) cleaned[/green]" if _RICH else f"  ✓ {len(cleaned_paths)} docs cleaned")

    # ── Step 7: Chunking ──────────────────────────────────────────────────────
    _print(f"\n[bold]Step 7: Chunking (strategy={strategy})…[/bold]" if _RICH else f"\nStep 7: Chunking [{strategy}]")
    all_chunks = []
    for cp in cleaned_paths:
        data = json.loads(Path(cp).read_text(encoding="utf-8"))
        text   = data["content"]
        source = data["source"]
        chunks = chunk_text(text, source, strategy=strategy)
        saved  = save_chunks(chunks, chunks_dir)
        _print(f"  [green]✓ {Path(source).name}: {len(chunks)} chunks[/green]" if _RICH else f"  ✓ {Path(source).name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    _print(f"  Total chunks: [bold]{len(all_chunks)}[/bold]" if _RICH else f"  Total chunks: {len(all_chunks)}")

    # ── Step 8: Embeddings ────────────────────────────────────────────────────
    _print("\n[bold]Step 8: Generating embeddings (gemini-embedding-2)…[/bold]" if _RICH else "\nStep 8: Embeddings")
    texts = [c.text for c in all_chunks]
    embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
    _print(f"  [green]✓ {len(embeddings)} embeddings generated[/green]" if _RICH else f"  ✓ {len(embeddings)} embeddings")

    # ── Step 9 & 10: Build & Save FAISS ──────────────────────────────────────
    _print("\n[bold]Step 9-10: Building FAISS vector index & saving knowledge base…[/bold]" if _RICH else "\nStep 9-10: FAISS index")
    store = FAISSVectorStore()
    store.build(all_chunks, embeddings)
    store.save(INDEX_PATH, META_PATH)

    _print("\n[bold green]✓ Knowledge base ready for query![/bold green]" if _RICH else "\n✓ Knowledge base ready!")
    return store


# ==============================================================================
# Phase 2 – Query (Steps 11–19)
# ==============================================================================

def query_loop(store: FAISSVectorStore, top_k: int = 5, evaluate: bool = False):
    """Interactive REPL for the RAG query pipeline."""
    _print("\n[bold cyan]═══ RAG Query Pipeline  (type 'exit' to quit) ═══[/bold cyan]" if _RICH else "\n=== RAG Query Mode (type 'exit') ===")

    while True:
        if _RICH:
            user_query = Prompt.ask("\n[bold yellow]You[/bold yellow]").strip()
        else:
            user_query = input("\nYou: ").strip()

        if not user_query or user_query.lower() in {"exit", "quit", "q"}:
            _print("[dim]Goodbye.[/dim]" if _RICH else "Goodbye.")
            break

        response = run_rag_pipeline(user_query, store, top_k=top_k, run_evaluation=evaluate)

        # Display answer
        _panel("🤖 Answer", response.answer)

        # Display metadata
        if _RICH and console:
            t = Table(title="Sources", show_lines=True)
            t.add_column("#", style="cyan", width=3)
            t.add_column("File", style="green")
            t.add_column("Score", style="magenta", width=7)
            t.add_column("Preview", no_wrap=False)
            for s in response.sources:
                t.add_row(str(s["ref"]), Path(s["source"]).name, str(s["score"]), s["preview"])
            console.print(t)
        else:
            print("\n--- Sources ---")
            for s in response.sources:
                print(f"[{s['ref']}] {s['source']}  score={s['score']}")
                print(f"     {s['preview']}")

        if response.guardrail_triggered:
            _print("[bold red]⚠ Output guardrail was triggered.[/bold red]" if _RICH else "⚠ Guardrail triggered.")
        if response.evaluation:
            _panel("📊 Evaluation", response.evaluation)

        _print(f"[dim]Model: {response.model_used} | Query type: {response.query_type}[/dim]" if _RICH else f"Model: {response.model_used} | Type: {response.query_type}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End RAG Application (Gemini + FAISS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest documents then query
  python main.py --ingest --query

  # Ingest only
  python main.py --ingest --strategy semantic

  # Query only (uses existing index)
  python main.py --query

  # Ask a single question (non-interactive)
  python main.py --query --question "What is retrieval augmented generation?"

  # Run with evaluation
  python main.py --query --evaluate
        """
    )
    parser.add_argument("--ingest",   action="store_true", help="Run ingestion pipeline (steps 1-10)")
    parser.add_argument("--query",    action="store_true", help="Run query pipeline (steps 11-19)")
    parser.add_argument("--question", type=str, default=None, help="Single question (non-interactive)")
    parser.add_argument("--strategy", type=str, default="semantic",
                        choices=["semantic", "fixed", "recursive", "hierarchical"],
                        help="Chunking strategy (default: semantic)")
    parser.add_argument("--top-k",   type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--evaluate", action="store_true", help="Enable LLM-as-a-judge evaluation")
    parser.add_argument("--raw-dir",  type=str, default="data/raw", help="Directory with source documents")

    args = parser.parse_args()

    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    if not args.ingest and not args.query:
        parser.print_help()
        sys.exit(0)

    store = None

    # ── Ingestion ─────────────────────────────────────────────────────────────
    if args.ingest:
        store = ingest(raw_dir=args.raw_dir, strategy=args.strategy)

    # ── Query ─────────────────────────────────────────────────────────────────
    if args.query:
        if store is None:
            # Load existing index
            if not Path(INDEX_PATH).exists():
                print("ERROR: No FAISS index found. Run with --ingest first.")
                sys.exit(1)
            store = FAISSVectorStore.load(INDEX_PATH, META_PATH)

        if args.question:
            # Single question mode
            response = run_rag_pipeline(args.question, store, top_k=args.top_k, run_evaluation=args.evaluate)
            _panel("🤖 Answer", response.answer)
            if response.evaluation:
                _panel("📊 Evaluation", response.evaluation)
        else:
            # Interactive loop
            query_loop(store, top_k=args.top_k, evaluate=args.evaluate)


if __name__ == "__main__":
    main()
