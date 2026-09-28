# ==============================================================================
# Step 5 & 6 – Data Processing (ETL) and Save Cleaned Data
# Handles: text extraction from PDF, DOCX, TXT, and raw .txt/.json files
# ==============================================================================
import os
import json
import re
from pathlib import Path

# ── Optional imports (graceful degradation) ───────────────────────────────────
try:
    from pypdf import PdfReader
    _PYPDF_OK = True
except ImportError:
    _PYPDF_OK = False

try:
    from docx import Document as DocxDocument
    _DOCX_OK = True
except ImportError:
    _DOCX_OK = False


def _clean_text(text: str) -> str:
    """Basic text cleaning: collapse whitespace, strip control chars."""
    text = re.sub(r"[\r\n]+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_text_from_file(file_path: str) -> str:
    """
    Extract raw text from supported file types:
      .pdf  → PyPDF
      .docx → python-docx
      .txt / .md / .json → read directly
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        if not _PYPDF_OK:
            raise ImportError("pypdf not installed. Run: pip install pypdf")
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return _clean_text("\n".join(pages))

    elif suffix == ".docx":
        if not _DOCX_OK:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
        doc = DocxDocument(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return _clean_text("\n".join(paragraphs))

    elif suffix in {".txt", ".md"}:
        return _clean_text(path.read_text(encoding="utf-8", errors="ignore"))

    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, str):
            return _clean_text(data)
        return _clean_text(json.dumps(data, indent=2))

    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def save_cleaned_data(text: str, source_path: str, cleaned_dir: str = "data/cleaned") -> str:
    """
    Save cleaned text as a JSON file (TOON-inspired: structured, token-friendly).
    Returns the path to the saved file.
    """
    os.makedirs(cleaned_dir, exist_ok=True)
    stem = Path(source_path).stem
    output_path = Path(cleaned_dir) / f"{stem}_cleaned.json"
    payload = {
        "source": str(source_path),
        "char_count": len(text),
        "content": text,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(output_path)


def process_directory(raw_dir: str = "data/raw", cleaned_dir: str = "data/cleaned") -> list[str]:
    """
    Process all supported files in raw_dir → save cleaned JSON to cleaned_dir.
    Returns list of cleaned file paths.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    supported = {".pdf", ".docx", ".txt", ".md", ".json"}
    files = [f for f in raw_path.iterdir() if f.suffix.lower() in supported]

    if not files:
        raise ValueError(f"No supported files found in {raw_dir}. Add .pdf/.docx/.txt/.md/.json files.")

    cleaned_paths = []
    for f in files:
        text = extract_text_from_file(str(f))
        saved = save_cleaned_data(text, str(f), cleaned_dir)
        cleaned_paths.append(saved)
        print(f"  ✓ Processed: {f.name} → {Path(saved).name}")

    return cleaned_paths
