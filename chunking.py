# ==============================================================================
# Step 7 – Data Chunking
# Strategies from the diagram:
#   1. Fixed-size chunking   (chunk_size tokens, overlap)
#   2. Recursive chunking    (split on separators, then by size)
#   3. Semantic chunking     [Mostly used] – split on paragraph / sentence breaks
#   4. Hierarchical chunking (page / chapter / paragraph)
# ==============================================================================
import json
import os
import re
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    char_start: int
    char_end: int
    chunk_index: int
    strategy: str


# ── 1. Fixed-size chunking ────────────────────────────────────────────────────

def _split_fixed(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) character ranges for fixed-size windows."""
    ranges = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        ranges.append((start, end))
        if end == len(text):
            break
        start += chunk_size - overlap
    return ranges


def chunk_fixed_size(text: str, source: str,
                     chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    ranges = _split_fixed(text, chunk_size, overlap)
    return [
        Chunk(
            chunk_id=f"{Path(source).stem}_fixed_{i}",
            source=source,
            text=text[s:e].strip(),
            char_start=s, char_end=e,
            chunk_index=i,
            strategy="fixed_size",
        )
        for i, (s, e) in enumerate(ranges)
        if text[s:e].strip()
    ]


# ── 2. Recursive chunking ─────────────────────────────────────────────────────

def chunk_recursive(text: str, source: str,
                    chunk_size: int = 1000, overlap: int = 200,
                    separators: list[str] | None = None) -> list[Chunk]:
    """Split on paragraph breaks first, then sentences, then fixed size."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split(txt: str, seps: list[str]) -> list[str]:
        if not seps or len(txt) <= chunk_size:
            return [txt] if txt.strip() else []
        sep = seps[0]
        parts = txt.split(sep) if sep else list(txt)
        result, current = [], ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                current = part
        if current:
            result.append(current)
        # Further split any oversized pieces
        final = []
        for piece in result:
            if len(piece) > chunk_size:
                final.extend(_split(piece, seps[1:]))
            else:
                final.append(piece)
        return final

    raw_chunks = _split(text, separators)
    # Apply overlap: prepend tail of previous chunk
    overlapped, pos = [], 0
    for i, chunk_text in enumerate(raw_chunks):
        start = max(0, text.find(chunk_text, pos) )
        end = start + len(chunk_text)
        overlapped.append(Chunk(
            chunk_id=f"{Path(source).stem}_recursive_{i}",
            source=source,
            text=chunk_text.strip(),
            char_start=start, char_end=end,
            chunk_index=i,
            strategy="recursive",
        ))
        pos = max(0, end - overlap)
    return [c for c in overlapped if c.text]


# ── 3. Semantic chunking [Mostly used] ───────────────────────────────────────

def chunk_semantic(text: str, source: str,
                   max_chunk_size: int = 1500) -> list[Chunk]:
    """
    Split on paragraph (double newline) boundaries, then group small
    paragraphs together up to max_chunk_size. Best semantic coherence.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current, char_pos = [], "", 0

    def _make_chunk(idx: int, txt: str, start: int) -> Chunk:
        return Chunk(
            chunk_id=f"{Path(source).stem}_semantic_{idx}",
            source=source,
            text=txt.strip(),
            char_start=start,
            char_end=start + len(txt),
            chunk_index=idx,
            strategy="semantic",
        )

    idx, start = 0, 0
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chunk_size:
            chunks.append(_make_chunk(idx, current, start))
            idx += 1
            start = text.find(para, start + len(current))
            current = para
        else:
            current = (current + "\n\n" + para) if current else para

    if current.strip():
        chunks.append(_make_chunk(idx, current, start))

    return chunks


# ── 4. Hierarchical chunking ──────────────────────────────────────────────────

def chunk_hierarchical(text: str, source: str,
                       max_chunk_size: int = 1200) -> list[Chunk]:
    """
    Split on heading-like boundaries (markdown ## or ALL CAPS lines),
    creating chapter → paragraph hierarchy.
    """
    # Detect section headers: markdown headings or UPPER CASE lines
    heading_pattern = re.compile(r"^(#{1,3} .+|[A-Z][A-Z0-9 \-:]{5,})$", re.MULTILINE)
    boundaries = [m.start() for m in heading_pattern.finditer(text)] + [len(text)]

    chunks, idx = [], 0
    for i in range(len(boundaries) - 1):
        section = text[boundaries[i]:boundaries[i + 1]].strip()
        if not section:
            continue
        # If section too large, sub-split semantically
        if len(section) > max_chunk_size:
            sub = chunk_semantic(section, source, max_chunk_size)
            for s in sub:
                s.chunk_id = f"{Path(source).stem}_hier_{idx}"
                s.chunk_index = idx
                s.strategy = "hierarchical"
                chunks.append(s)
                idx += 1
        else:
            chunks.append(Chunk(
                chunk_id=f"{Path(source).stem}_hier_{idx}",
                source=source,
                text=section,
                char_start=boundaries[i],
                char_end=boundaries[i + 1],
                chunk_index=idx,
                strategy="hierarchical",
            ))
            idx += 1

    return chunks


# ── Dispatcher & persistence ──────────────────────────────────────────────────

STRATEGIES = {
    "fixed": chunk_fixed_size,
    "recursive": chunk_recursive,
    "semantic": chunk_semantic,           # [Mostly used]
    "hierarchical": chunk_hierarchical,
}


def chunk_text(text: str, source: str, strategy: str = "semantic", **kwargs) -> list[Chunk]:
    """Dispatch to the chosen chunking strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES)}")
    return STRATEGIES[strategy](text, source, **kwargs)


def save_chunks(chunks: list[Chunk], chunks_dir: str = "data/chunks") -> list[str]:
    """Persist each chunk as an individual JSON file. Returns list of paths."""
    os.makedirs(chunks_dir, exist_ok=True)
    paths = []
    for chunk in chunks:
        out = Path(chunks_dir) / f"{chunk.chunk_id}.json"
        out.write_text(json.dumps(asdict(chunk), indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(str(out))
    return paths


def load_chunks_from_dir(chunks_dir: str = "data/chunks") -> list[Chunk]:
    """Load all chunk JSON files from directory."""
    chunks = []
    for f in sorted(Path(chunks_dir).glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        chunks.append(Chunk(**data))
    return chunks
