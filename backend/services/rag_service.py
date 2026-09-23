"""
RAG Service
Loads manufacturing knowledge base text files, creates a simple
TF-IDF vector index, and retrieves relevant passages for a query.
No external vector database required — self-contained and lightweight.
"""
import os
import math
import re
from typing import Optional

KNOWLEDGE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "rag", "knowledge")
)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return tokens


def _build_index(knowledge_dir: str) -> tuple[list[dict], dict]:
    """
    Load all .txt files and build a TF-IDF index.
    Returns:
        documents: list of { "source", "chunk", "tokens" }
        idf: dict of token -> IDF score
    """
    documents = []

    for fname in sorted(os.listdir(knowledge_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(knowledge_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            content = f.read()

        # Split into chunks of ~300 words
        words = content.split()
        chunk_size = 300
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            documents.append({
                "source": fname,
                "chunk": chunk,
                "tokens": _tokenize(chunk),
            })

    # Compute IDF
    N = len(documents)
    df: dict[str, int] = {}
    for doc in documents:
        for tok in set(doc["tokens"]):
            df[tok] = df.get(tok, 0) + 1

    idf = {tok: math.log((N + 1) / (freq + 1)) + 1 for tok, freq in df.items()}
    return documents, idf


def _tf_idf_score(query_tokens: list[str], doc_tokens: list[str], idf: dict) -> float:
    """Compute cosine-like TF-IDF similarity between query and document."""
    doc_tf: dict[str, float] = {}
    for tok in doc_tokens:
        doc_tf[tok] = doc_tf.get(tok, 0) + 1
    total = len(doc_tokens) + 1
    for tok in doc_tf:
        doc_tf[tok] /= total

    score = 0.0
    for tok in query_tokens:
        if tok in doc_tf:
            score += doc_tf[tok] * idf.get(tok, 1.0)
    return score


# ---- Module-level index (loaded once) ----
_documents: Optional[list] = None
_idf: Optional[dict] = None


def _ensure_index():
    global _documents, _idf
    if _documents is None:
        _documents, _idf = _build_index(KNOWLEDGE_DIR)


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve the top_k most relevant knowledge chunks for the query.
    Returns list of { "source", "chunk", "score" }.
    """
    _ensure_index()
    query_tokens = _tokenize(query)
    scored = []
    for doc in _documents:
        score = _tf_idf_score(query_tokens, doc["tokens"], _idf)
        scored.append({"source": doc["source"], "chunk": doc["chunk"], "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def retrieve_for_report(
    monitoring_result: dict,
    quality_result: dict,
    defect_result: dict,
) -> str:
    """
    Build a focused retrieval query from agent results and return
    the concatenated top knowledge passages as a context string.
    """
    query_parts = []

    # Add parameter-specific terms
    for p in monitoring_result.get("abnormal_parameters", []):
        param = p["parameter"]
        direction = "high" if p["value"] > {
            "temperature": 85, "pressure": 5.0, "vibration": 1.0,
            "machine_speed": 1000, "production_rate": 50
        }.get(param, 0) else "low"
        query_parts.append(f"{param} {direction} corrective action defect")

    if defect_result.get("defect_predicted"):
        query_parts.append("defect prediction quality control preventive maintenance")

    if not query_parts:
        query_parts.append("manufacturing quality control normal process")

    query = " ".join(query_parts)
    passages = retrieve(query, top_k=4)

    context_lines = []
    for p in passages:
        context_lines.append(f"[Source: {p['source']}]\n{p['chunk']}")

    return "\n\n---\n\n".join(context_lines)
