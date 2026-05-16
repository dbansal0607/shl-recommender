# catalog.py
"""
Loads the SHL catalog, normalises fields from scraped format,
builds a hybrid BM25 + FAISS semantic search index.
"""

import json, os, re, pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CATALOG_FILE = os.path.join(BASE_DIR, "catalog_data.json")
INDEX_DIR    = os.path.join(BASE_DIR, "faiss_index")
MODEL_NAME   = "all-MiniLM-L6-v2"

# The scraped catalog uses full names; map them to letter codes
KEY_TO_CODE = {
    "Ability & Aptitude":             "A",
    "Biodata & Situational Judgment": "B",
    "Competencies":                   "C",
    "Development & 360":              "D",
    "Assessment Exercises":           "E",
    "Knowledge & Skills":             "K",
    "Personality & Behavior":         "P",
    "Simulations":                    "S",
}


def _normalise(raw: dict) -> dict:
    """
    Convert a scraped catalog record into a clean internal format.
    The scraped format uses 'link' (not 'url'), 'keys' (array, not test_type),
    and 'remote'/'adaptive' as 'yes'/'no' strings.
    """
    keys  = raw.get("keys", [])
    codes = list(dict.fromkeys(
        KEY_TO_CODE.get(k, "") for k in keys if k in KEY_TO_CODE
    ))
    test_type = ",".join(c for c in codes if c)

    # Parse duration string e.g. "30 minutes" → keep as string, extract int separately
    duration_raw = raw.get("duration_raw", raw.get("duration", ""))
    dur_match    = re.search(r"(\d+)", str(duration_raw))
    duration_min = int(dur_match.group(1)) if dur_match else None

    return {
        "name":           raw.get("name", ""),
        "url":            raw.get("link", raw.get("url", "")),
        "description":    (raw.get("description", "") or "").replace("\r\n", " ").strip(),
        "test_type":      test_type or "K",
        "remote_testing": raw.get("remote", "no").lower() == "yes",
        "adaptive_irt":   raw.get("adaptive", "no").lower() == "yes",
        "job_levels":     raw.get("job_levels", []),
        "languages":      raw.get("languages", []),
        "duration":       raw.get("duration", ""),
        "duration_min":   duration_min,
        "keys":           keys,
        "entity_id":      raw.get("entity_id", ""),
    }


def _make_text(item: dict) -> str:
    """
    Create a single rich string per catalog item for embedding.
    We include all semantically useful fields.
    """
    return " | ".join(filter(None, [
        item["name"],
        item["description"][:400],
        "test type " + " ".join(item["keys"]),
        "levels " + " ".join(item["job_levels"]),
        "languages " + " ".join(item["languages"][:6]),
        "duration " + item["duration"],
    ]))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class CatalogSearch:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

        # Load and normalise
        print("Loading catalog from:", CATALOG_FILE)
        with open(CATALOG_FILE, encoding="utf-8") as f:
            raw_items = json.load(f)
        self.items = [_normalise(r) for r in raw_items]
        self.texts = [_make_text(i) for i in self.items]

        print(f"Catalog: {len(self.items)} assessments loaded")
        self._build_or_load_faiss()
        self._build_bm25()
        print("Search index ready ✓")

    # ── FAISS (semantic search) ────────────────────────────────────────────

    def _build_or_load_faiss(self):
        os.makedirs(INDEX_DIR, exist_ok=True)
        idx_path = f"{INDEX_DIR}/index.bin"
        cnt_path = f"{INDEX_DIR}/count.txt"

        # Reuse if catalog size hasn't changed
        if os.path.exists(idx_path) and os.path.exists(cnt_path):
            with open(cnt_path) as f:
                saved_count = int(f.read().strip())
            if saved_count == len(self.items):
                self.faiss_index = faiss.read_index(idx_path)
                print("FAISS index loaded from disk")
                return

        print("Building FAISS index (first run, ~1 min)…")
        embeddings = self.model.encode(
            self.texts,
            normalize_embeddings=True,   # makes inner product == cosine similarity
            show_progress_bar=True,
            batch_size=32,
        ).astype("float32")

        # IndexFlatIP = exact inner product search (cosine similarity when normalised)
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)

        faiss.write_index(self.faiss_index, idx_path)
        with open(cnt_path, "w") as f:
            f.write(str(len(self.items)))
        print(f"FAISS index built: {self.faiss_index.ntotal} vectors")

    # ── BM25 (keyword search) ──────────────────────────────────────────────

    def _build_bm25(self):
        tokenized = [_tokenize(t) for t in self.texts]
        self.bm25  = BM25Okapi(tokenized)

    # ── Hybrid search ──────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        60% semantic (FAISS) + 40% keyword (BM25).
        BM25 catches exact product names like 'OPQ32r' or 'Verify G+'.
        Semantic catches intent like 'personality test for sales leaders'.
        """
        n = len(self.items)

        # Semantic: get rank position for every item
        q_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        _, sem_indices = self.faiss_index.search(q_vec, n)
        sem_rank = np.zeros(n)
        for rank, idx in enumerate(sem_indices[0]):
            sem_rank[idx] = (n - rank) / n   # 1.0 = best, ~0 = worst

        # BM25: normalise to 0-1
        bm25_raw  = self.bm25.get_scores(_tokenize(query))
        max_bm25  = bm25_raw.max() or 1.0
        bm25_norm = bm25_raw / max_bm25

        # Combine
        combined = 0.6 * sem_rank + 0.4 * bm25_norm
        top_idx  = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_idx:
            item = dict(self.items[idx])
            item["_score"] = float(combined[idx])
            results.append(item)
        return results

    # ── Lookup by name (for URL validation) ───────────────────────────────

    def get_by_name(self, name: str) -> dict | None:
        if not name:
            return None
        nl = name.lower().strip()
        # Exact match first
        for item in self.items:
            if item["name"].lower() == nl:
                return item
        # Substring match (handles minor LLM rewording)
        for item in self.items:
            if nl in item["name"].lower() or item["name"].lower() in nl:
                return item
        return None


# Singleton — loaded once at startup, reused for every request
_catalog: CatalogSearch | None = None

def get_catalog() -> CatalogSearch:
    global _catalog
    if _catalog is None:
        _catalog = CatalogSearch()
    return _catalog