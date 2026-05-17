# catalog.py
import json, os, re
import numpy as np
from rank_bm25 import BM25Okapi

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CATALOG_FILE = os.path.join(BASE_DIR, "catalog_data.json")

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
    keys  = raw.get("keys", [])
    codes = list(dict.fromkeys(
        KEY_TO_CODE.get(k, "") for k in keys if k in KEY_TO_CODE
    ))
    test_type = ",".join(c for c in codes if c)

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
    return " | ".join(filter(None, [
        item["name"],
        item["description"][:300],
        "keys " + " ".join(item["keys"]),
        "levels " + " ".join(item["job_levels"]),
        "languages " + " ".join(item["languages"][:4]),
    ]))

def _tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


class CatalogSearch:
    def __init__(self):
        print("Loading catalog from:", CATALOG_FILE)
        with open(CATALOG_FILE, encoding="utf-8") as f:
            raw_items = json.load(f)

        self.items = [_normalise(r) for r in raw_items]
        self.texts = [_make_text(i) for i in self.items]

        print(f"Catalog: {len(self.items)} assessments loaded")
        self._build_bm25()
        print("Search index ready ✓")

    def _build_bm25(self):
        tokenized  = [_tokenize(t) for t in self.texts]
        self.bm25  = BM25Okapi(tokenized)
        print("BM25 index built")

    def search(self, query: str, top_k: int = 15) -> list:
        """
        Pure BM25 keyword search.
        Fast, zero RAM overhead, works well for SHL catalog
        because assessment names are specific keywords.
        """
        scores     = self.bm25.get_scores(_tokenize(query))
        top_idx    = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                item = dict(self.items[idx])
                item["_score"] = float(scores[idx])
                results.append(item)

        return results

    def get_by_name(self, name: str) -> dict | None:
        if not name:
            return None
        nl = name.lower().strip()
        for item in self.items:
            if item["name"].lower() == nl:
                return item
        for item in self.items:
            if nl in item["name"].lower() or item["name"].lower() in nl:
                return item
        return None


_catalog = None

def get_catalog() -> CatalogSearch:
    global _catalog
    if _catalog is None:
        _catalog = CatalogSearch()
    return _catalog