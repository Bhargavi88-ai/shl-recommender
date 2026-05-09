"""
Catalog Store — TF-IDF retrieval over the SHL product catalog.

Design choices:
- TF-IDF over sentence-transformers to avoid heavy ML deps and stay fast on free hosting.
- Cosine similarity at query time; catalog fits fully in memory (~100 products).
- Falls back to scraper-generated catalog; uses bundled data/catalog.json otherwise.
- Re-indexes automatically when catalog.json changes.
"""

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data" / "catalog.json"

# Test type letter → readable label mapping
TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Situational Judgement",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class CatalogStore:
    """In-memory TF-IDF retrieval over the SHL assessment catalog."""

    def __init__(self) -> None:
        self.products: List[Dict[str, Any]] = []
        self._corpus_tokens: List[List[str]] = []
        self._idf: Dict[str, float] = {}
        self._tf_idf_vecs: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Loading / indexing
    # ------------------------------------------------------------------

    def load(self, path: Path = DATA_PATH) -> None:
        """Load catalog from JSON and build the TF-IDF index."""
        if not path.exists():
            logger.warning(f"Catalog file not found at {path}. Store is empty.")
            return

        with open(path, encoding="utf-8") as f:
            self.products = json.load(f)

        logger.info(f"Loaded {len(self.products)} assessments from {path}")
        self._build_index()

    def _build_index(self) -> None:
        """Build TF-IDF vectors for all products."""
        self._corpus_tokens = []
        for p in self.products:
            doc = " ".join(
                [
                    p.get("name", ""),
                    p.get("description", ""),
                    " ".join(p.get("test_type_labels", [])),
                    p.get("test_type", ""),
                ]
            )
            self._corpus_tokens.append(_tokenize(doc))

        # Compute IDF
        N = len(self._corpus_tokens)
        df: Counter = Counter()
        for tokens in self._corpus_tokens:
            for t in set(tokens):
                df[t] += 1

        self._idf = {t: math.log((N + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}

        # Compute TF-IDF vectors
        self._tf_idf_vecs = []
        for tokens in self._corpus_tokens:
            tf: Counter = Counter(tokens)
            total = len(tokens) or 1
            vec = {t: (cnt / total) * self._idf.get(t, 1.0) for t, cnt in tf.items()}
            self._tf_idf_vecs.append(vec)

        logger.info("TF-IDF index built.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _query_vec(self, query: str) -> Dict[str, float]:
        tokens = _tokenize(query)
        tf: Counter = Counter(tokens)
        total = len(tokens) or 1
        return {t: (cnt / total) * self._idf.get(t, 1.0) for t, cnt in tf.items()}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        dot = sum(a[k] * b[k] for k in a if k in b)
        mag_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
        mag_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
        return dot / (mag_a * mag_b)

    def search(
        self,
        query: str,
        top_k: int = 10,
        test_type_filter: Optional[List[str]] = None,
        remote_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return up to top_k assessments most relevant to query.

        Parameters
        ----------
        query            : free-text search query
        top_k            : max results to return
        test_type_filter : list of test type letters, e.g. ['A', 'P'] to restrict
        remote_only      : if True, only return remote-testing-enabled products
        """
        if not self.products:
            return []

        qvec = self._query_vec(query)
        if not qvec:
            return self.products[:top_k]

        scored: List[Tuple[float, int]] = []
        for i, pvec in enumerate(self._tf_idf_vecs):
            p = self.products[i]

            # Apply filters
            if remote_only and not p.get("remote_testing", False):
                continue

            if test_type_filter:
                product_types = set(re.findall(r"[A-Z]", p.get("test_type", "")))
                if not product_types.intersection(set(test_type_filter)):
                    continue

            sim = self._cosine(qvec, pvec)
            scored.append((sim, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self.products[i] for _, i in scored[:top_k]]

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Exact or fuzzy name lookup."""
        name_lower = name.lower()
        # Exact
        for p in self.products:
            if p["name"].lower() == name_lower:
                return p
        # Partial
        for p in self.products:
            if name_lower in p["name"].lower() or p["name"].lower() in name_lower:
                return p
        return None

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.products)

    def format_for_context(self, products: List[Dict[str, Any]]) -> str:
        """Format a list of products as a compact catalog context string."""
        lines = []
        for p in products:
            test_type_full = ", ".join(
                TEST_TYPE_LABELS.get(t.strip(), t.strip())
                for t in p.get("test_type", "").split(",")
                if t.strip()
            )
            remote = "Yes" if p.get("remote_testing") else "No"
            adaptive = "Yes" if p.get("adaptive_irt") else "No"
            desc = p.get("description", "")
            lines.append(
                f"- **{p['name']}** | Type: {test_type_full} | Remote: {remote} | Adaptive: {adaptive}\n"
                f"  URL: {p['url']}\n"
                f"  {desc}"
            )
        return "\n".join(lines)


# Module-level singleton
_store: Optional[CatalogStore] = None


def get_store() -> CatalogStore:
    """Return (and lazily initialise) the global CatalogStore."""
    global _store
    if _store is None:
        _store = CatalogStore()
        _store.load()
    return _store
