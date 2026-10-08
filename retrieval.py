"""Dense, BM25 and hybrid (reciprocal rank fusion) retrieval over text chunks."""

import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of to in for on and or is are be by with as at from this that it i my me "
    "can do does what which who how when where under".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class Retriever:
    def __init__(self, chunks: list[str], embeddings: np.ndarray, embed_query, rrf_k: int = 60):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.embeddings = embeddings  # L2-normalised, shape (n_chunks, dim)
        self.embed_query = embed_query  # str -> L2-normalised vector
        self.bm25 = BM25Okapi([tokenize(c) for c in chunks])
        self.rrf_k = rrf_k

    def dense(self, query: str, k: int) -> list[int]:
        scores = self.embeddings @ self.embed_query(query)
        return list(np.argsort(-scores)[:k])

    def keyword(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:k])

    def hybrid(self, query: str, k: int, pool: int = 50) -> list[int]:
        fused: dict[int, float] = {}
        for ranking in (self.dense(query, pool), self.keyword(query, pool)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)[:k]

    def search(self, query: str, k: int = 3, mode: str = "hybrid") -> list[int]:
        return {"dense": self.dense, "bm25": self.keyword, "hybrid": self.hybrid}[mode](query, k)
