"""Recall@k / MRR for each chunk size and retriever.

A question counts as retrieved at k if one of the top-k chunks contains all
of its evidence phrases.

    python -m eval.retrieval_eval --sizes 400 1200 --out eval/retrieval_results.json

"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from build_index import CHUNK_OVERLAP, PDF_FOLDER, chunk_text, extract_text

QUESTIONS = Path(__file__).with_name("retrieval_questions.json")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_pages(folder: str) -> list[tuple[str, int, str]]:
    pages = []
    for pdf in sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf")):
        for page_num, text in extract_text(os.path.join(folder, pdf)):
            pages.append((pdf, page_num, text))
    return pages


def build_chunks(pages, size: int) -> list[str]:
    chunks = []
    for _, _, text in pages:
        chunks.extend(c for c in chunk_text(text, size, CHUNK_OVERLAP) if len(c.strip()) >= 30)
    return chunks


def evaluate(model, pages, questions, size: int, modes=("dense", "bm25", "hybrid"), ks=(1, 3, 5)) -> list[dict]:
    from retrieval import Retriever

    chunks = build_chunks(pages, size)
    normed = [norm(c) for c in chunks]
    emb = model.encode(chunks, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    retriever = Retriever(chunks, emb, lambda q: model.encode(q, normalize_embeddings=True))

    results = []
    for mode in modes:
        ranks, context = [], []
        for q in questions:
            top = retriever.search(q["question"], k=max(ks), mode=mode)
            evidence = [norm(e) for e in q["evidence"]]
            ranks.append(next((r + 1 for r, i in enumerate(top) if all(e in normed[i] for e in evidence)), None))
            context.append(sum(len(chunks[i]) for i in top[:3]))
        n = len(questions)
        results.append({
            "retriever": mode,
            "chunk_size": size,
            "chunks": len(chunks),
            **{f"recall@{k}": round(sum(r is not None and r <= k for r in ranks) / n, 3) for k in ks},
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
            "avg_context_chars_top3": int(np.mean(context)),
            "missed": [q["id"] for q, r in zip(questions, ranks) if r is None],
        })
        print({k: v for k, v in results[-1].items() if k != "missed"})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200])
    parser.add_argument("--out")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    pages = load_pages(PDF_FOLDER)
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    results = [row for s in args.sizes for row in evaluate(model, pages, questions, s)]
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
