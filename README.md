# Income Tax Chatbot

A chatbot for Indian income tax filing questions, built with Aikya Oruganti for our SMAI course
(report: `Income Tax Chatbot.pdf`). It answers in English or Hindi from the official ITR instruction
documents and shows which PDF and page each answer came from.

## How it works

- `build_index.py` splits the PDFs in `data/` into 1200-character chunks, embeds them with
  all-MiniLM-L6-v2 and stores them in ChromaDB.
- `app.py` (Streamlit) retrieves the top 5 chunks for a question and sends them to an LLM on Groq along
  with the last few turns of the chat.
- Retrieval combines BM25 keyword search with embedding search (reciprocal rank fusion, `retrieval.py`).
  Keyword search matters here because questions often hinge on exact section numbers like 80TTA or 87A.

## Retrieval evaluation

`eval/retrieval_questions.json` has 25 questions, each tagged with a phrase from the passage that answers it.
A question counts as retrieved if a returned chunk contains that phrase.

| setup | found in top 3 | found in top 5 |
|---|---|---|
| embeddings only, 400-char chunks (original) | 32% | 36% |
| hybrid BM25 + embeddings, 1200-char chunks | 60% | 68% |

Full results for each chunk size and retriever are in `eval/retrieval_results.json`. With only 25
questions, small differences aren't meaningful.

```bash
python -m eval.retrieval_eval --out eval/retrieval_results.json
```

## Running it

```bash
pip install -r requirements.txt
python build_index.py     # builds index/ (takes a few minutes)
streamlit run app.py      # paste a Groq API key in the sidebar
```

## Notes

- `generalfaqs.pdf` is a scanned PDF with no text, so it isn't searchable without OCR.
- The ITR-1/ITR-4 documents are for AY 2021-22 and the rest for AY 2023-24, so some rules may be outdated.
  Not tax advice.
