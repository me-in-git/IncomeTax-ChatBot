#  Income Tax FAQ Chatbot

> A bilingual AI-powered chatbot for Indian Income Tax queries, built with RAG (Retrieval-Augmented Generation), Streamlit, ChromaDB, and Groq LLM.

---

##  Features

- Bilingual Support — Responses in both English and Hindi (Devanagari script)
- Semantic Search — Vector-based retrieval using Sentence Transformers for high accuracy
- LLM-Powered — High-performance inference via Groq Cloud
- Web Fallback — Automatic web crawling when local PDFs lack answers
- Source Citation — Precise citations with PDF filenames and page numbers
- Chat Management — Context-aware follow-up suggestions and chat history export (`.txt`)

---

## Architecture

```
┌─────────────────────┐      ┌──────────────────────┐
│   OFFLINE INDEXING  │      │     ONLINE QUERY      │
│                     │      │                       │
│  PDFs               │      │    User Question      │
│    ↓ chunk & embed  │      │          ↓            │
│  ChromaDB           │      │   embed & retrieve    │
│ (Vector Database)   │      │          ↓            │
└──────────┬──────────┘      │  LLM (Groq) → Answer  │
           │                 └──────────┬────────────┘
           └─────────────────────────────┘
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- A [Groq API Key](https://console.groq.com/)
- Income Tax PDF documents (to be placed in `data/pdfs/`)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/me-in-git/IncomeTax-ChatBot.git
cd IncomeTax-ChatBot
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your PDF documents**
```bash
mkdir -p data/pdfs
# Move your tax-related PDFs into this folder
```

---

## How It Works

### Phase 1 — Indexing (`build_index.py`)

Run once to build your knowledge base from local PDFs:

```bash
python build_index.py
```

| Step | What Happens |
|------|--------------|
| **Extraction** | Reads text from PDFs in `data/pdfs/` |
| **Chunking** | Splits text into 400-character segments with 50-character overlap |
| **Embedding** | Converts chunks to vectors using `all-MiniLM-L6-v2` |
| **Storage** | Saves vectors locally to `vector_db/` via ChromaDB |

### Phase 2 — Query (`app.py`)

Launch the Streamlit UI and start asking questions:

```bash
streamlit run app.py
```

| Step | What Happens |
|------|--------------|
| **Retrieval** | Matches user question against ChromaDB vector space |
| **Augmentation** | Combines retrieved chunks with the prompt |
| **Generation** | Groq LLM produces a cited, context-aware response |

---

## Configuration

These parameters can be adjusted in `app.py` or `build_index.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_K` | `3` | Number of text chunks retrieved per query |
| `CHUNK_SIZE` | `400` | Characters per text segment |
| `CHUNK_OVERLAP` | `50` | Overlap between segments for context continuity |
| `DB_FOLDER` | `"vector_db"` | Local storage path for ChromaDB |

---

## Ablation Study Results

Optimal chunk size was found by testing retrieval score across configurations:

| Chunk Size | Total Chunks | Avg. Retrieval Score |
|------------|--------------|----------------------|
| 200 | 6220 | 0.676 |
| **400** | **2667** | **0.6472 ✅ Optimal** |
| 800 | 1247 | 0.5751 |

---

## Project Structure

```
IncomeTax-ChatBot/
├── app.py              # Streamlit web application
├── build_index.py      # Vector database builder
├── requirements.txt    # Python dependencies
├── data/               # Source PDF folder
│   └── pdfs/
└── vector_db/          # ChromaDB storage (auto-generated)
```

---

## Troubleshooting

**"Vector database not found"**
> Run `python build_index.py` after adding PDFs to `data/pdfs/`.

**API Key Error**
> Paste your Groq API key correctly in the Streamlit sidebar.

**Slow First Run**
> The app downloads the Sentence Transformers model (~300 MB) on first execution. Subsequent runs are fast.

---

## Future Enhancements

- [ ] Multi-language support for more regional Indian languages
- [ ] Document Upload UI within the app
- [ ] Fine-tuning on Indian Tax Law datasets
- [ ] REST API endpoint for external website integration

---

## Disclaimer

This chatbot is an **educational tool only**. For official guidance, always refer to the [Income Tax Department Website](https://www.incometax.gov.in/) or consult a qualified Tax Professional (CA).

---
