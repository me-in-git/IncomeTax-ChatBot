"""
Run this once after putting your PDFs in data/pdfs/
    python build_index.py
"""
import os
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer

PDF_FOLDER    = "data/pdfs"
DB_FOLDER     = "vector_db"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50


def extract_text(path):
    pages = []
    for i, page in enumerate(pypdf.PdfReader(path).pages):
        t = page.extract_text()
        if t and t.strip():
            pages.append((i + 1, t))
    return pages


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    parts, start = [], 0
    while start < len(text):
        parts.append(text[start:start + size])
        start += size - overlap
    return parts


def main():
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {PDF_FOLDER}/")
        print("Add Income Tax PDFs from https://www.incometax.gov.in then re-run.")
        return

    print(f"Found: {pdf_files}\n")
    print("Loading embedding model (downloads ~90 MB once)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_FOLDER)
    try:
        client.delete_collection("tax_docs")
    except Exception:
        pass
    col = client.create_collection("tax_docs")

    chunks, ids, metas, cid = [], [], [], 0
    for pdf in pdf_files:
        pages = extract_text(os.path.join(PDF_FOLDER, pdf))
        print(f"  {pdf}: {len(pages)} pages")
        for page_num, text in pages:
            for c in chunk_text(text):
                if len(c.strip()) < 30:
                    continue
                chunks.append(c)
                ids.append(f"c{cid}")
                metas.append({"source": pdf, "page": page_num})
                cid += 1

    print(f"\nEmbedding {len(chunks)} chunks...")
    embs = model.encode(chunks, show_progress_bar=True).tolist()

    for i in range(0, len(chunks), 500):
        col.add(
            documents  = chunks[i:i+500],
            embeddings = embs[i:i+500],
            ids        = ids[i:i+500],
            metadatas  = metas[i:i+500],
        )

    print(f"\nDone — {len(chunks)} chunks saved to {DB_FOLDER}/")
    print("Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
