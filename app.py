import os
import io
import requests
import streamlit as st
import chromadb
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer
import fitz

TEXTS = {
    "setup": ("Setup", "सेटअप"),
    "groq_api_key": ("Groq API Key", "Groq API कुंजी"),
    "get_free": ("Get one free at console.groq.com", "console.groq.com से मुफ्त प्राप्त करें"),
    "model": ("Model", "मॉडल"),
    "web_fallback": ("Web Crawl Fallback", "वेब क्रॉल वैकल्पिक"),
    "hindi_toggle": ("हिंदी में उत्तर दें", "हिंदी में उत्तर दें"),
    "download_chat": ("Download Chat (txt)", "चैट डाउनलोड करें (txt)"),
    "clear_chat": ("Clear Chat", "चैट साफ़ करें"),
    "try_asking": ("Try asking:", "पूछ सकते हैं:"),
    "db_error": ("Vector database not found or empty.\n\nPlease run `python build_index.py` first.",
                 "वेक्टर डेटाबेस नहीं मिला या खाली है।\n\nकृपया पहले `python build_index.py` चलाएँ।"),
    "loading_model": ("Loading model...", "मॉडल लोड हो रहा है..."),
    "sources_used": ("Sources used", "उपयोग किए गए स्रोत"),
    "web_used": ("Web fallback was used for this answer.", "इस उत्तर के लिए वेब वैकल्पिक का उपयोग किया गया।"),
    "you_might_ask": ("You might also ask:", "आप यह भी पूछ सकते हैं:"),
    "ask_placeholder": ("Ask about ITR, TDS, deductions, deadlines…",
                        "आईटीआर, टीडीएस, कटौती, समयसीमा के बारे में पूछें..."),
    "no_api_key": ("Please enter your Groq API key in the sidebar.",
                   "कृपया साइडबार में Groq API कुंजी दर्ज करें।"),
    "searching": ("Searching…", "खोज हो रही है…"),
    "web_fallback_msg": ("Web fallback…", "वेब वैकल्पिक…"),
    "error_prefix": ("Error: ", "त्रुटि: "),
}

def _(key, hindi):
    eng, hin = TEXTS.get(key, (key, key))
    return hin if hindi else eng

DB_FOLDER = "vector_db"
COLLECTION_NAME = "tax_docs"
TOP_K = 3
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
FALLBACK_URLS = [
    "https://www.incometax.gov.in",
    "https://www.incometaxindia.gov.in",
    "https://cleartax.in",
    "https://taxguru.in",
]

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_db():
    if not os.path.exists(DB_FOLDER):
        return None
    try:
        client = chromadb.PersistentClient(path=DB_FOLDER)
        existing = [c.name for c in client.list_collections()]
        if COLLECTION_NAME not in existing:
            return None
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == 0:
            return None
        return collection
    except Exception:
        return None

def crawl_website_for_pdfs(url, max_pages=5):
    pdf_links = []
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        for link in soup.find_all("a", href=True):
            h = link["href"]
            if h.endswith(".pdf"):
                pdf_links.append(urljoin(url, h))
        for link in soup.find_all("a", href=True):
            h = link["href"].lower()
            if any(k in h for k in ["itr", "tax", "return", "filing", "deduction"]):
                if not h.endswith(".pdf") and len(pdf_links) < max_pages:
                    pdf_links.extend(crawl_website_for_pdfs(urljoin(url, h), max_pages=1))
    except Exception as e:
        st.warning(f"Could not crawl {url}: {e}")
    return list(set(pdf_links))[:max_pages]

def download_and_process_pdf(pdf_url):
    try:
        r = requests.get(pdf_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        doc = fitz.open(stream=io.BytesIO(r.content), filetype="pdf")
        return "".join(page.get_text() for page in doc)
    except Exception as e:
        st.warning(f"Could not download PDF from {pdf_url}: {e}")
        return None

def retrieve_relevant_chunks(question, collection, model, top_k=TOP_K):
    q_vec = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = round(max(0.0, 1.0 - dist / 2.0), 3)
        chunks.append({"text": doc, "source": meta["source"],
                       "page": meta["page"], "score": score})
    return chunks

def build_prompt(question, chunks, web_results=None, chat_history=None, hindi=False):
    memory = ""
    if chat_history:
        memory = "RECENT CONVERSATION:\n"
        for turn in chat_history[-3:]:
            memory += f"User: {turn['q']}\nAssistant: {turn['a']}\n\n"

    context_text = ""
    for i, chunk in enumerate(chunks):
        context_text += (
            f"\n--- Source {i+1}: {chunk['source']} "
            f"(page {chunk['page']}, relevance {chunk['score']:.2f}) ---\n"
            f"{chunk['text']}\n"
        )

    web_section = ""
    if web_results:
        web_section = f"\n\n**WEB SEARCH RESULTS (if answer not in PDFs):**\n{web_results}\n"

    lang = "IMPORTANT: Reply entirely in Hindi (Devanagari script).\n" if hindi else ""

    return f"""You are a helpful Income Tax assistant for India.

INSTRUCTIONS:
1. FIRST, try to answer using ONLY the PDF CONTEXT below.
2. If answer is NOT in the PDF CONTEXT, check the WEB SEARCH RESULTS.
3. If found in web results, say "This information wasn't in your PDFs, but I found it online:"
4. If not found anywhere, say "Please check the official Income Tax website at incometax.gov.in"
5. ALWAYS end your answer with a line that starts with "Sources:" listing every PDF filename and page number you used.
6. Keep the answer simple — the user is a regular taxpayer, not a CA.
{lang}
{memory}
PDF CONTEXT:
{context_text}
{web_section}
QUESTION: {question}

ANSWER (end with Sources:):"""

def ask_groq(prompt, api_key, model="llama-3.3-70b-versatile"):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are a helpful Income Tax assistant for India. "
                        "Always cite the source PDF and page number at the end of every answer."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 1000,
    }
    r = requests.post(GROQ_URL, headers=headers, json=data, timeout=30)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    raise Exception(f"API Error {r.status_code}: {r.text}")

def get_follow_ups(question, api_key, model):
    try:
        text = ask_groq(
            f"Suggest exactly 3 short follow-up Indian Income Tax questions "
            f"after someone asked: {question!r}\n"
            "Output ONLY the 3 questions, one per line, no numbering.",
            api_key, model,
        )
        lines = [l.strip().lstrip("123.-) ") for l in text.strip().splitlines() if l.strip()]
        return lines[:3]
    except Exception:
        return []

st.set_page_config(page_title="Income Tax FAQ Chatbot", layout="wide")
st.title("Income Tax FAQ Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = []
if "pending_q" not in st.session_state:
    st.session_state.pending_q = ""
if "processing" not in st.session_state:
    st.session_state.processing = False
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "collection" not in st.session_state:
    st.session_state.collection = None
if "embed_model" not in st.session_state:
    st.session_state.embed_model = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

with st.sidebar:
    hindi_mode = st.toggle(_("hindi_toggle", False), value=False, key="hindi_mode")
    st.header(_("setup", hindi_mode))
    api_key = st.text_input(
        _("groq_api_key", hindi_mode),
        value=st.session_state.api_key,
        type="password",
        help=_("get_free", hindi_mode),
        key="api_key_input"
    )
    st.session_state.api_key = api_key
    model_choice = st.selectbox(_("model", hindi_mode), [
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "llama-3.1-8b-instant",
    ])
    enable_web = st.checkbox(_("web_fallback", hindi_mode), value=True)

    st.markdown("---")
    if st.button(_("clear_chat", hindi_mode)):
        st.session_state.messages = []
        st.session_state.memory = []
        st.session_state.pending_q = ""
        st.session_state.processing = False
        st.rerun()

    if st.session_state.messages:
        chat_text = ""
        for msg in st.session_state.messages:
            role = "User" if msg["role"]=="user" else "Assistant"
            chat_text += f"{role}: {msg['content']}\n\n"
        st.download_button(
            _("download_chat", hindi_mode),
            data=chat_text,
            file_name="tax_chat.txt",
            mime="text/plain"
        )

    st.markdown("---")
    st.markdown(_("try_asking", hindi_mode))
    samples = [
        ("Who should file ITR-1?", "आईटीआर-1 किसे दाखिल करना चाहिए?"),
        ("What is Section 80C?", "धारा 80C क्या है?"),
        ("What is the ITR filing deadline?", "आईटीआर दाखिल करने की अंतिम तिथि क्या है?"),
        ("How is TDS calculated?", "टीडीएस की गणना कैसे की जाती है?"),
    ]
    for eng, hin in samples:
        label = hin if hindi_mode else eng
        if st.button(label, key=f"s_{label[:12]}"):
            st.session_state.pending_q = eng
            st.rerun()

if not st.session_state.initialized:
    with st.spinner(_("loading_model", hindi_mode)):
        collection = load_vector_db()
        embed_model = load_embedding_model()
        st.session_state.collection = collection
        st.session_state.embed_model = embed_model
        st.session_state.initialized = True
else:
    collection = st.session_state.collection
    embed_model = st.session_state.embed_model

if collection is None:
    st.error(_("db_error", hindi_mode))
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            conf = msg.get("confidence", 0)
            st.progress(min(conf, 1.0))
            if msg.get("sources"):
                with st.expander(_("sources_used", hindi_mode)):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['source']}** — page {s['page']}  ·  score {s['score']:.2f}")
                        st.caption(s["text"][:250] + "…")
            if msg.get("web_used"):
                st.info(_("web_used", hindi_mode))
            if msg.get("suggestions"):
                st.markdown(_("you_might_ask", hindi_mode))
                cols = st.columns(len(msg["suggestions"]))
                for idx, (col, sug) in enumerate(zip(cols, msg["suggestions"])):
                    if col.button(sug, key=f"sug_{id(msg)}_{idx}_{sug[:10]}"):
                        if not st.session_state.processing:
                            st.session_state.pending_q = sug
                            st.session_state.processing = True
                            st.rerun()

user_question = st.chat_input(_("ask_placeholder", hindi_mode))
if st.session_state.pending_q:
    user_question = st.session_state.pending_q
    st.session_state.pending_q = ""
    st.session_state.processing = False

if user_question:
    if not st.session_state.api_key:
        st.warning(_("no_api_key", hindi_mode))
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner(_("searching", hindi_mode)):
            chunks = retrieve_relevant_chunks(user_question, collection, embed_model)
            top_score = chunks[0]["score"] if chunks else 0.0
            web_used = False
            web_results = None

            if enable_web and (not chunks or len(chunks) < 2):
                with st.spinner(_("web_fallback_msg", hindi_mode)):
                    for url in FALLBACK_URLS[:2]:
                        pdfs = crawl_website_for_pdfs(url, max_pages=2)
                        for pdf_url in pdfs[:2]:
                            pdf_text = download_and_process_pdf(pdf_url)
                            if pdf_text:
                                web_results = f"Found relevant information from: {pdf_url}\n{pdf_text[:1000]}..."
                                web_used = True
                                break
                        if web_used:
                            break

            prompt = build_prompt(
                user_question, chunks, web_results,
                chat_history=st.session_state.memory,
                hindi=hindi_mode,
            )
            try:
                answer = ask_groq(prompt, st.session_state.api_key, model_choice)
                suggestions = get_follow_ups(user_question, st.session_state.api_key, model_choice)
            except Exception as e:
                answer = _("error_prefix", hindi_mode) + str(e)
                suggestions = []

            st.markdown(answer)
            st.progress(min(top_score, 1.0))

            if chunks:
                with st.expander(_("sources_used", hindi_mode)):
                    for s in chunks:
                        st.markdown(f"**{s['source']}** — page {s['page']}  ·  score {s['score']:.2f}")
                        st.caption(s["text"][:250] + "…")

            if web_used:
                st.info(_("web_used", hindi_mode))

            if suggestions:
                st.markdown(_("you_might_ask", hindi_mode))
                cols = st.columns(len(suggestions))
                for idx, (col, sug) in enumerate(zip(cols, suggestions)):
                    if col.button(sug, key=f"new_{len(st.session_state.messages)}_{idx}_{sug[:10]}"):
                        if not st.session_state.processing:
                            st.session_state.pending_q = sug
                            st.session_state.processing = True
                            st.rerun()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "confidence": top_score,
        "sources": chunks,
        "web_used": web_used,
        "suggestions": suggestions,
    })
    st.session_state.memory.append({"q": user_question, "a": answer})
    st.rerun()