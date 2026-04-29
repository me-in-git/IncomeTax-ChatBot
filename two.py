
import os
import streamlit as st
import chromadb
import requests
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import fitz  # PyMuPDF for PDFs
from urllib.parse import urljoin, urlparse
import io

# settings
DB_FOLDER = "vector_db"
TOP_K_RESULTS = 3
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Income Tax India relevant URLs
INCOME_TAX_URLS = [
    "https://www.incometax.gov.in",
    "https://www.incometaxindia.gov.in",
    "https://cleartax.in",
    "https://taxguru.in"
]


@st.cache_resource
def load_embedding_model():
    """Load the same embedding model we used when building the index."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_db():
    """Connect to the ChromaDB database saved by step1_build_index.py"""
    client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = client.get_collection("tax_docs")
    return collection

def crawl_website_for_pdfs(url, max_pages=5):
    """Crawl a website and find PDF links"""
    pdf_links = []
    
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all PDF links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.pdf'):
                full_url = urljoin(url, href)
                pdf_links.append(full_url)
        
        # Also search for common tax document patterns
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            if any(keyword in href for keyword in ['itr', 'tax', 'return', 'filing', 'deduction']):
                if not href.endswith('.pdf'):
                    full_url = urljoin(url, href)
                    # Recursively crawl (limited depth)
                    if len(pdf_links) < max_pages:
                        pdf_links.extend(crawl_website_for_pdfs(full_url, max_pages=1))
                        
    except Exception as e:
        st.warning(f"Could not crawl {url}: {e}")
    
    return list(set(pdf_links))[:max_pages]

def download_and_process_pdf(pdf_url):
    """Download a PDF from URL and extract text"""
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        # Open PDF from memory
        pdf_file = io.BytesIO(response.content)
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        
        text = ""
        for page in doc:
            text += page.get_text()
        
        return text
    except Exception as e:
        st.warning(f"Could not download PDF from {pdf_url}: {e}")
        return None

def search_web_for_answer(question, api_key):
    """Search the web for relevant information"""
    search_prompt = f"""You need to find information about: {question}
    
    Please provide:
    1. Relevant Income Tax India website URLs that might contain this information
    2. Suggested search terms to find official PDFs
    
    Respond in JSON format:
    {{"urls": ["url1", "url2"], "search_terms": ["term1", "term2"]}}"""
    
    # Use Groq to suggest where to look
    response = ask_groq(search_prompt, api_key, model="mixtral-8x7b-32768")
    return response

def retrieve_relevant_chunks(question, collection, model, top_k=TOP_K_RESULTS):
    question_vector = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=top_k
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"]
        })
    return chunks

def build_prompt(question, chunks, web_results=None):
    context_text = ""
    for i, chunk in enumerate(chunks):
        context_text += f"\n--- Source {i+1}: {chunk['source']} (page {chunk['page']}) ---\n"
        context_text += chunk["text"] + "\n"
    
    web_section = ""
    if web_results:
        web_section = f"\n\n**WEB SEARCH RESULTS (if answer not in PDFs):**\n{web_results}\n"
    
    prompt = f"""You are a helpful Income Tax assistant for India.

**INSTRUCTIONS:**
1. FIRST, try to answer using ONLY the PDF CONTEXT below
2. If answer is NOT in the PDF CONTEXT, check the WEB SEARCH RESULTS
3. If found in web results, say "This information wasn't in your PDFs, but I found it online:"
4. If not found anywhere, say "Please check the official Income Tax website at incometax.gov.in"

**PDF CONTEXT:**
{context_text}
{web_section}

**QUESTION:** {question}

**ANSWER:**"""
    return prompt

def ask_groq(prompt, api_key, model="llama-3.3-70b-versatile"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful Income Tax assistant for India."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API Error {response.status_code}: {response.text}")

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Income Tax FAQ Chatbot",
    page_icon="🇮🇳",
    layout="centered"
)

st.title("🇮🇳 Income Tax FAQ Chatbot")
st.caption("Searches your PDFs + can crawl official websites if needed")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get a free key at https://console.groq.com"
    )
    
    enable_web_crawl = st.checkbox("Enable Web Crawling Fallback", value=True)
    
    model_choice = st.selectbox(
        "Choose Groq Model",
        ["mixtral-8x7b-32768", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. First searches your uploaded PDFs")
    st.markdown("2. If not found, can crawl Income Tax websites")
    st.markdown("3. Provides sources from both local and web")

# Check if vector DB exists
if not os.path.exists(DB_FOLDER):
    st.error("Vector database not found! Please run step1 first.")
    st.stop()

# Load models
with st.spinner("Loading models..."):
    embed_model = load_embedding_model()
    collection = load_vector_db()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_question = st.chat_input("Ask a question about Income Tax filing...")

if user_question:
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()
    
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    with st.chat_message("assistant"):
        with st.spinner("Searching PDFs..."):
            try:
                # First, search local PDFs
                chunks = retrieve_relevant_chunks(user_question, collection, embed_model)
                
                # Check if we need web crawling
                web_results = None
                if enable_web_crawl and (not chunks or len(chunks) < 2):
                    with st.spinner("Not found in PDFs. Crawling websites..."):
                        # Search for relevant PDFs online
                        for url in INCOME_TAX_URLS[:2]:
                            pdfs = crawl_website_for_pdfs(url, max_pages=2)
                            for pdf_url in pdfs[:2]:
                                pdf_text = download_and_process_pdf(pdf_url)
                                if pdf_text:
                                    web_results = f"Found relevant information from: {pdf_url}\n{pdf_text[:1000]}..."
                                    break
                            if web_results:
                                break
                
                # Build prompt with web results if available
                prompt = build_prompt(user_question, chunks, web_results)
                answer = ask_groq(prompt, api_key, model_choice)
                
                st.markdown(answer)
                
                if web_results:
                    st.info("This information was found online, not in your uploaded PDFs.")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {e}")