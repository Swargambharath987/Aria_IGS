import os
import requests
import chromadb
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
SLURM_BASE = "https://slurm.schedmd.com/archive/slurm-23.11.6/"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")


# ── 1. Parse all institute PDFs ───────────────────────────────────────────────

PDF_SOURCES = {
    "ENG-SLURMGridSchedulerOverview-010426-1155-10.pdf": "institute_doc",
    "SLURM_Troubleshooting.pdf": "institute_troubleshooting_doc",
}

def parse_pdfs():
    print("1. Parsing institute PDFs...")
    all_docs = []
    for filename, source_label in PDF_SOURCES.items():
        pdf_path = BASE_DIR / "data" / "raw" / filename
        if not pdf_path.exists():
            print(f"   [skip] {filename} not found")
            continue
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                all_docs.append(Document(
                    text=text.strip(),
                    metadata={"source": source_label, "priority": "high", "page": i + 1}
                ))
        print(f"   {filename}: {len(reader.pages)} pages")
    print(f"   Total PDF docs: {len(all_docs)}")
    return all_docs


# ── 2. Scrape Slurm 23.11.6 docs ─────────────────────────────────────────────

def get_slurm_links():
    resp = requests.get(SLURM_BASE, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".html") and not href.startswith("http"):
            links.add(SLURM_BASE + href)
    return list(links)


def scrape_page(url):
    resp = requests.get(url, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else url.split("/")[-1]
    content = soup.get_text(separator="\n", strip=True)
    return title, content


def scrape_slurm_docs():
    print("\n2. Scraping Slurm 23.11.6 docs...")
    links = get_slurm_links()
    print(f"   Found {len(links)} pages to scrape")
    docs = []
    for i, url in enumerate(links):
        try:
            title, content = scrape_page(url)
            docs.append(Document(
                text=content,
                metadata={
                    "source": "official_slurm_docs",
                    "priority": "normal",
                    "title": title,
                    "url": url,
                }
            ))
            print(f"   [{i+1}/{len(links)}] {title}")
        except Exception as e:
            print(f"   [skip] {url} — {e}")
    print(f"   {len(docs)} pages scraped successfully")
    return docs


# ── 3. Embed and store in ChromaDB ───────────────────────────────────────────

def build_index(documents):
    print(f"\n3. Embedding {len(documents)} documents into ChromaDB...")
    print("   (this will take a few minutes)")

    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=OLLAMA_BASE_URL)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection("slurm")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )
    print("\nDone. Knowledge base is ready.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Slurm Grid AI — Ingestion Pipeline ===\n")

    pdf_docs = parse_pdfs()
    slurm_docs = scrape_slurm_docs()

    all_docs = pdf_docs + slurm_docs
    print(f"\n   Total documents: {len(all_docs)}")

    build_index(all_docs)
