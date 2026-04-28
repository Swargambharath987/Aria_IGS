"""Add a single PDF to the existing ChromaDB without full re-ingestion."""
import sys
import chromadb
from pathlib import Path
from pypdf import PdfReader
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"


def add_pdf(pdf_path: Path, source_label: str, priority: str = "high"):
    print(f"Adding: {pdf_path.name}")
    reader = PdfReader(pdf_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            docs.append(Document(
                text=text.strip(),
                metadata={"source": source_label, "priority": priority, "page": i + 1}
            ))
    print(f"  {len(docs)} pages extracted")

    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection("slurm")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )
    print(f"Done. {pdf_path.name} added to knowledge base.")


if __name__ == "__main__":
    pdf = BASE_DIR / "data" / "raw" / "SLURM_Troubleshooting.pdf"
    add_pdf(pdf, source_label="institute_troubleshooting_doc", priority="high")
