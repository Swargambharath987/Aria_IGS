import os
import uuid
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import chromadb
from fastapi import Depends, FastAPI, File, HTTPException, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────
CHROMA_DIR   = Path(os.environ.get("CHROMA_DIR",   "/app/data/chroma_db"))
PROMPT_PATH  = Path(os.environ.get("PROMPT_PATH",  "/app/prompts/system_prompt.txt"))
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",        "http://localhost:11434")
LLM_MODEL    = os.environ.get("LLM_MODEL",          "gemma4:e4b")
EMBED_MODEL  = os.environ.get("EMBED_MODEL",        "nomic-embed-text")

# Comma-separated list of valid tokens: API_TOKENS=token1,token2
_raw_tokens  = os.environ.get("API_TOKENS", "igs-dev-token")
VALID_TOKENS = {t.strip() for t in _raw_tokens.split(",") if t.strip()}

# In-memory session store: session_id → chat_engine
# For multi-server production, replace with Redis.
_sessions: dict = {}

# ── Auth ───────────────────────────────────────────────────────────────────
def require_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token not in VALID_TOKENS:
        raise HTTPException(status_code=403, detail="Invalid token")
    return token


# ── Engine factory ─────────────────────────────────────────────────────────
def _load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text().strip()
    return "You are the IGS Grid Assistant. Help users with SLURM and IGS cluster questions."


def _build_engine():
    embed = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_HOST)
    llm   = Ollama(model=LLM_MODEL, base_url=OLLAMA_HOST, request_timeout=120.0)

    # Set globally so LlamaIndex uses Ollama everywhere, not the OpenAI default
    Settings.llm = llm
    Settings.embed_model = embed

    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("slurm")
    store      = ChromaVectorStore(chroma_collection=collection)
    ctx        = StorageContext.from_defaults(vector_store=store)
    index      = VectorStoreIndex.from_vector_store(store, storage_context=ctx, embed_model=embed)

    return index, llm, embed, client


def _new_chat_engine(index, llm):
    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=llm,
        memory=ChatMemoryBuffer.from_defaults(token_limit=3000),
        system_prompt=_load_system_prompt(),
        similarity_top_k=5,
        verbose=False,
    )


# ── App lifecycle ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    index, llm, embed, chroma_client = _build_engine()
    app.state.index         = index
    app.state.llm           = llm
    app.state.embed         = embed
    app.state.chroma_client = chroma_client
    yield
    _sessions.clear()


app = FastAPI(title="Aria API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:    str
    session_id: Optional[str] = None   # omit to start a new session


class ChatResponse(BaseModel):
    response:   str
    session_id: str


class IngestResponse(BaseModel):
    status:      str
    chunks_added: int
    filename:    str


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    collection = app.state.chroma_client.get_or_create_collection("slurm")
    return {"status": "ok", "model": LLM_MODEL, "chunks_in_db": collection.count()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _token: str = Depends(require_token)):
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in _sessions:
        _sessions[session_id] = _new_chat_engine(app.state.index, app.state.llm)

    engine = _sessions[session_id]
    response = engine.chat(req.message)
    return ChatResponse(response=str(response), session_id=session_id)


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...), _token: str = Depends(require_token)):
    allowed = {".pdf", ".txt", ".md"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"File type not supported. Allowed: {allowed}")

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        docs = SimpleDirectoryReader(tmpdir).load_data()

    embed = app.state.embed
    index = app.state.index
    for doc in docs:
        index.insert(doc)

    return IngestResponse(
        status="ok",
        chunks_added=len(docs),
        filename=file.filename,
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str, _token: str = Depends(require_token)):
    _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
