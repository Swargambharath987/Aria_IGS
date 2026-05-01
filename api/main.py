import json
import logging
import os
import threading
import time
import uuid
import shutil
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import cachetools

import requests as http_requests
from bs4 import BeautifulSoup
from jose import JWTError, jwt

import chromadb
from fastapi import Depends, FastAPI, File, HTTPException, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole as LLMMessageRole
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from db.models import Base, Feedback, KnowledgeDoc, Message, MessageRole, Session as DBSession, User, UserRole
from db.session import SessionLocal, engine, get_db
from tools.slurm_ssh import get_live_data, needs_live_data
from tools.file_reader import get_file_data, needs_file_read

# ── Config ─────────────────────────────────────────────────────────────────
CHROMA_DIR  = Path(os.environ.get("CHROMA_DIR",  "/app/data/chroma_db"))
PROMPT_PATH = Path(os.environ.get("PROMPT_PATH", "/app/prompts/system_prompt.txt"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST",       "http://localhost:11434")
LLM_MODEL   = os.environ.get("LLM_MODEL",         "gemma4:e4b")
EMBED_MODEL = os.environ.get("EMBED_MODEL",        "nomic-embed-text")

# LLM backend — "ollama" (default, local dev) or "vllm" (production GPU server)
LLM_BACKEND   = os.environ.get("LLM_BACKEND",   "ollama")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8080/v1")
VLLM_MODEL    = os.environ.get("VLLM_MODEL",    LLM_MODEL)

_raw_tokens  = os.environ.get("API_TOKENS", "igs-dev-token")
VALID_TOKENS = {t.strip() for t in _raw_tokens.split(",") if t.strip()}

# JWT config — stub auth until LDAP/AD is wired (Phase 3)
JWT_SECRET    = os.environ.get("JWT_SECRET", "aria-dev-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_H  = 24

# Demo users loaded from DEMO_USERS env var: "admin:pass,user1:pass1"
# Replaced by LDAP validation once IT provides AD details
DEMO_USERS: dict[str, str] = {}
for _entry in os.environ.get("DEMO_USERS", "admin:aria_demo").split(","):
    if ":" in _entry:
        _u, _p = _entry.strip().split(":", 1)
        DEMO_USERS[_u] = _p

# In-memory store for LlamaIndex chat engines (stateful objects, not serializable)
# Key: session_id (str UUID), Value: chat engine instance
# Capped at 50 entries; least-recently-used entry is evicted when full.
_engines: cachetools.LRUCache = cachetools.LRUCache(maxsize=50)
_engines_lock: threading.Lock = threading.Lock()


# ── Auth ───────────────────────────────────────────────────────────────────

def require_auth(authorization: Optional[str] = Header(None)) -> dict:
    """Accept either a static API token (for scripts/curl) or a user JWT (from browser login)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    # Static API token — backward compat for curl and admin scripts
    if token in VALID_TOKENS:
        return {"username": "api", "role": "admin"}
    # JWT issued by /auth/login
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"username": payload["sub"], "role": payload.get("role", "user")}
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid or expired token")


# ── DB helpers ─────────────────────────────────────────────────────────────

def _get_or_create_user(db: Session, ldap_uid: str) -> User:
    """Return the User row for ldap_uid, creating it if this is the first request."""
    user = db.query(User).filter(User.ldap_uid == ldap_uid).first()
    if not user:
        user = User(
            ldap_uid=ldap_uid,
            display_name=ldap_uid,
            role=UserRole.user,
        )
        db.add(user)
        db.flush()  # get user.id without committing
    return user


def _get_or_create_session(db: Session, session_id: str, user: User, first_message: str) -> DBSession:
    """Return the Session row, creating it with an auto-title if new."""
    sid = uuid.UUID(session_id)
    session = db.query(DBSession).filter(DBSession.id == sid).first()
    if not session:
        title = first_message[:60] + ("…" if len(first_message) > 60 else "")
        session = DBSession(id=sid, user_id=user.id, title=title)
        db.add(session)
        db.flush()
    return session


def _persist_exchange(
    db: Session,
    session: DBSession,
    user_message: str,
    assistant_response: str,
    latency_ms: int,
    model: str,
) -> Message:
    """Write the user turn and assistant turn to the messages table."""
    db.add(Message(
        session_id=session.id,
        role=MessageRole.user,
        content=user_message,
    ))
    assistant_msg = Message(
        session_id=session.id,
        role=MessageRole.assistant,
        content=assistant_response,
        model_used=model,
        latency_ms=latency_ms,
    )
    db.add(assistant_msg)

    # Bump session activity timestamp
    from sqlalchemy.sql import func
    session.last_active_at = func.now()

    db.flush()
    return assistant_msg


# ── LlamaIndex engine factory ──────────────────────────────────────────────
def _load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text().strip()
    return "You are Aria, the IGS research computing assistant. Help users with their research workflows, SLURM jobs, bioinformatics tools, and coding."


def _build_rag_index():
    # Embeddings always run via Ollama (vLLM is a text-generation server, not an embedding server)
    embed = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_HOST)

    if LLM_BACKEND == "vllm":
        from llama_index.llms.openai_like import OpenAILike  # type: ignore[import]
        llm = OpenAILike(
            model=VLLM_MODEL,
            api_base=VLLM_BASE_URL,
            api_key="not-needed",         # vLLM doesn't require a real key
            request_timeout=120.0,
            is_chat_model=True,
            context_window=32768,
        )
        logger.info("LLM backend: vLLM — model=%s base_url=%s", VLLM_MODEL, VLLM_BASE_URL)
    else:
        llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_HOST, request_timeout=120.0)
        logger.info("LLM backend: Ollama — model=%s", LLM_MODEL)

    Settings.llm         = llm
    Settings.embed_model = embed

    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("slurm")
    store      = ChromaVectorStore(chroma_collection=collection)
    ctx        = StorageContext.from_defaults(vector_store=store)
    index      = VectorStoreIndex.from_vector_store(store, storage_context=ctx, embed_model=embed)
    return index, llm, embed, client


# ── Priority metadata reranker ─────────────────────────────────────────────

class PriorityReranker(BaseNodePostprocessor):
    """Boost nodes tagged priority='high' by +0.15 and re-sort descending.

    This postprocessor reads the ``priority`` metadata key written by
    ``_parse_into_nodes`` at ingest time.  Nodes from the ``slurm`` and
    ``institute`` collections are tagged "high"; everything else is "normal".
    After the boost the list is re-sorted so high-priority chunks bubble up
    ahead of lower-scoring normal chunks.
    """

    boost: float = 0.15

    @classmethod
    def class_name(cls) -> str:
        return "PriorityReranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        for nws in nodes:
            if nws.node.metadata.get("priority") == "high":
                nws.score = (nws.score or 0.0) + self.boost
        nodes.sort(key=lambda n: n.score or 0.0, reverse=True)
        return nodes


# ── Hybrid retriever builder ───────────────────────────────────────────────

def _build_hybrid_retriever(index: VectorStoreIndex) -> QueryFusionRetriever:
    """Build a hybrid BM25 + vector retriever fused via reciprocal-rank fusion.

    Architecture
    ------------
    1. VectorIndexRetriever  — cosine-similarity search over ChromaDB embeddings
    2. BM25Retriever (optional) — term-frequency keyword search over raw node text
       Nodes for BM25 are fetched from ChromaDB via ``index.vector_store.get_nodes``
       because ``VectorStoreIndex.from_vector_store`` does NOT populate the
       in-memory docstore when the vector store already stores text (stores_text=True).
    3. QueryFusionRetriever  — merges both result lists using reciprocal-rank fusion
       with ``num_queries=1`` (no query rewriting — just fuse as-is).

    Fallback
    --------
    If ``llama-index-retrievers-bm25`` is not installed, the function falls back
    to a semantic-only ``QueryFusionRetriever`` wrapping a single vector retriever.
    Priority reranking still applies in both paths via ``PriorityReranker``.
    """
    # Vector retriever — always available
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=8,
    )

    retrievers = [vector_retriever]
    mode = "semantic-only"

    try:
        from llama_index.retrievers.bm25 import BM25Retriever  # type: ignore[import]

        # Fetch all stored nodes from ChromaDB (docstore is empty for from_vector_store)
        all_nodes = index.vector_store.get_nodes(node_ids=None)
        if all_nodes:
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=all_nodes,
                similarity_top_k=8,
            )
            retrievers.append(bm25_retriever)
            mode = "hybrid"
        else:
            logger.warning(
                "BM25Retriever: ChromaDB returned 0 nodes — falling back to semantic-only retrieval"
            )
    except ImportError:
        logger.info(
            "llama-index-retrievers-bm25 not installed; using semantic-only retrieval. "
            "Install with: pip install llama-index-retrievers-bm25==0.4.0"
        )

    logger.info("RAG retriever mode: %s (retrievers: %d)", mode, len(retrievers))

    return QueryFusionRetriever(
        retrievers=retrievers,
        similarity_top_k=8,
        num_queries=1,           # no query rewriting — just fuse results as-is
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=False,
    )


def _new_chat_engine(index: VectorStoreIndex, llm):
    """Build a CondensePlusContextChatEngine with hybrid retrieval + priority reranking.

    Replaces the previous ``index.as_chat_engine(chat_mode='condense_plus_context', ...)``
    call with a manually wired engine so we can inject:
      - A hybrid BM25 + vector retriever (or semantic-only fallback)
      - PriorityReranker postprocessor to surface high-priority chunks
    """
    retriever = _build_hybrid_retriever(index)

    return CondensePlusContextChatEngine(
        retriever=retriever,
        llm=llm,
        memory=ChatMemoryBuffer.from_defaults(token_limit=3000),
        system_prompt=_load_system_prompt(),
        node_postprocessors=[PriorityReranker()],
        verbose=False,
    )


def _restore_engine_memory(engine, session_id: str, db: Session) -> None:
    """Reconstruct conversation context from PostgreSQL into a freshly created engine.

    This is called on cache miss (container restart or LRU eviction) so that the
    engine has the full prior chat history rather than starting blank.

    The condense_plus_context engine stores history on ``engine._memory``
    (a ``ChatMemoryBuffer`` / ``BaseMemory`` instance).  Its ``.set()`` method
    replaces the underlying chat-store contents atomically.

    MessageRole mapping:
      - DB MessageRole.user      → LLMMessageRole.USER
      - DB MessageRole.assistant → LLMMessageRole.ASSISTANT
      - DB MessageRole.tool      → LLMMessageRole.TOOL
    """
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        return  # malformed session_id — leave engine blank

    messages = (
        db.query(Message)
        .filter(Message.session_id == sid)
        .order_by(Message.created_at)
        .all()
    )
    if not messages:
        return  # no prior history — brand-new session

    _role_map = {
        MessageRole.user:      LLMMessageRole.USER,
        MessageRole.assistant: LLMMessageRole.ASSISTANT,
        MessageRole.tool:      LLMMessageRole.TOOL,
    }

    chat_history = [
        ChatMessage(
            role=_role_map.get(m.role, LLMMessageRole.USER),
            content=m.content,
        )
        for m in messages
    ]

    engine._memory.set(chat_history)


def _parse_into_nodes(docs: list, filename: str, collection: str) -> list:
    """
    Split documents into smaller nodes with metadata for better RAG retrieval.
    512-token chunks with 64-token overlap prevents splitting mid-command.
    Metadata tags let the retriever prefer institute docs over generic Slurm docs.
    """
    parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
        paragraph_separator="\n\n",
    )
    nodes = parser.get_nodes_from_documents(docs)
    priority = "high" if collection in ("slurm", "institute") else "normal"
    for i, node in enumerate(nodes):
        node.metadata.update({
            "source_file": filename,
            "collection": collection,
            "priority": priority,
            "chunk_index": i,
            "total_chunks": len(nodes),
        })
        node.excluded_llm_metadata_keys = ["chunk_index", "total_chunks"]
    return nodes


# ── App lifecycle ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all DB tables if they don't exist yet (idempotent — safe for dev)
    # In production, run: alembic upgrade head
    Base.metadata.create_all(bind=engine)

    index, llm, embed, chroma_client = _build_rag_index()
    app.state.index         = index
    app.state.llm           = llm
    app.state.embed         = embed
    app.state.chroma_client = chroma_client
    yield
    with _engines_lock:
        _engines.clear()


app = FastAPI(title="Aria API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:    str
    session_id: Optional[str] = None
    user_id:    str = "dev"     # LDAP uid — hardcoded default until Phase 3 LDAP auth


class ChatResponse(BaseModel):
    response:   str
    session_id: str
    message_id: str


class FeedbackRequest(BaseModel):
    message_id: str
    user_id:    str = "dev"
    rating:     int             # must be +1 or -1
    comment:    Optional[str] = None


class SessionSummary(BaseModel):
    session_id: str
    title:      Optional[str]
    created_at: str
    last_active_at: str
    message_count: int


class MessageOut(BaseModel):
    message_id: str
    role:       str
    content:    str
    created_at: str
    latency_ms: Optional[int] = None


class IngestResponse(BaseModel):
    status:       str
    chunks_added: int
    filename:     str
    doc_id:       str


class DocOut(BaseModel):
    doc_id:       str
    filename:     str
    collection:   str
    chunk_count:  int
    file_size_bytes: Optional[int]
    ingested_at:  str
    is_active:    bool


class StatsOut(BaseModel):
    total_users:    int
    total_sessions: int
    total_messages: int
    total_docs:     int
    total_chunks:   int
    feedback_up:    int
    feedback_down:  int
    avg_latency_ms: Optional[float]
    top_collections: list


# ── Routes ─────────────────────────────────────────────────────────────────

MAX_MESSAGE_CHARS = 32_000


def _source_label(collection: Optional[str], priority: Optional[str], source_url: Optional[str]) -> str:
    """Map raw metadata fields to a human-readable source category for the UI."""
    if source_url:
        try:
            domain = urlparse(source_url).netloc
            return f"Web Source: {domain}"
        except Exception:
            return "Web Source"
    if collection == "slurm":
        return "IGS Slurm Documentation" if priority == "high" else "Official Slurm Documentation"
    if collection == "institute":
        return "IGS Confluence Page"
    return collection or "Knowledge Base"

@app.get("/health")
def health(db: Session = Depends(get_db)):
    # ChromaDB chunk count
    chroma_collection = app.state.chroma_client.get_or_create_collection("slurm")
    chunks = chroma_collection.count()

    # Ollama reachability
    ollama_ok = False
    try:
        r = http_requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        ollama_ok = r.ok
    except Exception:
        pass

    # DB connectivity
    db_ok = False
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    status = "ok" if (ollama_ok and db_ok) else "degraded"
    return {
        "status":        status,
        "model":         LLM_MODEL,
        "chunks_in_db":  chunks,
        "ollama_ok":     ollama_ok,
        "db_ok":         db_ok,
        "engines_cached": len(_engines),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db), _auth: dict = Depends(require_auth)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(req.message) > MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=400, detail=f"Message too long ({len(req.message)} chars). Limit: {MAX_MESSAGE_CHARS}")
    session_id = req.session_id or str(uuid.uuid4())

    # Ensure chat engine exists for this session (thread-safe LRU access)
    with _engines_lock:
        if session_id not in _engines:
            engine = _new_chat_engine(app.state.index, app.state.llm)
            # Rebuild conversation context from DB on cache miss (restart / eviction)
            _restore_engine_memory(engine, session_id, db)
            _engines[session_id] = engine

    # Fetch live cluster data or file contents if the question needs them
    live_tool = needs_live_data(req.message)
    file_action = needs_file_read(req.message)
    context_blocks: list[str] = []
    sources: list[dict] = []

    if live_tool:
        live_data = get_live_data(live_tool, req.user_id, req.message)
        context_blocks.append(f"[Live cluster data]\n{live_data}")
        sources.append({"tool": live_tool, "live": True})

    if file_action:
        file_data = get_file_data(file_action, req.user_id)
        context_blocks.append(f"[File contents]\n{file_data}")
        sources.append({"tool": "file_reader", "action": file_action["action"]})

    if context_blocks:
        llm_message = "\n\n".join(context_blocks) + f"\n\n[User question]\n{req.message}"
    else:
        llm_message = req.message

    with _engines_lock:
        engine = _engines[session_id]
    t0 = time.monotonic()
    response = engine.chat(llm_message)
    latency_ms = int((time.monotonic() - t0) * 1000)
    response_text = str(response)

    # Capture which RAG chunks the engine used for this answer
    rag_sources = []
    for node in (getattr(response, "source_nodes", None) or []):
        meta = (node.node.metadata or {}) if hasattr(node, "node") else {}
        col  = meta.get("collection")
        pri  = meta.get("priority")
        url  = meta.get("source_url")
        rag_sources.append({
            "type":        "rag",
            "label":       _source_label(col, pri, url),
            "chunk_text":  node.node.text[:300] if hasattr(node, "node") else str(node)[:300],
            "score":       round(node.score, 4) if node.score is not None else None,
            "source_file": meta.get("source_file"),
            "collection":  col,
        })
    all_sources = sources + rag_sources

    # Persist to PostgreSQL
    user    = _get_or_create_user(db, req.user_id)
    session = _get_or_create_session(db, session_id, user, req.message)
    msg     = _persist_exchange(db, session, req.message, response_text, latency_ms, LLM_MODEL)
    if all_sources:
        msg.sources_used = all_sources

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        message_id=str(msg.id),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, _auth: dict = Depends(require_auth)):
    """SSE streaming version of /chat. Returns text/event-stream with typed JSON events:
    - {type:'meta', session_id} — sent first, carries the resolved session_id
    - {type:'token', text}      — one per LLM token as it's generated
    - {type:'done', message_id, latency_ms} — final event after DB persist
    - {type:'error', message}   — if something goes wrong mid-stream
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(req.message) > MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=400, detail=f"Message too long ({len(req.message)} chars). Limit: {MAX_MESSAGE_CHARS}")
    session_id = req.session_id or str(uuid.uuid4())

    with _engines_lock:
        if session_id not in _engines:
            # For streaming, open a short-lived DB session to restore memory
            _db = SessionLocal()
            try:
                engine = _new_chat_engine(app.state.index, app.state.llm)
                _restore_engine_memory(engine, session_id, _db)
                _engines[session_id] = engine
            finally:
                _db.close()

    # Fetch live context before streaming starts (these are fast network calls)
    live_tool   = needs_live_data(req.message)
    file_action = needs_file_read(req.message)
    context_blocks: list[str] = []
    sources: list[dict] = []

    if live_tool:
        context_blocks.append(f"[Live cluster data]\n{get_live_data(live_tool, req.user_id, req.message)}")
        sources.append({"tool": live_tool, "live": True})
    if file_action:
        context_blocks.append(f"[File contents]\n{get_file_data(file_action, req.user_id)}")
        sources.append({"tool": "file_reader", "action": file_action["action"]})

    llm_message = (
        "\n\n".join(context_blocks) + f"\n\n[User question]\n{req.message}"
        if context_blocks else req.message
    )

    # Capture the engine reference before entering the generator (LRU-safe snapshot)
    with _engines_lock:
        _stream_engine = _engines[session_id]

    def generate():
        yield f"data: {json.dumps({'type': 'meta', 'session_id': session_id})}\n\n"

        db = SessionLocal()
        try:
            user    = _get_or_create_user(db, req.user_id)
            session = _get_or_create_session(db, session_id, user, req.message)

            t0 = time.monotonic()
            full_response = ""
            stream_response = _stream_engine.stream_chat(llm_message)

            for token in stream_response.response_gen:
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

            # Capture RAG source chunks used for this response
            rag_sources = []
            for node in (getattr(stream_response, "source_nodes", None) or []):
                meta = (node.node.metadata or {}) if hasattr(node, "node") else {}
                col  = meta.get("collection")
                pri  = meta.get("priority")
                url  = meta.get("source_url")
                rag_sources.append({
                    "type":        "rag",
                    "label":       _source_label(col, pri, url),
                    "chunk_text":  node.node.text[:300] if hasattr(node, "node") else str(node)[:300],
                    "score":       round(node.score, 4) if node.score is not None else None,
                    "source_file": meta.get("source_file"),
                    "collection":  col,
                })

            latency_ms = int((time.monotonic() - t0) * 1000)
            msg = _persist_exchange(db, session, req.message, full_response, latency_ms, LLM_MODEL)
            all_sources = sources + rag_sources
            if all_sources:
                msg.sources_used = all_sources
            db.commit()

            yield f"data: {json.dumps({'type': 'done', 'message_id': str(msg.id), 'latency_ms': latency_ms})}\n\n"
        except Exception as exc:
            db.rollback()
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            db.close()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions", response_model=List[SessionSummary])
def list_sessions(user_id: str = "dev", db: Session = Depends(get_db), _auth: dict = Depends(require_auth)):
    """List all sessions for a user, most recent first."""
    user = db.query(User).filter(User.ldap_uid == user_id).first()
    if not user:
        return []

    sessions = (
        db.query(DBSession)
        .filter(DBSession.user_id == user.id, DBSession.is_archived == False)
        .order_by(DBSession.last_active_at.desc())
        .limit(50)
        .all()
    )

    result = []
    for s in sessions:
        count = db.query(Message).filter(Message.session_id == s.id).count()
        result.append(SessionSummary(
            session_id=str(s.id),
            title=s.title,
            created_at=s.created_at.isoformat(),
            last_active_at=s.last_active_at.isoformat(),
            message_count=count,
        ))
    return result


@app.get("/sessions/{session_id}/messages", response_model=List[MessageOut])
def get_messages(session_id: str, db: Session = Depends(get_db), _auth: dict = Depends(require_auth)):
    """Return full message history for a session, ordered chronologically."""
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    messages = (
        db.query(Message)
        .filter(Message.session_id == sid)
        .order_by(Message.created_at)
        .all()
    )
    return [
        MessageOut(
            message_id=str(m.id),
            role=m.role.value,
            content=m.content,
            created_at=m.created_at.isoformat(),
            latency_ms=m.latency_ms,
        )
        for m in messages
    ]


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db), _auth: dict = Depends(require_auth)):
    """Store a thumbs up (+1) or thumbs down (-1) on an assistant message."""
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating must be +1 or -1")

    try:
        mid = uuid.UUID(req.message_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid message_id format")

    msg = db.query(Message).filter(Message.id == mid).first()
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")

    user = _get_or_create_user(db, req.user_id)

    existing = db.query(Feedback).filter(Feedback.message_id == mid).first()
    if existing:
        existing.rating  = req.rating
        existing.comment = req.comment
    else:
        db.add(Feedback(
            message_id=mid,
            user_id=user.id,
            rating=req.rating,
            comment=req.comment,
        ))

    return {"status": "ok", "message_id": req.message_id, "rating": req.rating}


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    file: UploadFile = File(...),
    user_id: str = "dev",
    collection: str = "slurm",
    db: Session = Depends(get_db),
    _auth: dict = Depends(require_auth),
):
    allowed = {".pdf", ".txt", ".md"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        file_size = dest.stat().st_size
        docs = SimpleDirectoryReader(tmpdir).load_data()

    nodes = _parse_into_nodes(docs, file.filename, collection)
    app.state.index.insert_nodes(nodes)

    user = _get_or_create_user(db, user_id)
    doc_record = KnowledgeDoc(
        filename=file.filename,
        original_name=file.filename,
        collection=collection,
        ingested_by=user.id,
        chunk_count=len(nodes),
        file_size_bytes=file_size,
    )
    db.add(doc_record)
    db.flush()

    return IngestResponse(
        status="ok",
        chunks_added=len(nodes),
        filename=file.filename,
        doc_id=str(doc_record.id),
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str, db: Session = Depends(get_db), _auth: dict = Depends(require_auth)):
    """Clear the in-memory LlamaIndex engine and archive the session in the DB."""
    with _engines_lock:
        _engines.pop(session_id, None)

    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    session = db.query(DBSession).filter(DBSession.id == sid).first()
    if session:
        session.is_archived = True

    return {"status": "archived", "session_id": session_id}


# ── Auth endpoints ─────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
def auth_login(req: LoginRequest):
    expected = DEMO_USERS.get(req.username)
    if not expected or expected != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    payload = {
        "sub": req.username,
        "role": "admin" if req.username == "admin" else "user",
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_H),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"access_token": token, "token_type": "bearer", "username": req.username, "role": payload["role"]}


@app.get("/auth/me")
def auth_me(_auth: dict = Depends(require_auth)):
    return _auth


# ── URL ingestion ──────────────────────────────────────────────────────────

class IngestUrlRequest(BaseModel):
    url:        str
    collection: str = "slurm"
    user_id:    str = "dev"


@app.post("/ingest/url", response_model=IngestResponse)
def ingest_url(req: IngestUrlRequest, db: Session = Depends(get_db), _auth: dict = Depends(require_auth)):
    try:
        resp = http_requests.get(req.url, timeout=30, headers={"User-Agent": "AriaBot/1.0"})
        resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {exc}")

    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)

    if len(text) < 100:
        raise HTTPException(
            status_code=400,
            detail="Scraped content too short — page may require JavaScript or login",
        )

    from llama_index.core import Document
    parsed   = urlparse(req.url)
    filename = f"{parsed.netloc}{parsed.path.replace('/', '_')[:60]}"
    docs     = [Document(text=text, metadata={"source_url": req.url})]
    nodes    = _parse_into_nodes(docs, filename, req.collection)
    app.state.index.insert_nodes(nodes)

    user_id = _auth["username"] if _auth["username"] != "api" else req.user_id
    user    = _get_or_create_user(db, user_id)
    doc_record = KnowledgeDoc(
        filename=filename,
        original_name=req.url,
        collection=req.collection,
        source_url=req.url,
        ingested_by=user.id,
        chunk_count=len(nodes),
    )
    db.add(doc_record)
    db.flush()

    return IngestResponse(
        status="ok",
        chunks_added=len(nodes),
        filename=filename,
        doc_id=str(doc_record.id),
    )


# ── Admin endpoints ────────────────────────────────────────────────────────

@app.get("/admin/docs", response_model=List[DocOut])
def admin_list_docs(
    collection: Optional[str] = None,
    active_only: bool = True,
    db: Session = Depends(get_db),
    _auth: dict = Depends(require_auth),
):
    q = db.query(KnowledgeDoc)
    if active_only:
        q = q.filter(KnowledgeDoc.is_active == True)
    if collection:
        q = q.filter(KnowledgeDoc.collection == collection)
    docs = q.order_by(KnowledgeDoc.ingested_at.desc()).all()
    return [
        DocOut(
            doc_id=str(d.id),
            filename=d.original_name,
            collection=d.collection,
            chunk_count=d.chunk_count,
            file_size_bytes=d.file_size_bytes,
            ingested_at=d.ingested_at.isoformat(),
            is_active=d.is_active,
        )
        for d in docs
    ]


@app.delete("/admin/docs/{doc_id}")
def admin_delete_doc(
    doc_id: str,
    db: Session = Depends(get_db),
    _auth: dict = Depends(require_auth),
):
    try:
        did = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid doc_id format")

    doc = db.query(KnowledgeDoc).filter(KnowledgeDoc.id == did).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.is_active = False
    db.commit()
    return {"status": "deactivated", "doc_id": doc_id, "filename": doc.original_name}


@app.get("/admin/stats", response_model=StatsOut)
def admin_stats(db: Session = Depends(get_db), _auth: dict = Depends(require_auth)):
    from sqlalchemy import func as sqlfunc

    total_users    = db.query(sqlfunc.count(User.id)).scalar() or 0
    total_sessions = db.query(sqlfunc.count(DBSession.id)).scalar() or 0
    total_messages = db.query(sqlfunc.count(Message.id)).scalar() or 0

    doc_agg = db.query(
        sqlfunc.count(KnowledgeDoc.id),
        sqlfunc.coalesce(sqlfunc.sum(KnowledgeDoc.chunk_count), 0),
    ).filter(KnowledgeDoc.is_active == True).one()
    total_docs, total_chunks = doc_agg

    feedback_up   = db.query(sqlfunc.count(Feedback.id)).filter(Feedback.rating == 1).scalar() or 0
    feedback_down = db.query(sqlfunc.count(Feedback.id)).filter(Feedback.rating == -1).scalar() or 0

    avg_latency = db.query(sqlfunc.avg(Message.latency_ms)).filter(
        Message.role == MessageRole.assistant,
        Message.latency_ms != None,
    ).scalar()

    collections = db.query(
        KnowledgeDoc.collection,
        sqlfunc.count(KnowledgeDoc.id).label("doc_count"),
        sqlfunc.coalesce(sqlfunc.sum(KnowledgeDoc.chunk_count), 0).label("chunk_count"),
    ).filter(KnowledgeDoc.is_active == True).group_by(KnowledgeDoc.collection).all()

    return StatsOut(
        total_users=total_users,
        total_sessions=total_sessions,
        total_messages=total_messages,
        total_docs=total_docs,
        total_chunks=total_chunks,
        feedback_up=feedback_up,
        feedback_down=feedback_down,
        avg_latency_ms=round(avg_latency, 1) if avg_latency else None,
        top_collections=[
            {"collection": c.collection, "docs": c.doc_count, "chunks": c.chunk_count}
            for c in collections
        ],
    )
