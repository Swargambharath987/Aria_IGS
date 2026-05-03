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
from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole as LLMMessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from db.models import Base, Feedback, KnowledgeDoc, Message, MessageRole, Session as DBSession, User, UserRole
from db.session import SessionLocal, engine, get_db
from tools.agent_tools import build_tools

# ── Config ─────────────────────────────────────────────────────────────────
CHROMA_DIR  = Path(os.environ.get("CHROMA_DIR",  "/app/data/chroma_db"))
PROMPT_PATH = Path(os.environ.get("PROMPT_PATH", "/app/prompts/system_prompt.txt"))

# Embeddings always run on Ollama (lightweight, local, no GPU needed)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# LLM — OpenAI-compatible endpoint. Works with Ollama (/v1), vLLM, or any
# other OpenAI-API-compatible server. Change the URL + model, nothing else.
# Local dev:   LLM_BASE_URL=http://host.docker.internal:11434/v1  (Ollama)
# Production:  LLM_BASE_URL=http://igs-gpu:8080/v1               (vLLM)
LLM_BASE_URL      = os.environ.get("LLM_BASE_URL",      "http://localhost:11434/v1")
LLM_MODEL         = os.environ.get("LLM_MODEL",         "gemma4:e4b")
LLM_API_KEY       = os.environ.get("LLM_API_KEY",       "not-needed")
LLM_CONTEXT_WINDOW = int(os.environ.get("LLM_CONTEXT_WINDOW", "32768"))

_raw_tokens  = os.environ.get("API_TOKENS", "igs-dev-token")
VALID_TOKENS = {t.strip() for t in _raw_tokens.split(",") if t.strip()}

# JWT config — demo stub until LDAP/AD is wired (Phase 3)
JWT_SECRET    = os.environ.get("JWT_SECRET", "aria-dev-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_H  = 24

# Demo users: "admin:pass,user1:pass1" — replaced by LDAP once IT provides AD
DEMO_USERS: dict[str, str] = {}
for _entry in os.environ.get("DEMO_USERS", "admin:aria_demo").split(","):
    if ":" in _entry:
        _u, _p = _entry.strip().split(":", 1)
        DEMO_USERS[_u] = _p

# Per-session conversation memory (ChatMemoryBuffer objects), capped at 50.
# The agent itself is stateless and rebuilt per request; only memory is cached.
_memories: cachetools.LRUCache = cachetools.LRUCache(maxsize=50)
_memories_lock: threading.Lock = threading.Lock()


# ── Auth ───────────────────────────────────────────────────────────────────

def require_auth(authorization: Optional[str] = Header(None)) -> dict:
    """Accept a static API token (scripts/curl) or a user JWT (browser login)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token in VALID_TOKENS:
        return {"username": "api", "role": "admin"}
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"username": payload["sub"], "role": payload.get("role", "user")}
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid or expired token")


# ── DB helpers ─────────────────────────────────────────────────────────────

def _get_or_create_user(db: Session, ldap_uid: str) -> User:
    user = db.query(User).filter(User.ldap_uid == ldap_uid).first()
    if not user:
        user = User(ldap_uid=ldap_uid, display_name=ldap_uid, role=UserRole.user)
        db.add(user)
        db.flush()
    return user


def _get_or_create_session(db: Session, session_id: str, user: User, first_message: str) -> DBSession:
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
    db.add(Message(session_id=session.id, role=MessageRole.user, content=user_message))
    assistant_msg = Message(
        session_id=session.id,
        role=MessageRole.assistant,
        content=assistant_response,
        model_used=model,
        latency_ms=latency_ms,
    )
    db.add(assistant_msg)
    from sqlalchemy.sql import func
    session.last_active_at = func.now()
    db.flush()
    return assistant_msg


# ── LlamaIndex setup ───────────────────────────────────────────────────────

def _load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text().strip()
    return (
        "You are Aria, the IGS research computing assistant. "
        "Help researchers with their Slurm jobs, bioinformatics workflows, "
        "cluster usage, and coding. Be concise, precise, and always cite your sources."
    )


def _build_llm() -> OpenAILike:
    """Single LLM factory — works with Ollama /v1 locally and vLLM in production."""
    return OpenAILike(
        model=LLM_MODEL,
        api_base=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        request_timeout=120.0,
        is_chat_model=True,
        context_window=LLM_CONTEXT_WINDOW,
    )


def _build_rag_index(llm: OpenAILike):
    embed = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_HOST)
    Settings.llm         = llm
    Settings.embed_model = embed

    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("slurm")
    store      = ChromaVectorStore(chroma_collection=collection)
    ctx        = StorageContext.from_defaults(vector_store=store)
    index      = VectorStoreIndex.from_vector_store(store, storage_context=ctx, embed_model=embed)
    return index, embed, client


# ── Priority metadata reranker ─────────────────────────────────────────────

class PriorityReranker(BaseNodePostprocessor):
    """Boost nodes tagged priority='high' by +0.15 and re-sort descending."""

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


# ── Hybrid retriever ───────────────────────────────────────────────────────

def _build_hybrid_retriever(index: VectorStoreIndex) -> QueryFusionRetriever:
    """
    Build a BM25 + vector retriever fused via reciprocal-rank fusion.
    Falls back to semantic-only if llama-index-retrievers-bm25 is not installed.
    Built once at startup and cached on app.state — rebuilt after each ingest.
    """
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=8)
    retrievers = [vector_retriever]
    mode = "semantic-only"

    try:
        from llama_index.retrievers.bm25 import BM25Retriever  # type: ignore[import]
        all_nodes = index.vector_store.get_nodes(node_ids=None)
        if all_nodes:
            bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=8)
            retrievers.append(bm25_retriever)
            mode = "hybrid"
        else:
            logger.warning("BM25: ChromaDB returned 0 nodes — falling back to semantic-only")
    except ImportError:
        logger.info("llama-index-retrievers-bm25 not installed; using semantic-only retrieval")

    logger.info("RAG retriever mode: %s", mode)
    return QueryFusionRetriever(
        retrievers=retrievers,
        similarity_top_k=8,
        num_queries=1,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=False,
    )


# ── Agent memory ───────────────────────────────────────────────────────────

def _restore_agent_memory(memory: ChatMemoryBuffer, session_id: str, db: Session) -> None:
    """Populate a fresh ChatMemoryBuffer with prior messages from PostgreSQL.

    Called on cache miss (container restart or LRU eviction) so the agent
    has full conversation history rather than starting blank.
    """
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        return

    messages = (
        db.query(Message)
        .filter(Message.session_id == sid)
        .order_by(Message.created_at)
        .all()
    )
    if not messages:
        return

    _role_map = {
        MessageRole.user:      LLMMessageRole.USER,
        MessageRole.assistant: LLMMessageRole.ASSISTANT,
        MessageRole.tool:      LLMMessageRole.TOOL,
    }
    chat_history = [
        ChatMessage(role=_role_map.get(m.role, LLMMessageRole.USER), content=m.content)
        for m in messages
    ]
    memory.set(chat_history)


def _get_or_create_memory(session_id: str, db: Session) -> ChatMemoryBuffer:
    """Return the cached ChatMemoryBuffer for this session, or create one from DB."""
    with _memories_lock:
        if session_id not in _memories:
            memory = ChatMemoryBuffer(token_limit=LLM_CONTEXT_WINDOW // 8)
            _restore_agent_memory(memory, session_id, db)
            _memories[session_id] = memory
        return _memories[session_id]


# ── Source label ───────────────────────────────────────────────────────────

def _source_label(collection: Optional[str], priority: Optional[str], source_url: Optional[str]) -> str:
    """Map raw chunk metadata to a human-readable source category for the UI."""
    if source_url:
        try:
            return f"Web Source: {urlparse(source_url).netloc}"
        except Exception:
            return "Web Source"
    if collection == "slurm":
        return "IGS Slurm Documentation" if priority == "high" else "Official Slurm Documentation"
    if collection == "institute":
        return "IGS Confluence Page"
    return collection or "Knowledge Base"


# ── Document ingestion helper ──────────────────────────────────────────────

def _parse_into_nodes(docs: list, filename: str, collection: str) -> list:
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=64, paragraph_separator="\n\n")
    nodes = parser.get_nodes_from_documents(docs)
    priority = "high" if collection in ("slurm", "institute") else "normal"
    for i, node in enumerate(nodes):
        node.metadata.update({
            "source_file":   filename,
            "collection":    collection,
            "priority":      priority,
            "chunk_index":   i,
            "total_chunks":  len(nodes),
        })
        node.excluded_llm_metadata_keys = ["chunk_index", "total_chunks"]
    return nodes


# ── App lifecycle ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    llm                  = _build_llm()
    index, embed, client = _build_rag_index(llm)
    retriever            = _build_hybrid_retriever(index)

    app.state.llm          = llm
    app.state.index        = index
    app.state.embed        = embed
    app.state.chroma_client = client
    app.state.retriever    = retriever

    logger.info("Aria backend ready — LLM: %s @ %s | embed: %s", LLM_MODEL, LLM_BASE_URL, EMBED_MODEL)
    yield
    with _memories_lock:
        _memories.clear()


app = FastAPI(title="Aria API", version="3.0.0", lifespan=lifespan)

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
    user_id:    str = "dev"


class ChatResponse(BaseModel):
    response:   str
    session_id: str
    message_id: str


class FeedbackRequest(BaseModel):
    message_id: str
    user_id:    str = "dev"
    rating:     int
    comment:    Optional[str] = None


class SessionSummary(BaseModel):
    session_id:     str
    title:          Optional[str]
    created_at:     str
    last_active_at: str
    message_count:  int


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
    doc_id:          str
    filename:        str
    collection:      str
    chunk_count:     int
    file_size_bytes: Optional[int]
    ingested_at:     str
    is_active:       bool


class StatsOut(BaseModel):
    total_users:     int
    total_sessions:  int
    total_messages:  int
    total_docs:      int
    total_chunks:    int
    feedback_up:     int
    feedback_down:   int
    avg_latency_ms:  Optional[float]
    top_collections: list


# ── Routes ─────────────────────────────────────────────────────────────────

MAX_MESSAGE_CHARS = 32_000


@app.get("/health")
def health(db: Session = Depends(get_db)):
    chroma_collection = app.state.chroma_client.get_or_create_collection("slurm")
    chunks = chroma_collection.count()

    llm_ok = False
    try:
        r = http_requests.get(
            LLM_BASE_URL.rstrip("/v1").rstrip("/") + "/v1/models",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            timeout=3,
        )
        llm_ok = r.ok
    except Exception:
        pass

    db_ok = False
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    status = "ok" if (llm_ok and db_ok) else "degraded"
    return {
        "status":          status,
        "model":           LLM_MODEL,
        "llm_base_url":    LLM_BASE_URL,
        "chunks_in_db":    chunks,
        "llm_ok":          llm_ok,
        "db_ok":           db_ok,
        "sessions_cached": len(_memories),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Session = Depends(get_db), _auth: dict = Depends(require_auth)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(req.message) > MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=400, detail=f"Message too long ({len(req.message)} chars). Limit: {MAX_MESSAGE_CHARS}")

    session_id = req.session_id or str(uuid.uuid4())
    username   = _auth.get("username") if _auth.get("username") != "api" else req.user_id

    # Source sink — tool closures append to this list as they execute
    source_sink: list = []

    # Build agent fresh per request (stateless; only memory is cached)
    tools  = build_tools(username, app.state.retriever, source_sink, _source_label)
    agent  = ReActAgent(
        tools=tools,
        llm=app.state.llm,
        system_prompt=_load_system_prompt(),
        verbose=False,
    )
    memory = _get_or_create_memory(session_id, db)

    t0 = time.monotonic()
    handler  = agent.run(req.message, memory=memory)
    result   = await handler
    latency_ms   = int((time.monotonic() - t0) * 1000)
    response_text = result.response

    user    = _get_or_create_user(db, username)
    session = _get_or_create_session(db, session_id, user, req.message)
    msg     = _persist_exchange(db, session, req.message, response_text, latency_ms, LLM_MODEL)
    if source_sink:
        msg.sources_used = source_sink
    db.commit()

    return ChatResponse(response=response_text, session_id=session_id, message_id=str(msg.id))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, _auth: dict = Depends(require_auth)):
    """SSE streaming. Events: {type:'meta'}, {type:'token'}, {type:'done'}, {type:'error'}"""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(req.message) > MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=400, detail=f"Message too long ({len(req.message)} chars). Limit: {MAX_MESSAGE_CHARS}")

    session_id = req.session_id or str(uuid.uuid4())
    username   = _auth.get("username") if _auth.get("username") != "api" else req.user_id
    source_sink: list = []

    # Build agent and restore memory before streaming starts (errors become HTTP errors)
    tools  = build_tools(username, app.state.retriever, source_sink, _source_label)
    agent  = ReActAgent(
        tools=tools,
        llm=app.state.llm,
        system_prompt=_load_system_prompt(),
        streaming=True,
        verbose=False,
    )

    # Memory restore needs a DB session; use a short-lived one before the generator
    _db = SessionLocal()
    try:
        memory = _get_or_create_memory(session_id, _db)
    finally:
        _db.close()

    async def generate():
        yield f"data: {json.dumps({'type': 'meta', 'session_id': session_id})}\n\n"

        db = SessionLocal()
        try:
            user    = _get_or_create_user(db, username)
            session = _get_or_create_session(db, session_id, user, req.message)

            t0 = time.monotonic()
            full_response = ""

            handler = agent.run(req.message, memory=memory)
            async for event in handler.stream_events():
                if hasattr(event, "delta") and event.delta:
                    full_response += event.delta
                    yield f"data: {json.dumps({'type': 'token', 'text': event.delta})}\n\n"

            result = await handler
            if not full_response:
                full_response = result.response

            latency_ms = int((time.monotonic() - t0) * 1000)
            msg = _persist_exchange(db, session, req.message, full_response, latency_ms, LLM_MODEL)
            if source_sink:
                msg.sources_used = source_sink
            db.commit()

            yield f"data: {json.dumps({'type': 'done', 'message_id': str(msg.id), 'latency_ms': latency_ms})}\n\n"

        except Exception as exc:
            db.rollback()
            logger.exception("Stream error for session %s", session_id)
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
        db.add(Feedback(message_id=mid, user_id=user.id, rating=req.rating, comment=req.comment))

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

    # Rebuild retriever so new docs are immediately searchable via BM25
    app.state.retriever = _build_hybrid_retriever(app.state.index)

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

    return IngestResponse(status="ok", chunks_added=len(nodes), filename=file.filename, doc_id=str(doc_record.id))


@app.delete("/session/{session_id}")
def clear_session(session_id: str, db: Session = Depends(get_db), _auth: dict = Depends(require_auth)):
    with _memories_lock:
        _memories.pop(session_id, None)

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
        "sub":  req.username,
        "role": "admin" if req.username == "admin" else "user",
        "exp":  datetime.utcnow() + timedelta(hours=JWT_EXPIRE_H),
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

    # Rebuild retriever after ingest so BM25 includes new nodes
    app.state.retriever = _build_hybrid_retriever(app.state.index)

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

    return IngestResponse(status="ok", chunks_added=len(nodes), filename=filename, doc_id=str(doc_record.id))


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
def admin_delete_doc(doc_id: str, db: Session = Depends(get_db), _auth: dict = Depends(require_auth)):
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
