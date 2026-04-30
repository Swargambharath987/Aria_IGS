import json
import os
import time
import uuid
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import chromadb
from fastapi import Depends, FastAPI, File, HTTPException, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel
from sqlalchemy.orm import Session

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

_raw_tokens  = os.environ.get("API_TOKENS", "igs-dev-token")
VALID_TOKENS = {t.strip() for t in _raw_tokens.split(",") if t.strip()}

# In-memory store for LlamaIndex chat engines (stateful objects, not serializable)
# Key: session_id (str UUID), Value: chat engine instance
_engines: dict = {}


# ── Auth ───────────────────────────────────────────────────────────────────
def require_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token not in VALID_TOKENS:
        raise HTTPException(status_code=403, detail="Invalid token")
    return token


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
    embed = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_HOST)
    llm   = Ollama(model=LLM_MODEL, base_url=OLLAMA_HOST, request_timeout=120.0)
    Settings.llm         = llm
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


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    collection = app.state.chroma_client.get_or_create_collection("slurm")
    return {"status": "ok", "model": LLM_MODEL, "chunks_in_db": collection.count()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db), _token: str = Depends(require_token)):
    session_id = req.session_id or str(uuid.uuid4())

    # Ensure chat engine exists for this session
    if session_id not in _engines:
        _engines[session_id] = _new_chat_engine(app.state.index, app.state.llm)

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

    t0 = time.monotonic()
    response = _engines[session_id].chat(llm_message)
    latency_ms = int((time.monotonic() - t0) * 1000)
    response_text = str(response)

    # Persist to PostgreSQL
    user    = _get_or_create_user(db, req.user_id)
    session = _get_or_create_session(db, session_id, user, req.message)
    msg     = _persist_exchange(db, session, req.message, response_text, latency_ms, LLM_MODEL)
    if sources:
        msg.sources_used = sources

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        message_id=str(msg.id),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, _token: str = Depends(require_token)):
    """SSE streaming version of /chat. Returns text/event-stream with typed JSON events:
    - {type:'meta', session_id} — sent first, carries the resolved session_id
    - {type:'token', text}      — one per LLM token as it's generated
    - {type:'done', message_id, latency_ms} — final event after DB persist
    - {type:'error', message}   — if something goes wrong mid-stream
    """
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in _engines:
        _engines[session_id] = _new_chat_engine(app.state.index, app.state.llm)

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

    def generate():
        yield f"data: {json.dumps({'type': 'meta', 'session_id': session_id})}\n\n"

        db = SessionLocal()
        try:
            user    = _get_or_create_user(db, req.user_id)
            session = _get_or_create_session(db, session_id, user, req.message)

            t0 = time.monotonic()
            full_response = ""

            for token in _engines[session_id].stream_chat(llm_message).response_gen:
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

            latency_ms = int((time.monotonic() - t0) * 1000)
            msg = _persist_exchange(db, session, req.message, full_response, latency_ms, LLM_MODEL)
            if sources:
                msg.sources_used = sources
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
def list_sessions(user_id: str = "dev", db: Session = Depends(get_db), _token: str = Depends(require_token)):
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
def get_messages(session_id: str, db: Session = Depends(get_db), _token: str = Depends(require_token)):
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
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db), _token: str = Depends(require_token)):
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
    _token: str = Depends(require_token),
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
def clear_session(session_id: str, db: Session = Depends(get_db), _token: str = Depends(require_token)):
    """Clear the in-memory LlamaIndex engine and archive the session in the DB."""
    _engines.pop(session_id, None)

    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    session = db.query(DBSession).filter(DBSession.id == sid).first()
    if session:
        session.is_archived = True

    return {"status": "archived", "session_id": session_id}
