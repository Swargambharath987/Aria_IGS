# IGS Grid AI — Developer Log

This file is a running log of all sessions, decisions, errors, and next steps.
Intended as a self-reference for future sessions — read this first before touching anything.

---

## Session 1 — Demo Build (pre-Apr 16)

### What was built
- Full RAG pipeline: Ollama + Qwen 2.5 7B + ChromaDB + LlamaIndex + Streamlit
- Knowledge base: IGS SLURM Grid Scheduler Overview PDF + SLURM Troubleshooting PDF + Slurm 23.11.6 official docs (84 pages scraped)
- 1019 chunks ingested into ChromaDB collection named `"slurm"`
- Ran locally on MacBook Pro M4 24GB RAM
- Demo shown at Optimus meeting Apr 16, 2026

### Exact package versions (main venv — slurm_ai, Python 3.14)
```
chromadb==1.5.7
pydantic==2.13.0
llama-index-core==0.14.20
llama-index-llms-ollama==0.10.1
llama-index-embeddings-ollama==0.9.0
llama-index-vector-stores-chroma==0.5.5
streamlit (latest at time)
pypdf
requests
beautifulsoup4
```

### Ollama models in use
- LLM: `qwen2.5:7b`
- Embedding: `nomic-embed-text`

### Key file locations
```
~/slurm-grid-ai/
  app/streamlit_app.py       — Streamlit UI
  rag/query_engine.py        — LlamaIndex chat engine builder + is_slurm_related()
  ingest/ingest.py           — PDF parser + Slurm doc scraper + ChromaDB ingestion
  prompts/system_prompt.txt  — Full system prompt with guardrails + few-shot examples
  data/chroma_db/            — Persisted ChromaDB (bind-mounted in Docker, not named volume)
  data/raw/                  — Source PDFs
  pipelines/                 — Open WebUI pipeline files (added Session 2)
  docker-compose.yml
  requirements.txt
```

### ChromaDB path
- Local (native): `~/slurm-grid-ai/data/chroma_db/`
- In Docker containers: bind-mounted at `/app/data/chroma_db`
- CRITICAL: This must be a bind mount (`./data/chroma_db:/app/data/chroma_db`), NOT a named Docker volume. A named volume creates empty storage and ignores existing data.

---

## Session 2 — Open WebUI + Pipelines Integration (Apr 19, 2026)

### Goal
Replace Streamlit with Open WebUI as the chat interface, keeping the existing LlamaIndex + ChromaDB RAG backend. Open WebUI Pipelines used as the bridge.

### Architecture
```
User browser → Open WebUI (port 3000)
                    ↓ HTTP (internal Docker network)
             Pipelines server (port 9099)
                    ↓ imports
             LlamaIndex chat engine
                    ↓ reads
             ChromaDB at ./data/chroma_db (bind mount)
                    ↓ queries
             Ollama on host (port 11434, M4 Metal acceleration)
```

### IMPORTANT: Ollama runs natively on macOS, NOT in Docker
Docker on Mac runs in a Linux VM. A Dockerized Ollama loses Metal/Neural Engine acceleration — responses are 10-20x slower. Ollama must run natively on the host. Containers reach it via `host.docker.internal:11434`.

### docker-compose.yml (final working state)
```yaml
services:

  app:
    build: .
    container_name: slurm_ai_app
    ports:
      - "8501:8501"
    volumes:
      - ./data/chroma_db:/app/data/chroma_db
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434

  pipelines:
    image: ghcr.io/open-webui/pipelines:main
    container_name: slurm_ai_pipelines
    ports:
      - "9099:9099"
    volumes:
      - ./pipelines:/app/pipelines
      - ./prompts:/app/prompts
      - ./data/chroma_db:/app/data/chroma_db
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
      - PIPELINES_API_KEY=igs-local-only
      - PIPELINES_REQUIREMENTS_PATH=/app/pipelines/requirements.txt
    restart: unless-stopped

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: slurm_ai_webui
    ports:
      - "3000:8080"
    volumes:
      - open_webui_data:/app/backend/data
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - OPENAI_API_BASE_URL=http://pipelines:9099
      - OPENAI_API_KEY=igs-local-only
    depends_on:
      - pipelines
    restart: unless-stopped

volumes:
  open_webui_data:
```

### How Open WebUI connects to Pipelines
`OPENAI_API_BASE_URL` and `OPENAI_API_KEY` in Open WebUI point to the Pipelines container. This is purely internal Docker network traffic — nothing contacts OpenAI servers. The "OpenAI" label is the protocol format used between these two local containers. The key `igs-local-only` is a local password matching `PIPELINES_API_KEY` in the Pipelines container.

### pipelines/requirements.txt
CRITICAL: Must pin to exact same versions as the main venv to avoid ChromaDB format mismatch.
```
chromadb==1.5.7
llama-index-core==0.14.20
llama-index-llms-ollama==0.10.1
llama-index-embeddings-ollama==0.9.0
llama-index-vector-stores-chroma==0.5.5
```
No pydantic pin needed — pydantic 2.13.0 works fine with this stack.

### pipelines/igs_rag_pipeline.py
The pipeline reads from the existing ChromaDB (same data, same collection `"slurm"`), loads the system prompt from `/app/prompts/system_prompt.txt`, and applies the same guardrail and greeting logic as the Streamlit app.

---

## Errors Hit in Session 2 — with fixes

### Error 1: Ollama container healthcheck failing
```
dependency failed to start: container slurm_ai_ollama is unhealthy
```
**Cause:** `curl` is not in the Ollama Docker image.
**Fix:** Changed healthcheck to `["CMD", "ollama", "list"]`. But ultimately removed the Ollama container entirely (see Session 2 architecture note above).

### Error 2: ChromaDB empty in Docker
Pipelines container couldn't find any data.
**Cause:** Used a named Docker volume (`chroma_data`) which is separate from the local filesystem. The ingested data lives at `./data/chroma_db` on the host.
**Fix:** Changed all `chroma_data:/app/data/chroma_db` to `./data/chroma_db:/app/data/chroma_db` (bind mount). Removed `chroma_data` from the volumes section.

### Error 3: Pydantic import crash on first run
```
ImportError: cannot import name 'ArbitraryTypeWarning' from 'pydantic.warnings'
```
**Cause:** The Pipelines container had pydantic partially installed. When llama-index installed on top, it created a mixed/broken state. NOT a permanent version incompatibility — pydantic 2.13.0 works fine.
**Fix:** Used `PIPELINES_REQUIREMENTS_PATH` to pre-install all packages from `pipelines/requirements.txt` before the pipeline is loaded. This avoids the mixed-state problem.

### Error 4: ChromaDB Rust bindings crash
```
thread panicked at rust/sqlite/src/db.rs:157:42:
range start index 10 out of range for slice of length 9
pyo3_runtime.PanicException: range start index 10 out of range for slice of length 9
```
**Cause:** The Pipelines container installed a different version of chromadb than what was used to write the ChromaDB files. chromadb uses Rust-based storage internally and different versions have incompatible on-disk formats.
**Fix:** Pinned `chromadb==1.5.7` (matching the main venv) in `pipelines/requirements.txt`.

### Error 5: Pipeline file moved to failed/ on each crash
When a pipeline fails `on_startup`, the Pipelines server moves the `.py` file to `/app/pipelines/failed/`. It will not be retried until moved back.
**Fix pattern:** `mv pipelines/failed/igs_rag_pipeline.py pipelines/igs_rag_pipeline.py` before restarting.
**Long-term:** Once startup is stable, this won't recur. But if you're debugging startup errors, remember to check `ls pipelines/failed/`.

### Error 6: nomic-embed-text does not support chat
```
400: "nomic-embed-text:latest" does not support chat
```
**Not a bug.** nomic-embed-text is an embedding-only model. Open WebUI lists all Ollama models including embedding models. Never select it as a chat model. It has no chat capability by design.

### Error 7: IGS RAG Pipeline TransferEncodingError (UNRESOLVED — next session)
```
Response payload is not completed: TransferEncodingError 400
```
**Status:** Pipeline loads successfully. Error occurs when a message is sent.
**Likely cause:** The `response_gen` generator from LlamaIndex's `stream_chat` is not compatible with Open WebUI Pipelines' expected streaming format. The Pipelines server expects the generator to yield plain strings in a specific way.
**What to try next:**
- Return a plain string instead of generator (non-streaming) to confirm the pipeline logic works end-to-end
- Then re-add streaming with the correct format once confirmed
- Replace `return self.chat_engine.stream_chat(user_message).response_gen` with a non-streaming test first:
  ```python
  response = self.chat_engine.chat(user_message)
  return str(response)
  ```

---

## Current State (end of Session 2)

| Component | Status |
|---|---|
| Streamlit app (port 8501) | Working (unchanged) |
| Open WebUI (port 3000) | Running |
| Pipelines server (port 9099) | Running, pipeline loads |
| IGS RAG Pipeline in Open WebUI | Loads but streaming error on response |
| Direct qwen2.5:7b in Open WebUI | Working (no RAG) |

---

## Broader Roadmap (from demo feedback + architecture discussions)

### Decided direction
- **Two interfaces only:** Open WebUI (web) + Claude Code-like CLI (agentic)
- **Foundation:** vLLM on GPU server (A100 or H200, 1-2 GPUs from 8 available)
- **Model:** Not locked to Qwen/Llama — evaluate GLM (Zhipu AI), DeepSeek, Mistral at build time. Pick best benchmark performer that supports tool calling.
- **RAG + fine-tuning combined:** Fine-tune for behavior/style, RAG for dynamic knowledge retrieval
- **Tool scope:** Not just Slurm — Python/R coding, script generation, general research computing

### What was explicitly dropped
- Training model from scratch
- Log-based memory prediction
- LangFlow (no-code builder — wrong philosophy for this project)
- LangChain (LlamaIndex already covers this)
- Three interfaces (only two: web + CLI)

### Roadmap goals in order
1. **Fix Open WebUI pipeline streaming** (immediate — next session)
2. **vLLM on GPU** — production inference serving, model-agnostic
3. **FastAPI middleware** — auth, routing, per-user memory, tool dispatch
4. **Open WebUI wired to FastAPI** (not directly to vLLM)
5. **IGS AI CLI** — agentic tool, phases:
   - Phase 1: read-only Slurm commands (squeue, sacct, seff)
   - Phase 2: file-aware (read job scripts, log files)
   - Phase 3: execution (sbatch, scancel) — requires auth design + Mike/Dustin sign-off
6. **Fine-tuning pipeline** — after 200+ labeled Q&A pairs available
7. **Confluence docs ingestion** — Mike to provide pages
8. **Network isolation** — restrict GPU server to IGS internal network only

### GPU planning
- 100 users, ~10% peak concurrency = 10 simultaneous requests
- Single A100 80GB with Qwen/GLM 32B handles this comfortably
- H200 (141GB) can run 70B models
- Second GPU: hot standby or fine-tuning workloads

### Privacy requirements (hard constraints)
- All inference, storage, embeddings stay within IGS network
- Open WebUI data: local Docker volume only
- ChromaDB: bind-mounted to local filesystem
- Ollama/vLLM: hosted on IGS server, no external model API calls
- Network isolation at server level preferred over trusting each dependency

---

## Key Architecture Principles Decided

1. **Ollama always native on host** — never in Docker on Mac (loses Metal acceleration)
2. **ChromaDB always bind-mounted** — never named Docker volume (would be empty)
3. **PIPELINES_REQUIREMENTS_PATH** for dependency installs — not docstring requirements (causes race conditions and mixed state)
4. **All package versions pinned** to match the venv used for ingestion — different chromadb versions = incompatible on-disk format
5. **vLLM for production** — Ollama is development only, doesn't handle multi-user concurrency
6. **Model-agnostic serving** — interface layer should not hard-code Qwen or any model family
