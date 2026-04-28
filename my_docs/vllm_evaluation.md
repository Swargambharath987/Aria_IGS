# vLLM Evaluation — IGS Grid AI

**Date:** April 28, 2026  
**Author:** IGS Grid AI team  
**Status:** Decision pending — adopt now vs at GPU server stage

---

## Summary (TL;DR)

**Recommendation: Hybrid approach — keep Ollama for local dev, adopt vLLM when the GPU server is provisioned.**

Do not switch to vLLM on your M4 MacBook today. vLLM has no meaningful advantage over Ollama on CPU/MPS for single-user development, adds operational complexity before the streaming bug is fixed, and the GPU hardware required to realize its benefits isn't available yet. The code changes to switch are small (< 20 lines), so there is no "rewrite penalty" for waiting.

When the GPU server is provisioned: switch to vLLM then, not before.

---

## 1. What vLLM Solves (That Ollama Does Not)

### The core problem: Ollama serializes requests

Ollama processes one inference request at a time. When request 2 arrives while request 1 is running, it waits. For a single developer running tests, this is invisible. For 10 simultaneous users, the last user in queue waits for all 9 ahead of them to finish — at ~5–10 seconds per request on an A100, that's 45–90 seconds of queuing on top of their own inference time.

### How vLLM solves it

**PagedAttention** — The KV cache (the memory used to store attention context during generation) is allocated in fixed-size pages rather than contiguous blocks. This eliminates memory fragmentation and allows many sequences to share GPU memory efficiently. In practice: instead of allocating a worst-case 4096-token block per request, vLLM allocates only what's actually used, fitting many more concurrent requests in VRAM.

**Continuous batching** — vLLM does not wait for a batch to finish before adding new requests. Requests join and leave the running batch dynamically at the token level. A short request that finishes early frees its slot immediately. Ollama runs each request fully before starting the next.

**Throughput at concurrency** — At 1 user, Ollama and vLLM have nearly identical latency (both are bottlenecked by the model, not the serving infrastructure). At 10 concurrent users, vLLM serves all 10 in parallel; Ollama queues them. For this project's target load (10 concurrent users at peak), vLLM is the correct choice for the GPU server.

### Why Ollama is fine for dev

Single-developer usage hits Ollama at 1 request at a time. Metal acceleration on the M4 makes it fast for development and demo purposes. Ollama is the right tool for local dev — the issue is not its speed, it's its concurrency model.

---

## 2. How vLLM Would Replace Ollama in This Architecture

### Current architecture

```
Open WebUI (port 3000)
    ↓ HTTP (Docker internal)
Pipelines server (port 9099)
    ↓ LlamaIndex imports
LlamaIndex chat engine (llama-index-llms-ollama)
    ↓ HTTP
Ollama on host (port 11434, M4 Metal)
```

### Target architecture (GPU server)

```
Open WebUI (port 3000)
    ↓ HTTP (Docker internal)
Pipelines server (port 9099)
    ↓ LlamaIndex imports
LlamaIndex chat engine (llama-index-llms-openai, OpenAI-compatible)
    ↓ HTTP
vLLM server (port 8000, A100/H200 CUDA)

Embeddings (separate):
Ollama on host OR dedicated embedding container (nomic-embed-text, port 11435)
```

### 2a. Changes to `rag/query_engine.py`

The only change is in the LLM instantiation. Everything else — ChromaDB, LlamaIndex index, chat engine, memory — is unchanged.

**Current:**
```python
from llama_index.llms.ollama import Ollama

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

llm = Ollama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL, request_timeout=120.0)
```

**After (vLLM):**
```python
from llama_index.llms.openai_like import OpenAILike

VLLM_BASE_URL = os.environ.get("VLLM_HOST", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:32b")

llm = OpenAILike(
    model=LLM_MODEL,
    api_base=VLLM_BASE_URL,
    api_key="not-required",          # vLLM accepts any non-empty key
    is_chat_model=True,
    request_timeout=120.0,
)
```

**Why `OpenAILike` and not `llama-index-llms-vllm`?**

There is a `llama_index.llms.vllm` package that directly imports the `vllm` Python library in-process. Do not use it — it requires vLLM installed in the same environment as LlamaIndex, which is impractical in a Docker container meant to run on a CPU node (the Pipelines container). vLLM only installs on CUDA environments.

The correct approach is `OpenAILike`, which speaks to vLLM over HTTP using its OpenAI-compatible REST API. This is how vLLM is designed to be used in multi-service architectures.

**Package change:**
```
# Remove from requirements.txt / pipelines/requirements.txt:
llama-index-llms-ollama==0.10.1

# Add:
llama-index-llms-openai==0.4.x   # provides OpenAILike
```

Check latest compatible version: `pip index versions llama-index-llms-openai`

### 2b. Changes to `pipelines/igs_rag_pipeline.py`

Same pattern — only the LLM instantiation changes. The `OllamaEmbedding` stays because vLLM does not serve embeddings by default (see section 2c).

**Current (lines 43–51):**
```python
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=OLLAMA_BASE_URL)
llm = Ollama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL, request_timeout=120.0)
```

**After (vLLM):**
```python
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai_like import OpenAILike

VLLM_BASE_URL = os.environ.get("VLLM_HOST", "http://vllm:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:32b")

embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434"),
)
llm = OpenAILike(
    model=LLM_MODEL,
    api_base=VLLM_BASE_URL,
    api_key="not-required",
    is_chat_model=True,
    request_timeout=120.0,
)
```

The `OLLAMA_HOST` environment variable continues to point to Ollama on the host (or a dedicated embedding container) for embeddings only. The `VLLM_HOST` variable is new and points to the vLLM container on the GPU server.

### 2c. Embeddings: vLLM vs separate service

vLLM added an `/v1/embeddings` endpoint (OpenAI-compatible) in v0.4.x. You can technically serve embeddings from vLLM, but there are two reasons to keep embeddings separate:

1. **Memory**: Running both the LLM and the embedding model in vLLM on the same GPU wastes A100 VRAM. nomic-embed-text is a small model (< 1GB); loading it into vLLM alongside a 32B parameter LLM is wasteful. Run it in a lightweight container or natively.

2. **Architecture simplicity**: Embedding calls happen at query time and at ingest time. If vLLM is the only embedding endpoint and it goes down for a model swap or restart, ingest breaks too. Keeping nomic-embed-text on Ollama (or a separate `sentence-transformers` container) decouples these.

**Decision for now:** Keep `nomic-embed-text` on Ollama. Revisit only if Ollama's throughput becomes a bottleneck for embedding (unlikely at this scale).

### 2d. Docker / deployment changes

The vLLM server runs as its own container on the GPU server with the model pre-loaded. The existing Pipelines container gets a new environment variable pointing to it.

```yaml
# docker-compose.yml additions (GPU server deployment only)

services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: slurm_ai_vllm
    runtime: nvidia
    ports:
      - "8000:8000"
    volumes:
      - /data/models:/root/.cache/huggingface   # pre-downloaded model cache
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}      # only needed for gated models
    command: >
      --model Qwen/Qwen2.5-32B-Instruct
      --dtype bfloat16
      --max-model-len 8192
      --gpu-memory-utilization 0.90
      --tensor-parallel-size 1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

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
      - OLLAMA_HOST=http://host.docker.internal:11434   # embeddings only
      - VLLM_HOST=http://vllm:8000/v1                  # new: LLM inference
      - LLM_MODEL=Qwen/Qwen2.5-32B-Instruct            # new: model name
      - PIPELINES_API_KEY=igs-local-only
      - PIPELINES_REQUIREMENTS_PATH=/app/pipelines/requirements.txt
    depends_on:
      - vllm
    restart: unless-stopped
```

**Note on model names:** When using Ollama, the model name is `qwen2.5:7b` (Ollama tag). When using vLLM, the model name must match the HuggingFace repo ID: `Qwen/Qwen2.5-32B-Instruct`. These are different strings — the `LLM_MODEL` environment variable handles this correctly.

---

## 3. Now vs Later Analysis

### Arguments for adopting vLLM now

**Avoid the integration rewrite later.** The LlamaIndex code change is 4 lines. But switching means retesting the full pipeline end-to-end, updating `requirements.txt` and `pipelines/requirements.txt`, pinning new package versions, and verifying streaming compatibility with Open WebUI Pipelines. Doing this twice (once now on CPU, once on GPU) is wasteful.

**OpenAI-compatible API means minimal risk.** vLLM's `/v1/chat/completions` endpoint is spec-compatible with what `OpenAILike` in LlamaIndex expects. The integration surface is small and well-tested. There's no "heroic" porting effort here.

**Forces proper architecture early.** Separating the LLM server from the application code is the right design. Doing it now avoids the temptation to keep Ollama-specific coupling in the codebase.

### Arguments for adopting vLLM later

**No GPU server yet — CPU performance is unusable.** vLLM is designed for CUDA GPUs. Running vLLM on CPU is documented as unsupported and extremely slow. On your M4 Mac, the choice is: Ollama (Metal-accelerated, fast) or vLLM-on-CPU (slow, unsupported, requires workarounds). This is not a viable dev environment.

**The streaming bug is the actual blocker.** The immediate problem is Error 7 from Session 2 — `TransferEncodingError 400` when a message is sent through the pipeline. This is a LlamaIndex/Pipelines streaming protocol issue that exists regardless of which LLM backend you use. Switching to vLLM does not fix it, and could introduce new variables that make debugging harder.

**Operational complexity before the system is proven.** vLLM adds a CUDA-runtime dependency, a model download step, GPU driver requirements, and a separate Docker service — all before confirming the pipeline works end-to-end. The correct order is: fix the pipeline, confirm it works, then upgrade the inference backend.

**Risk of package version conflicts.** Switching from `llama-index-llms-ollama` to `llama-index-llms-openai` means re-pinning in both `requirements.txt` and `pipelines/requirements.txt`. Given the ChromaDB version sensitivity already documented, any package churn before the system is stable adds debugging risk.

---

## 4. Recommended Path

### Phase 1: Local dev / current state (now → GPU server provisioned)

**Keep Ollama. Fix the streaming bug first.**

The immediate action is not the LLM backend — it's getting the pipeline to return a response at all. The non-streaming fallback in `igs_rag_pipeline.py` (currently active) should be confirmed working, then streaming re-enabled properly.

Do not switch LLM backends until:
- Pipeline returns a response without errors
- Open WebUI displays the response correctly
- The system has been tested end-to-end at least once

### Phase 2: GPU server provisioned

Switch to vLLM at this point. The switch happens in one git commit:

1. Update `rag/query_engine.py` — replace `Ollama` with `OpenAILike`
2. Update `pipelines/igs_rag_pipeline.py` — same replacement
3. Update `pipelines/requirements.txt` — swap `llama-index-llms-ollama` for `llama-index-llms-openai`
4. Add vLLM service to `docker-compose.yml` (for GPU server deployment only)
5. Add `VLLM_HOST` and `LLM_MODEL` environment variables

The local dev workflow does not change: Ollama continues to run natively on macOS for local testing. The Docker Compose file can support both targets via a separate `docker-compose.gpu.yml` override or a `.env` file that sets `VLLM_HOST`.

### Hybrid dev/prod pattern (recommended long-term)

```
Local dev (M4 Mac):
  VLLM_HOST=            (unset)
  OLLAMA_HOST=http://localhost:11434
  → LlamaIndex uses Ollama (Metal-accelerated)

GPU server (production/staging):
  VLLM_HOST=http://vllm:8000/v1
  OLLAMA_HOST=http://host.docker.internal:11434  (embeddings only)
  LLM_MODEL=Qwen/Qwen2.5-32B-Instruct
  → LlamaIndex uses vLLM (CUDA, concurrent)
```

`query_engine.py` detects which backend to use:

```python
VLLM_HOST = os.environ.get("VLLM_HOST", "")

if VLLM_HOST:
    from llama_index.llms.openai_like import OpenAILike
    LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")
    llm = OpenAILike(
        model=LLM_MODEL,
        api_base=VLLM_HOST,
        api_key="not-required",
        is_chat_model=True,
        request_timeout=120.0,
    )
else:
    from llama_index.llms.ollama import Ollama
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    llm = Ollama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL, request_timeout=120.0)
```

This pattern means the same codebase runs on your Mac today and on the GPU server tomorrow, without any changes.

---

## 5. Concrete Code Changes

### `rag/query_engine.py` — full updated file

```python
import os
import chromadb
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
PROMPT_PATH = BASE_DIR / "prompts" / "system_prompt.txt"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
VLLM_HOST = os.environ.get("VLLM_HOST", "")


def _build_llm():
    if VLLM_HOST:
        from llama_index.llms.openai_like import OpenAILike
        return OpenAILike(
            model=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
            api_base=VLLM_HOST,
            api_key="not-required",
            is_chat_model=True,
            request_timeout=120.0,
        )
    from llama_index.llms.ollama import Ollama
    return Ollama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL, request_timeout=120.0)


def load_system_prompt():
    with open(PROMPT_PATH, "r") as f:
        return f.read().strip()


def build_chat_engine():
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    llm = _build_llm()

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection("slurm")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    system_prompt = load_system_prompt()

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=llm,
        memory=memory,
        system_prompt=system_prompt,
        similarity_top_k=5,
        verbose=False,
    )
    return chat_engine


def is_slurm_related(query: str) -> bool:
    keywords = [
        "slurm", "sbatch", "srun", "salloc", "squeue", "scancel", "sinfo",
        "sacct", "sstat", "seff", "job", "grid", "cluster", "node", "gpu",
        "cpu", "memory", "partition", "queue", "batch", "interactive",
        "module", "login", "ssh", "array", "thread", "mpi", "submit",
        "compute", "resource", "allocation", "script", "bash", "core",
        "virgil", "hal", "igs", "goro", "medusa", "thanos", "hook",
        "smaug", "him", "arthas", "metallo", "karn", "ravellab", "hush",
        "mileena", "sareena", "prof-x", "shodan", "jade", "kano", "magog",
        "izzy", "toga", "toph", "error", "failed", "pending", "running",
        "walltime", "time limit", "gres", "vram", "environment",
        "password", "account", "access", "credential", "som", "network",
        "vpn", "connect", "connection", "permission", "profile", "bash_profile",
        "command not found", "jira", "ticket", "confluence", "how do i",
        "what is", "how to", "can i", "help", "setup", "set up", "install",
        "run", "execute", "submit", "check", "monitor", "cancel", "kill",
        "efficiency", "output", "log", "debug", "troubleshoot", "issue", "problem"
    ]
    q = query.lower()
    return any(kw in q for kw in keywords)
```

### `pipelines/igs_rag_pipeline.py` — updated `on_startup`

Replace lines 43–74 with:

```python
async def on_startup(self):
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    vllm_host = os.environ.get("VLLM_HOST", "")
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=OLLAMA_BASE_URL)

    if vllm_host:
        from llama_index.llms.openai_like import OpenAILike
        llm = OpenAILike(
            model=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
            api_base=vllm_host,
            api_key="not-required",
            is_chat_model=True,
            request_timeout=120.0,
        )
    else:
        from llama_index.llms.ollama import Ollama
        llm = Ollama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL, request_timeout=120.0)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection("slurm")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    with open(PROMPT_PATH, "r") as f:
        system_prompt = f.read().strip()

    self.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=llm,
        memory=ChatMemoryBuffer.from_defaults(token_limit=3000),
        system_prompt=system_prompt,
        similarity_top_k=5,
        verbose=False,
    )
```

### `pipelines/requirements.txt` — when switching to vLLM

```
# Current (Ollama):
chromadb==1.5.7
llama-index-core==0.14.20
llama-index-llms-ollama==0.10.1
llama-index-embeddings-ollama==0.9.0
llama-index-vector-stores-chroma==0.5.5

# After vLLM switch (add, keep the rest):
llama-index-llms-openai==0.4.x   # provides OpenAILike — check latest compatible
# llama-index-llms-ollama can be removed if Ollama fallback is dropped
# keep llama-index-embeddings-ollama — embeddings still use Ollama
```

---

## 6. GPU Server Setup

### Hardware target

- Node: IGS GPU server (A100 80GB × 8, allocating 1–2 for this project)
- OS: Linux (Ubuntu 22.04 or RHEL 8/9)
- CUDA: 12.x
- Driver: 525+ (for A100)

### vLLM installation

```bash
# On the GPU server, in a dedicated venv or conda env
pip install vllm

# Verify GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Model download (do this before starting vLLM)

```bash
# Download model to server-local cache to avoid re-download on container restart
huggingface-cli download Qwen/Qwen2.5-32B-Instruct \
  --local-dir /data/models/Qwen2.5-32B-Instruct \
  --local-dir-use-symlinks False
```

For gated models (DeepSeek, some Mistral): set `HUGGING_FACE_HUB_TOKEN` before downloading.

### Startup command (bare metal, for testing)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen2.5-32B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --port 8000 \
  --host 0.0.0.0
```

`--tensor-parallel-size 1` uses a single A100. Set to `2` if using two GPUs.  
`--max-model-len 8192` limits context window to avoid OOM; increase to 16384 if VRAM allows.  
`--gpu-memory-utilization 0.90` reserves 10% VRAM headroom for KV cache spikes.

### Verify it's working

```bash
curl http://localhost:8000/v1/models
# Should return the model name in JSON

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-32B-Instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

### How it plugs into Docker Compose

The GPU server runs a Docker Compose stack that is a superset of the current one:

```
docker-compose.yml         ← base (same as current, no Ollama or vLLM hardcoded)
docker-compose.gpu.yml     ← GPU server overlay: adds vLLM service, overrides env vars
```

```bash
# On GPU server:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# On local Mac (no changes):
docker compose up -d
```

`docker-compose.gpu.yml`:

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: slurm_ai_vllm
    runtime: nvidia
    ports:
      - "8000:8000"
    volumes:
      - /data/models:/root/.cache/huggingface
    command: >
      --model Qwen/Qwen2.5-32B-Instruct
      --dtype bfloat16
      --max-model-len 8192
      --gpu-memory-utilization 0.90
      --tensor-parallel-size 1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  pipelines:
    environment:
      - VLLM_HOST=http://vllm:8000/v1
      - LLM_MODEL=Qwen/Qwen2.5-32B-Instruct
    depends_on:
      - vllm
```

This override adds the vLLM service and injects `VLLM_HOST` into the Pipelines container, triggering the `OpenAILike` branch in the updated code. No changes to the base `docker-compose.yml`.

### Model selection at launch time

One of the advantages of this architecture is that the model is set by environment variable, not hardcoded. To evaluate GLM-4, DeepSeek-R1, or Mistral-Large on the GPU server:

```bash
# docker-compose.gpu.yml — change two values:
command: --model THUDM/glm-4-9b-chat ...
environment:
  - LLM_MODEL=THUDM/glm-4-9b-chat
```

The LlamaIndex code requires no changes.

---

## Appendix: Package Notes

| Package | Local dev | GPU server |
|---|---|---|
| `llama-index-llms-ollama` | Required | Optional (keep for fallback) |
| `llama-index-llms-openai` | Optional (add when ready) | Required |
| `llama-index-embeddings-ollama` | Required | Required (embeddings stay on Ollama) |
| `vllm` | Do NOT install | Install on GPU node only |

`vllm` must never be installed in the Pipelines container or the app container. It is a server-side dependency only.
