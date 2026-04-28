import os
import chromadb
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
PROMPT_PATH = BASE_DIR / "prompts" / "system_prompt.txt"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4")


def load_system_prompt():
    with open(PROMPT_PATH, "r") as f:
        return f.read().strip()


def build_agent() -> ReActAgent:
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=120.0)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection("slurm")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

    slurm_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="slurm_knowledge_base",
            description=(
                "Search the IGS SLURM Grid documentation, troubleshooting guides, and official "
                "Slurm 23.11.6 docs. Use this for questions about job submission, GPU/CPU resource "
                "allocation, cluster commands (sbatch, squeue, sacct, seff, sinfo, scancel), "
                "error messages, IGS-specific node names (virgil, hal, goro, medusa, thanos, "
                "smaug, arthas, metallo, karn, etc.), partitions, walltime limits, environment "
                "setup, VPN/SSH access, and general IGS grid usage."
            ),
        ),
    )

    system_prompt = load_system_prompt()

    agent = ReActAgent.from_tools(
        [slurm_tool],
        llm=llm,
        system_prompt=system_prompt,
        max_iterations=5,
        verbose=False,
    )
    return agent


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
