"""
title: IGS Grid RAG Pipeline
author: IGS
description: ReAct agent that reasons over IGS SLURM docs. No external calls.
"""

import os
from typing import List, Union, Generator, Iterator
from pathlib import Path


CHROMA_DIR = Path("/app/data/chroma_db")
PROMPT_PATH = Path("/app/prompts/system_prompt.txt")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4")

GREETINGS = {"hi", "hello", "hey", "howdy", "greetings", "good morning", "good afternoon"}

SLURM_KEYWORDS = [
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
    "efficiency", "output", "log", "debug", "troubleshoot", "issue", "problem",
]


class Pipeline:
    def __init__(self):
        self.agent = None

    async def on_startup(self):
        # NOTE: If this pipeline crashes on startup, the Pipelines server moves the .py file to
        # /app/pipelines/failed/. To recover:
        #   mv pipelines/failed/igs_rag_pipeline.py pipelines/igs_rag_pipeline.py
        # then restart the Pipelines server.
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.core.agent import ReActAgent
        from llama_index.core.tools import QueryEngineTool, ToolMetadata
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb

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

        with open(PROMPT_PATH, "r") as f:
            system_prompt = f.read().strip()

        self.agent = ReActAgent.from_tools(
            [slurm_tool],
            llm=llm,
            system_prompt=system_prompt,
            max_iterations=5,
            verbose=False,
        )

    async def on_shutdown(self):
        pass

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:

        if user_message.strip().lower().rstrip("!,.") in GREETINGS:
            return "Hi! I'm the IGS Grid Assistant. Ask me anything about using the IGS computational grid or Slurm — job submission, GPU resources, troubleshooting, and more."

        if not any(kw in user_message.lower() for kw in SLURM_KEYWORDS):
            return "I can only help with IGS grid and Slurm-related questions. For other topics, please refer to the appropriate resource."

        response = self.agent.chat(user_message)
        return str(response)
