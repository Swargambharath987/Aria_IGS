"""
title: IGS Aria
author: IGS
description: AI Research Intelligence Assistant — forwards queries to the Aria backend. No ML libraries loaded here.
"""

import os
from typing import List, Union, Generator, Iterator

import httpx

API_URL       = os.environ.get("AGENT_API_URL",   "http://api:8000")
API_TOKEN     = os.environ.get("AGENT_API_TOKEN",  "igs-dev-token")

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
        pass

    async def on_startup(self):
        # Docker's healthcheck + condition: service_healthy guarantees the API
        # is fully ready before this container starts. One check is enough.
        r = httpx.get(f"{API_URL}/health", timeout=10)
        r.raise_for_status()

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
            return "Hi! I'm Aria — the IGS AI Research Intelligence Assistant. Ask me anything about the IGS computational grid, Slurm, coding, or research computing."

        if not any(kw in user_message.lower() for kw in SLURM_KEYWORDS):
            return "I can only help with IGS grid and Slurm-related questions. For other topics, please refer to the appropriate resource."

        # Use Open WebUI's chat_id as session_id so conversation memory persists
        session_id = body.get("chat_id")

        try:
            r = httpx.post(
                f"{API_URL}/chat",
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                json={"message": user_message, "session_id": session_id},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["response"]
        except Exception as e:
            return f"Error reaching the IGS Agent API: {e}"
