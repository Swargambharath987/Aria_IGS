"""
agent_tools.py — FunctionTool wrappers for the Aria ReActAgent.

Each tool has a precise docstring the LLM reads to decide when to call it.
Tools are bound to a username at construction time (closure pattern) so the
agent never needs to ask the user for credentials.

source_sink is a list shared with the calling endpoint — tool functions append
source metadata to it as they run so the endpoint can persist it to sources_used.
"""
import json
import uuid
from datetime import datetime, timedelta
from typing import Callable, List

from llama_index.core.tools import FunctionTool

from tools.slurm_ssh import (
    SSH_AVAILABLE,
    cluster_status,
    job_efficiency,
    job_history,
    job_status,
)
from tools.file_reader import list_job_files, read_file

# ── Pending actions store ─────────────────────────────────────────────────────
# Keyed by action_id (str UUID). Each value is a dict:
#   { type, details, command, username, created_at }
# Entries older than 10 minutes are expired on lookup/insert.

_pending_actions: dict[str, dict] = {}

_PENDING_ACTION_TTL = timedelta(minutes=10)


def _expire_pending_actions() -> None:
    """Remove entries older than 10 minutes."""
    cutoff = datetime.utcnow() - _PENDING_ACTION_TTL
    expired = [aid for aid, entry in _pending_actions.items() if entry["created_at"] < cutoff]
    for aid in expired:
        _pending_actions.pop(aid, None)


def build_tools(
    username: str,
    retriever,
    source_sink: list,
    source_label_fn: Callable,
) -> List[FunctionTool]:
    """
    Build all FunctionTools for one agent invocation.

    username        — LDAP uid; SSH calls run as this user
    retriever       — pre-built QueryFusionRetriever (BM25 + vector)
    source_sink     — list owned by the calling endpoint; tools append source
                      metadata here so the endpoint can persist it to the DB
    source_label_fn — _source_label() from main.py
    """

    # ── Knowledge base ───────────────────────────────────────────────────────

    def search_knowledge_base(query: str) -> str:
        """
        Search the IGS Slurm and research computing knowledge base.
        Contains Slurm documentation, how-to guides, command references,
        IGS-specific policies, and bioinformatics tool guides.

        Use this when the user:
        - Asks HOW to do something (submit a job, set memory, use a partition)
        - Wants a command example or template
        - Asks what a Slurm option or flag means
        - Has an error message they want explained
        - Asks about bioinformatics tools (GATK, samtools, etc.)

        Do NOT use this for live cluster state. For current job queue,
        cluster load, or specific job results, use the cluster/job tools.
        """
        nodes = retriever.retrieve(query)
        if not nodes:
            return "No relevant documentation found in the knowledge base for that query."

        chunks = []
        for node in nodes:
            meta = node.node.metadata or {}
            label = source_label_fn(
                meta.get("collection"),
                meta.get("priority"),
                meta.get("source_url"),
            )
            source_sink.append({
                "type":       "rag",
                "label":      label,
                "score":      round(node.score, 4) if node.score is not None else None,
                "chunk_text": node.node.text[:300],
                "collection": meta.get("collection"),
            })
            chunks.append(f"[{label}]\n{node.node.text}")

        return "\n\n---\n\n".join(chunks)

    # ── Live Slurm tools ─────────────────────────────────────────────────────

    def get_job_status() -> str:
        """
        Get the current live job queue for this user from the IGS cluster.

        Use when the user asks:
        - What jobs are running, pending, or queued
        - "Show my jobs", "check my queue", "what is running"
        - Whether a specific job is still running (without a job ID)

        Returns live SSH data, not documentation. No arguments needed —
        this always queries for the current authenticated user.
        """
        if not SSH_AVAILABLE:
            return "[SSH not configured — live cluster data unavailable. Set SLURM_SSH_HOST to enable.]"
        source_sink.append({"type": "tool", "label": "Live Job Queue", "tool": "get_job_status"})
        return job_status(username)

    def get_job_history(days: int = 7) -> str:
        """
        Get completed job history for this user from the IGS cluster.

        Use when the user asks about past, finished, or completed jobs,
        or asks to review what jobs ran recently. Defaults to the last
        7 days; pass a different number of days if the user specifies.
        """
        if not SSH_AVAILABLE:
            return "[SSH not configured — live cluster data unavailable. Set SLURM_SSH_HOST to enable.]"
        source_sink.append({"type": "tool", "label": f"Job History (last {days}d)", "tool": "get_job_history"})
        return job_history(username, days)

    def get_job_efficiency(job_id: str) -> str:
        """
        Get CPU and memory efficiency report for a specific Slurm job.

        Use when:
        - The user provides a numeric job ID and wants efficiency/resource info
        - The user asks why a job failed or used too much / too little memory
        - The user wants to know CPU utilization for a specific job

        Requires: job_id — the numeric Slurm job ID extracted from the message.
        """
        if not SSH_AVAILABLE:
            return "[SSH not configured — live cluster data unavailable. Set SLURM_SSH_HOST to enable.]"
        source_sink.append({"type": "tool", "label": f"Job Efficiency (job {job_id})", "tool": "get_job_efficiency"})
        return job_efficiency(username, job_id)

    def get_cluster_status() -> str:
        """
        Get live cluster-wide status: partitions, available nodes, and load.

        Use when the user asks:
        - How busy the cluster is right now
        - Whether nodes are available in a specific partition
        - The current state of the grid / HPC cluster
        - Which partitions exist and what they support
        """
        if not SSH_AVAILABLE:
            return "[SSH not configured — live cluster data unavailable. Set SLURM_SSH_HOST to enable.]"
        source_sink.append({"type": "tool", "label": "Live Cluster Status", "tool": "get_cluster_status"})
        return cluster_status()

    # ── File tools ───────────────────────────────────────────────────────────

    def read_user_file(path: str) -> str:
        """
        Read a job script, log, or output file from the user's home directory
        on the IGS cluster login node.

        Use when the user shares a file path (e.g. ~/jobs/submit.sh,
        ~/logs/job_12345.out) and asks you to read, analyze, or debug it.

        Allowed file types: .sh .slurm .sbatch .out .err .log .py .R .txt .yaml .yml
        Only files within the user's home directory are accessible.
        """
        if not SSH_AVAILABLE:
            return "[SSH not configured — file reading unavailable. Set SLURM_SSH_HOST to enable.]"
        source_sink.append({"type": "tool", "label": f"File: {path}", "tool": "read_user_file"})
        return read_file(username, path)

    def list_user_files(directory: str) -> str:
        """
        List job scripts and output files in a directory within the user's
        home directory on the IGS cluster.

        Use when the user asks to see what files are in a directory,
        e.g. "list my scripts in ~/jobs" or "what .sh files are in ~/slurm".
        """
        if not SSH_AVAILABLE:
            return "[SSH not configured — file listing unavailable. Set SLURM_SSH_HOST to enable.]"
        source_sink.append({"type": "tool", "label": f"Directory: {directory}", "tool": "list_user_files"})
        return list_job_files(username, directory)

    # ── Write tools (require approval) ──────────────────────────────────────

    def request_job_submission(
        script_content: str,
        job_name: str = "my_job",
        partition: str = "shared",
        mem: str = "4G",
        cpus: int = 1,
    ) -> str:
        """
        Use this when the user explicitly asks to SUBMIT a job script to the
        Slurm cluster via sbatch.

        Do NOT use this for questions about how to submit a job, resource
        estimates, or any read-only operation. Only use it when the user
        clearly says they want to run / submit / execute a job script right now.

        Parameters:
            script_content — the full text of the job script to submit
            job_name       — job name (default: my_job)
            partition      — Slurm partition to use (default: shared)
            mem            — memory request, e.g. 4G, 16G (default: 4G)
            cpus           — number of CPUs to request (default: 1)

        Returns a pending-action marker that the UI will display as an
        Approve / Deny card. The job is NOT submitted until the user approves.
        """
        if not SSH_AVAILABLE:
            return "[SSH not configured — job submission unavailable. Set SLURM_SSH_HOST to enable.]"

        _expire_pending_actions()

        action_id = str(uuid.uuid4())
        # Escape single quotes in script content for safe shell injection
        safe_script = script_content.replace("'", "'\\''")
        command = (
            f"echo '{safe_script}' | sbatch "
            f"--job-name={job_name} "
            f"--partition={partition} "
            f"--mem={mem} "
            f"--cpus-per-task={cpus}"
        )
        summary = (
            f"Submit job: {job_name}, partition: {partition}, "
            f"mem: {mem}, cpus: {cpus}"
        )
        _pending_actions[action_id] = {
            "type":       "sbatch",
            "details":    {"job_name": job_name, "partition": partition, "mem": mem, "cpus": cpus},
            "command":    command,
            "username":   username,
            "created_at": datetime.utcnow(),
        }
        source_sink.append({"type": "tool", "label": "Job Submission Request", "tool": "request_job_submission"})
        marker = json.dumps({"action_id": action_id, "type": "sbatch", "summary": summary})
        return (
            f"I've prepared your job submission. Please review the details and approve or deny below.\n\n"
            f"**Job:** {job_name}  |  **Partition:** {partition}  |  **Memory:** {mem}  |  **CPUs:** {cpus}\n\n"
            f"ARIA_PENDING_ACTION:{marker}"
        )

    def request_job_cancellation(job_id: str) -> str:
        """
        Use this when the user explicitly asks to CANCEL a specific job by its
        numeric Slurm job ID (e.g. "cancel job 12345", "scancel 12345").

        Do NOT use this for checking job status, listing jobs, or any read
        operation. Only use it when the user clearly wants to cancel / kill /
        stop a specific running or pending job right now.

        Parameters:
            job_id — the numeric Slurm job ID to cancel

        Returns a pending-action marker that the UI will display as an
        Approve / Deny card. The job is NOT cancelled until the user approves.
        """
        if not SSH_AVAILABLE:
            return "[SSH not configured — job cancellation unavailable. Set SLURM_SSH_HOST to enable.]"

        _expire_pending_actions()

        action_id = str(uuid.uuid4())
        command = f"scancel --user={username} {job_id}"
        summary = f"Cancel job ID {job_id}"
        _pending_actions[action_id] = {
            "type":       "scancel",
            "details":    {"job_id": job_id},
            "command":    command,
            "username":   username,
            "created_at": datetime.utcnow(),
        }
        source_sink.append({"type": "tool", "label": f"Job Cancellation Request (job {job_id})", "tool": "request_job_cancellation"})
        marker = json.dumps({"action_id": action_id, "type": "scancel", "summary": summary})
        return (
            f"I've prepared the cancellation request for job **{job_id}**. "
            f"Please approve or deny below.\n\n"
            f"ARIA_PENDING_ACTION:{marker}"
        )

    # ── Resource estimator ───────────────────────────────────────────────────

    def estimate_job_resources(script_content: str) -> str:
        """
        Analyze a job script and recommend Slurm resource allocations.

        Use this when the user:
        - Pastes or shares a job script and asks how much memory/CPU to request
        - Asks "what sbatch flags should I use for this script?"
        - Wants to know which partition to use for their workload
        - Asks for help writing the #SBATCH header for their job
        - Asks how many CPUs or how much memory a specific bioinformatics tool needs

        Input: the full text content of the script (bash, Python, R, or workflow file).
        Returns: recommended resources + a ready-to-paste #SBATCH header block.
        """
        from tools.resource_estimator import estimate
        result = estimate(script_content)
        source_sink.append({"type": "tool", "label": "Resource Estimator", "tool": "estimate_job_resources"})
        return result

    return [
        FunctionTool.from_defaults(fn=search_knowledge_base),
        FunctionTool.from_defaults(fn=get_job_status),
        FunctionTool.from_defaults(fn=get_job_history),
        FunctionTool.from_defaults(fn=get_job_efficiency),
        FunctionTool.from_defaults(fn=get_cluster_status),
        FunctionTool.from_defaults(fn=read_user_file),
        FunctionTool.from_defaults(fn=list_user_files),
        FunctionTool.from_defaults(fn=estimate_job_resources),
        FunctionTool.from_defaults(fn=request_job_submission),
        FunctionTool.from_defaults(fn=request_job_cancellation),
    ]
