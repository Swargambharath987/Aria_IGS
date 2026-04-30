"""
Slurm SSH tools — Phase 1.1 (read-only)
Executes Slurm commands on the IGS login node via SSH and returns live data.

Phase 2 (sbatch, scancel) requires explicit sign-off from Mike and Dustin before
any write commands are added here.

All functions return a plain string — callers inject it into the LLM context.
If SSH is not configured or fails, functions return a descriptive error string
so the LLM can tell the user what happened instead of crashing.
"""
import os
import re

import paramiko

# ── SSH config (all from environment — never hardcoded) ────────────────────
SLURM_SSH_HOST         = os.environ.get("SLURM_SSH_HOST", "")
SLURM_SSH_KEY_PATH     = os.environ.get("SLURM_SSH_KEY_PATH", "")
SLURM_SSH_DEFAULT_USER = os.environ.get("SLURM_SSH_DEFAULT_USER", "")
SLURM_SSH_TIMEOUT      = int(os.environ.get("SLURM_SSH_TIMEOUT", "15"))

# SSH tools are silently disabled when SLURM_SSH_HOST is not set.
# The rest of the system works normally without them.
SSH_AVAILABLE = bool(SLURM_SSH_HOST)


# ── Intent detection ───────────────────────────────────────────────────────
# Only match phrases that clearly ask for LIVE state.
# How-to questions ("how do I check my jobs") go to the knowledge base.

_INTENT_JOB_STATUS = re.compile(
    r"\b(my jobs?|jobs? (running|pending|queued|in (the )?queue)|"
    r"what'?s? (running|in (my )?queue)|check (my )?queue|show (my )?jobs?|squeue)\b",
    re.IGNORECASE,
)
_INTENT_JOB_HISTORY = re.compile(
    r"\b(my (job )?history|completed jobs?|finished jobs?|past jobs?|sacct)\b",
    re.IGNORECASE,
)
_INTENT_JOB_EFFICIENCY = re.compile(
    r"\b(job efficiency|seff|how efficient|memory (used|usage)|cpu (used|usage))\b",
    re.IGNORECASE,
)
_INTENT_CLUSTER_STATUS = re.compile(
    r"\b(cluster (status|busy|load)|available nodes?|free nodes?|node status|sinfo|how busy (is the cluster)?)\b",
    re.IGNORECASE,
)
# Phrases that look like live-data requests but are actually how-to questions
_HOW_TO_PREFIX = re.compile(r"^\s*(how (do i|can i|to)|what is|what does|what are)\b", re.IGNORECASE)


def needs_live_data(message: str) -> str | None:
    """
    Return a tool name if the message needs live cluster data, or None.
    Checked in priority order: efficiency > job status > history > cluster.
    How-to questions are explicitly excluded — they go to the knowledge base.
    """
    if not SSH_AVAILABLE:
        return None
    if _HOW_TO_PREFIX.match(message):
        return None
    if _INTENT_JOB_EFFICIENCY.search(message):
        return "job_efficiency"
    if _INTENT_JOB_STATUS.search(message):
        return "job_status"
    if _INTENT_JOB_HISTORY.search(message):
        return "job_history"
    if _INTENT_CLUSTER_STATUS.search(message):
        return "cluster_status"
    return None


# ── SSH executor ───────────────────────────────────────────────────────────

def _run(username: str, command: str) -> str:
    """
    SSH into the Slurm login node, run command, return stdout.
    On any failure returns an error string — never raises.
    """
    if not SLURM_SSH_HOST:
        return "[SSH tools not configured — SLURM_SSH_HOST not set]"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        connect_kwargs: dict = {
            "hostname": SLURM_SSH_HOST,
            "username": username,
            "timeout": SLURM_SSH_TIMEOUT,
        }
        if SLURM_SSH_KEY_PATH:
            connect_kwargs["key_filename"] = SLURM_SSH_KEY_PATH

        client.connect(**connect_kwargs)
        _, stdout, stderr = client.exec_command(command, timeout=SLURM_SSH_TIMEOUT)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        return out if out else (err if err else "(command returned no output)")
    except paramiko.AuthenticationException:
        return f"[SSH authentication failed for user {username} on {SLURM_SSH_HOST}]"
    except paramiko.SSHException as e:
        return f"[SSH error: {e}]"
    except TimeoutError:
        return f"[SSH timed out connecting to {SLURM_SSH_HOST}]"
    except Exception as e:
        return f"[SSH failed: {e}]"
    finally:
        client.close()


# ── Tool functions ─────────────────────────────────────────────────────────

def job_status(username: str) -> str:
    out = _run(
        username,
        f"module load slurm 2>/dev/null; "
        f"squeue -u {username} --format='%.10i %.12P %.30j %.8T %.10M %.9l %.4D %R'",
    )
    return f"Live job queue for {username}:\n{out}"


def job_history(username: str, days: int = 7) -> str:
    out = _run(
        username,
        f"module load slurm 2>/dev/null; "
        f"sacct -u {username} --format=JobID,JobName%30,State,Elapsed,MaxRSS,AllocCPUS "
        f"--starttime=now-{days}days --endtime=now",
    )
    return f"Job history for {username} (last {days} days):\n{out}"


def job_efficiency(username: str, job_id: str) -> str:
    out = _run(username, f"module load slurm 2>/dev/null; seff {job_id}")
    return f"Efficiency report for job {job_id}:\n{out}"


def cluster_status() -> str:
    # sinfo doesn't need a user context — use the default user or first available
    user = SLURM_SSH_DEFAULT_USER or "aria"
    out = _run(
        user,
        "module load slurm 2>/dev/null; "
        "sinfo --format='%.15P %.5a %.10l %.6D %.6t %N'",
    )
    return f"Current cluster status:\n{out}"


# ── Dispatcher ─────────────────────────────────────────────────────────────

def get_live_data(tool: str, username: str, message: str) -> str:
    """
    Run the tool indicated by needs_live_data() and return a formatted string
    ready to be injected into the LLM context prefix.
    """
    if tool == "job_status":
        return job_status(username)
    if tool == "job_history":
        return job_history(username)
    if tool == "cluster_status":
        return cluster_status()
    if tool == "job_efficiency":
        match = re.search(r"\b(\d{5,})\b", message)
        job_id = match.group(1) if match else "unknown"
        if job_id == "unknown":
            return "[Could not extract a job ID from your message. Please include the numeric job ID.]"
        return job_efficiency(username, job_id)
    return ""
