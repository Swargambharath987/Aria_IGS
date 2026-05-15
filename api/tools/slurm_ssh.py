"""
Slurm command executor — runs Slurm CLI commands as local subprocesses.

Aria is deployed on the cluster GPU node, so Slurm commands (squeue, sacct,
sinfo, seff, sbatch, scancel) are available directly in the environment.
No SSH, no service account, no keys required.

If SLURM_SSH_HOST is set (legacy env var), it is ignored — subprocess is
always used. Set SLURM_ENABLED=false to disable all Slurm tools.
"""
import os
import re
import shutil
import subprocess

SLURM_ENABLED = os.environ.get("SLURM_ENABLED", "true").lower() != "false"
SLURM_TIMEOUT = int(os.environ.get("SLURM_TIMEOUT", "30"))

# Detect whether Slurm is actually installed on this node at startup.
_SLURM_FOUND = shutil.which("squeue") is not None

# Exported alias used by agent_tools.py — true when both the env var allows it
# and the Slurm binaries are actually present.
SSH_AVAILABLE = SLURM_ENABLED and _SLURM_FOUND


def _run(username: str, command: str) -> str:
    """
    Run a Slurm command locally via subprocess.
    username is used for filtering/logging; the command runs as the Aria process user.
    Returns stdout on success, or a descriptive error string — never raises.
    """
    if not SLURM_ENABLED:
        return "[Slurm tools disabled — set SLURM_ENABLED=true to enable]"
    if not _SLURM_FOUND:
        return "[Slurm commands not found on this node — is squeue in PATH?]"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=SLURM_TIMEOUT,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if result.returncode != 0 and not out:
            return f"[Slurm error: {err or 'command returned non-zero exit'}]"
        return out if out else (err if err else "(command returned no output)")
    except subprocess.TimeoutExpired:
        return f"[Slurm command timed out after {SLURM_TIMEOUT}s]"
    except Exception as e:
        return f"[Slurm command failed: {type(e).__name__}: {e}]"


# ── Intent detection ───────────────────────────────────────────────────────
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
_HOW_TO_PREFIX = re.compile(r"^\s*(how (do i|can i|to)|what is|what does|what are)\b", re.IGNORECASE)


def needs_live_data(message: str) -> str | None:
    if not SLURM_ENABLED or not _SLURM_FOUND:
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


# ── Tool functions ─────────────────────────────────────────────────────────

def job_status(username: str) -> str:
    out = _run(
        username,
        f"squeue -u {username} --format='%.10i %.12P %.30j %.8T %.10M %.9l %.4D %R'",
    )
    return f"Live job queue for {username}:\n{out}"


def job_history(username: str, days: int = 7) -> str:
    out = _run(
        username,
        f"sacct -u {username} --format=JobID,JobName%30,State,Elapsed,MaxRSS,AllocCPUS "
        f"--starttime=now-{days}days --endtime=now",
    )
    return f"Job history for {username} (last {days} days):\n{out}"


def job_efficiency(username: str, job_id: str) -> str:
    if not str(job_id).isdigit():
        return f"[Invalid job ID '{job_id}' — must be a numeric Slurm job ID]"
    out = _run(username, f"seff {job_id}")
    return f"Efficiency report for job {job_id}:\n{out}"


def cluster_status() -> str:
    out = _run("", "sinfo --format='%.15P %.5a %.10l %.6D %.6t %N'")
    return f"Current cluster status:\n{out}"


def get_live_data(tool: str, username: str, message: str) -> str:
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
            return "[Could not extract a job ID — please include the numeric job ID]"
        return job_efficiency(username, job_id)
    return ""
