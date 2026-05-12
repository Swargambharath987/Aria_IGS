"""
Slurm command executor for the MCP server.

Runs Slurm CLI commands as local subprocesses — Aria is deployed on the
cluster GPU node so squeue, sacct, sinfo, seff, sbatch, scancel are
available directly in PATH. No SSH, no service account required.
"""
import os
import shutil
import subprocess

SLURM_TIMEOUT = int(os.environ.get("SLURM_TIMEOUT", "30"))
_SLURM_FOUND  = shutil.which("squeue") is not None


def run(username: str, command: str) -> str:
    """
    Run a Slurm command locally. username is used for logging/context only.
    Returns stdout on success, or a descriptive error string — never raises.
    """
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
            return f"[Slurm error: {err or 'non-zero exit'}]"
        return out if out else (err if err else "(no output)")
    except subprocess.TimeoutExpired:
        return f"[Slurm command timed out after {SLURM_TIMEOUT}s]"
    except Exception as e:
        return f"[Command failed: {type(e).__name__}: {e}]"
