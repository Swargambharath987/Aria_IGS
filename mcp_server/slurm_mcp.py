"""
Slurm IGS MCP Server
====================
Exposes IGS cluster operations as MCP tools so any MCP-compatible client
(Claude Desktop, Cursor, other agents) can query the cluster directly.

All tools are read-only. Write operations (sbatch, scancel) are not
implemented until Mike and Dustin sign off.

Transport:
  SSE  (Docker service, port 8001) — for HTTP clients and remote access
  stdio (local)                    — for Claude Desktop direct integration
"""

import os
import sys

from mcp.server.fastmcp import FastMCP

# When running in Docker the tools/ directory is at /app/tools/
# When running locally (stdio) point at the api/ source tree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..","api"))

from tools.slurm_ssh import (
    SSH_AVAILABLE,
    cluster_status,
    job_efficiency,
    job_history,
    job_status,
)
from tools.file_reader import list_job_files, read_file

mcp = FastMCP(
    name="slurm-igs",
    instructions=(
        "Tools for the IGS HPC cluster at University of Maryland. "
        "Provides live Slurm job and cluster data. All operations are read-only."
    ),
)


# ── Job tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def get_job_status(username: str) -> str:
    """
    Get the current live job queue for a user from the IGS Slurm cluster.
    Shows running, pending, and recently completed jobs.

    Args:
        username: The LDAP username of the researcher (e.g. jsmith1)
    """
    if not SSH_AVAILABLE:
        return "[SSH not configured — set SLURM_SSH_HOST environment variable to enable]"
    return job_status(username)


@mcp.tool()
def get_job_history(username: str, days: int = 7) -> str:
    """
    Get completed job history for a user from the IGS cluster accounting database.

    Args:
        username: The LDAP username of the researcher
        days:     How many days back to search (default: 7)
    """
    if not SSH_AVAILABLE:
        return "[SSH not configured — set SLURM_SSH_HOST environment variable to enable]"
    return job_history(username, days)


@mcp.tool()
def get_job_efficiency(username: str, job_id: str) -> str:
    """
    Get CPU and memory efficiency for a specific completed Slurm job.
    Shows how well the job used its requested resources.

    Args:
        username: The LDAP username who owns the job
        job_id:   The numeric Slurm job ID (e.g. 1234567)
    """
    if not SSH_AVAILABLE:
        return "[SSH not configured — set SLURM_SSH_HOST environment variable to enable]"
    return job_efficiency(username, job_id)


@mcp.tool()
def get_cluster_status() -> str:
    """
    Get live cluster-wide status: partitions, available nodes, current load.
    Does not require a username — queries cluster-level info only.
    """
    if not SSH_AVAILABLE:
        return "[SSH not configured — set SLURM_SSH_HOST environment variable to enable]"
    return cluster_status()


# ── File tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def read_user_file(username: str, path: str) -> str:
    """
    Read a job script, log, or output file from a user's home directory
    on the IGS cluster login node. Path must be within the user's home dir.

    Supported extensions: .sh .slurm .sbatch .out .err .log .py .R .txt .yaml .yml

    Args:
        username: The LDAP username who owns the file
        path:     Full path or ~/relative path (e.g. ~/jobs/align.sh)
    """
    if not SSH_AVAILABLE:
        return "[SSH not configured — set SLURM_SSH_HOST environment variable to enable]"
    return read_file(username, path)


@mcp.tool()
def list_user_files(username: str, directory: str) -> str:
    """
    List job scripts and output files in a directory within a user's
    home directory on the IGS cluster.

    Args:
        username:  The LDAP username
        directory: Directory to list (e.g. ~/jobs or ~/slurm_logs)
    """
    if not SSH_AVAILABLE:
        return "[SSH not configured — set SLURM_SSH_HOST environment variable to enable]"
    return list_job_files(username, directory)
