"""
MCP Tools — all Slurm operations exposed to the LLM.

Auth flow per request:
  1. Bearer token extracted from MCP request context
  2. JWT validated against SLURM_JWT_KEY (rejects expired/invalid tokens)
  3. Username extracted from 'sun' JWT claim
  4. Token + username forwarded to Slurm REST API as X-SLURM-USER-* headers
  5. Slurm enforces its own per-user permissions — the MCP server is a passthrough
"""
import asyncio
from typing import Optional

from fastmcp import FastMCP, Context

from auth import validate_token
from client import SlurmClient
from config import settings
from models import JobListFilters, JobSubmissionRequest
from utils import format_job_details, format_job_list


def _auth(ctx: Context) -> tuple[str, str]:
    """Extract and validate the Bearer token from the MCP context. Returns (username, token)."""
    token = ""
    auth_header = (ctx.request_context.request.headers or {}).get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()

    if not token:
        raise ValueError("No Bearer token provided. Generate one with: scontrol token lifespan=3600")

    username = validate_token(token, settings.slurm_jwt_key)
    if not username:
        raise ValueError("Invalid or expired Slurm JWT. Generate a new one with: scontrol token lifespan=3600")

    return username, token


def register_tools(app: FastMCP):

    @app.tool(tags=["connectivity", "auth"])
    async def ping(ctx: Context) -> dict:
        """
        Test connectivity to the Slurm REST API and validate your JWT token.
        Run this first to confirm everything is configured correctly.
        """
        username, token = _auth(ctx)
        client = SlurmClient(username, token)
        data   = await client.get("/ping")
        return {
            "success":     True,
            "user":        username,
            "cluster_url": settings.slurm_api_url,
            "api_version": settings.slurm_api_version,
            "pings":       data.get("pings", []),
        }

    @app.tool(tags=["job", "list", "queue"])
    async def list_cluster_jobs(ctx: Context, filters: Optional[JobListFilters] = None) -> str:
        """
        List jobs in the Slurm queue. Optionally filter by user, state, or partition.

        Args:
            filters: Optional filters — user, state (RUNNING/PENDING/FAILED/etc.),
                     partition, limit (default 20), offset (default 0)
        """
        username, token = _auth(ctx)
        client  = SlurmClient(username, token)
        params: dict = {}
        if filters:
            if filters.user:      params["user"]      = filters.user
            if filters.partition: params["partition"]  = filters.partition
            if filters.state:     params["state"]      = ",".join(filters.state)
        data = await client.get("/jobs", params=params)
        jobs = data.get("jobs", [])
        f    = filters or JobListFilters()
        if f.offset:
            jobs = jobs[f.offset:]
        jobs = jobs[:f.limit or 20]
        return format_job_list(jobs)

    @app.tool(tags=["job", "detail"])
    async def get_job_details(ctx: Context, job_id: str) -> str:
        """
        Get full details for a specific Slurm job by ID.

        Args:
            job_id: The numeric Slurm job ID (e.g. "1234567")
        """
        username, token = _auth(ctx)
        client = SlurmClient(username, token)
        data   = await client.get(f"/job/{job_id}")
        jobs   = data.get("jobs", [])
        if not jobs:
            return f"Job {job_id} not found."
        return format_job_details(jobs[0])

    @app.tool(tags=["job", "submit"])
    async def submit_slurm_job(ctx: Context, request: JobSubmissionRequest) -> dict:
        """
        Submit a new job to the Slurm cluster (equivalent to sbatch).

        Args:
            request: JobSubmissionRequest with a JobSpec (resources/partition/etc.)
                     and a script (full bash script content as a string)

        Returns job_id on success. Errors and warnings from Slurm are included in the response.
        """
        username, token = _auth(ctx)
        client = SlurmClient(username, token)
        body   = {
            "job":    request.job.model_dump(exclude_none=True),
            "script": request.script,
        }
        data = await client.post("/job/submit", body)
        return {
            "success":  not data.get("errors"),
            "job_id":   data.get("job_id"),
            "message":  f"Job submitted with ID {data.get('job_id')}" if data.get("job_id") else "Submission failed",
            "errors":   data.get("errors",   []),
            "warnings": data.get("warnings", []),
        }

    @app.tool(tags=["job", "cancel"])
    async def cancel_slurm_job(ctx: Context, job_id: str) -> dict:
        """
        Cancel a Slurm job by ID (equivalent to scancel).
        You can only cancel your own jobs unless you have admin privileges.

        Args:
            job_id: The numeric Slurm job ID to cancel
        """
        username, token = _auth(ctx)
        client = SlurmClient(username, token)
        data   = await client.delete(f"/job/{job_id}")
        return {
            "success":  not data.get("errors"),
            "job_id":   job_id,
            "message":  f"Job {job_id} cancelled" if not data.get("errors") else "Cancel failed",
            "errors":   data.get("errors",   []),
            "warnings": data.get("warnings", []),
        }

    @app.tool(tags=["cluster", "summary"])
    async def get_cluster_summary(ctx: Context, partition: Optional[str] = None) -> str:
        """
        Get a full cluster overview: job counts by state, node states, and partition load.
        Optionally filter to a specific partition.

        Args:
            partition: Optional partition name to focus on (e.g. "gpu", "highmem")
        """
        username, token = _auth(ctx)
        client = SlurmClient(username, token)

        jobs_data, parts_data, nodes_data = await asyncio.gather(
            client.get("/jobs"),
            client.get("/partitions"),
            client.get("/nodes"),
        )

        jobs  = jobs_data.get("jobs",       [])
        parts = parts_data.get("partitions", [])
        nodes = nodes_data.get("nodes",      [])

        if partition:
            jobs  = [j for j in jobs  if j.get("partition") == partition]
            parts = [p for p in parts if p.get("name")      == partition]
            nodes = [n for n in nodes if partition in n.get("partitions", [])]

        # Job state summary
        state_counts: dict[str, int] = {}
        for j in jobs:
            s = j.get("job_state", "UNKNOWN")
            state_counts[s] = state_counts.get(s, 0) + 1

        # Node state summary
        node_counts: dict[str, int] = {}
        for n in nodes:
            states = n.get("state", ["UNKNOWN"])
            s = states[0] if states else "UNKNOWN"
            node_counts[s] = node_counts.get(s, 0) + 1

        lines = [
            f"## IGS Cluster Summary{f' — {partition}' if partition else ''}",
            "",
            "### Jobs",
            *[f"- {state}: {count}" for state, count in sorted(state_counts.items())],
            "",
            "### Nodes",
            *[f"- {state}: {count}" for state, count in sorted(node_counts.items())],
            "",
            "### Partitions",
            "| Partition | State | Total nodes |",
            "|-----------|-------|-------------|",
            *[
                f"| {p.get('name','')} | {p.get('state','')} | {p.get('nodes',{}).get('total','')} |"
                for p in parts
            ],
        ]
        return "\n".join(lines)
