"""
MCP Tools — all Slurm operations exposed to the LLM via subprocess.

Auth flow per request:
  1. Bearer token (Aria JWT) extracted from MCP request context
  2. JWT validated against JWT_SECRET → username extracted
  3. Slurm commands run locally as subprocesses (Aria runs on the cluster node)
  4. Read commands: filtered by -u {username}
  5. Write commands (sbatch, scancel): run as the Aria deployment user
"""
import shlex
import tempfile
from typing import Optional

from fastmcp import FastMCP, Context

from auth import validate_token
from models import JobListFilters, JobSubmissionRequest
from ssh import run


# ── Helpers ──────────────────────────────────────────────────────────────────

def _auth(ctx: Context) -> str:
    """Extract and validate the Aria JWT. Returns the username or raises."""
    token = ""
    auth_header = (ctx.request_context.request.headers or {}).get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()

    if not token:
        raise ValueError("No Bearer token provided. Log in via the Aria web UI first.")

    username = validate_token(token)
    if not username:
        raise ValueError("Invalid or expired token. Please log in again via the Aria web UI.")

    return username


def _squeue_format() -> str:
    return "--format='%.10i %.12P %.30j %.8u %.8T %.10M %.9l %.4D %R'"


# ── Tool registration ─────────────────────────────────────────────────────────

def register_tools(app: FastMCP):

    @app.tool(tags=["connectivity"])
    async def ping(ctx: Context) -> dict:
        """
        Test SSH connectivity to the Slurm login node and validate your JWT token.
        Run this first to confirm the MCP server can reach the cluster.
        """
        username = _auth(ctx)
        out = run(username, "sinfo --version")
        ok = not out.startswith("[")
        return {
            "success": ok,
            "user":    username,
            "detail":  out,
        }

    @app.tool(tags=["job", "list", "queue"])
    async def list_cluster_jobs(ctx: Context, filters: Optional[JobListFilters] = None) -> str:
        """
        List jobs in the Slurm queue. Shows your own jobs by default.
        Optionally filter by user, state, or partition.

        Args:
            filters: Optional — user, state (RUNNING/PENDING/FAILED/etc.),
                     partition, limit (default 20), offset (default 0).
                     If filters.user is omitted, shows the authenticated user's jobs.
        """
        username = _auth(ctx)
        f = filters or JobListFilters()
        target_user = f.user or username

        parts = [f"squeue -u {shlex.quote(target_user)} {_squeue_format()}"]
        if f.partition:
            parts.append(f"-p {shlex.quote(f.partition)}")
        if f.state:
            parts.append(f"--states={','.join(f.state)}")

        cmd = " ".join(parts)
        out = run(username, cmd)

        lines = out.splitlines()
        if f.offset:
            lines = lines[f.offset:]
        if f.limit:
            lines = lines[: f.limit]

        return f"Jobs for {target_user}:\n" + "\n".join(lines)

    @app.tool(tags=["job", "detail"])
    async def get_job_details(ctx: Context, job_id: str) -> str:
        """
        Get full details for a specific Slurm job by ID.
        Includes state, allocated resources, runtime, node list, and exit code.

        Args:
            job_id: The numeric Slurm job ID (e.g. "1234567")
        """
        username = _auth(ctx)
        jid = shlex.quote(job_id)

        # scontrol for live/queued jobs; sacct for completed ones
        cmd = (
            f"scontrol show job {jid} 2>/dev/null || "
            f"sacct -j {jid} --format=JobID,JobName%30,State,Elapsed,MaxRSS,AllocCPUS,"
            f"ExitCode,NodeList --noheader"
        )
        out = run(username, cmd)
        if not out or out.startswith("["):
            return f"Job {job_id} not found (may have expired from Slurm history)."
        return f"Details for job {job_id}:\n{out}"

    @app.tool(tags=["job", "efficiency"])
    async def get_job_efficiency(ctx: Context, job_id: str) -> str:
        """
        Get the CPU and memory efficiency report for a completed Slurm job.
        Shows how much of the requested resources the job actually used.

        Args:
            job_id: The numeric Slurm job ID
        """
        username = _auth(ctx)
        out = run(username, f"seff {shlex.quote(job_id)}")
        return f"Efficiency report for job {job_id}:\n{out}"

    @app.tool(tags=["job", "history"])
    async def get_job_history(ctx: Context, days: int = 7, user: Optional[str] = None) -> str:
        """
        Get completed job history from sacct. Defaults to the authenticated user's
        last 7 days. Admins can pass a different user.

        Args:
            days: Number of days to look back (default 7)
            user: Username to query (defaults to the authenticated user)
        """
        username = _auth(ctx)
        target_user = user or username
        cmd = (
            f"sacct -u {shlex.quote(target_user)} "
            f"--format=JobID,JobName%30,State,Elapsed,MaxRSS,AllocCPUS,ExitCode "
            f"--starttime=now-{int(days)}days --endtime=now"
        )
        out = run(username, cmd)
        return f"Job history for {target_user} (last {days} days):\n{out}"

    @app.tool(tags=["job", "submit"])
    async def submit_slurm_job(ctx: Context, request: JobSubmissionRequest) -> dict:
        """
        Submit a new job to the Slurm cluster (equivalent to sbatch).

        Requires aria_service to have Slurm operator privileges — contact the cluster
        admin to grant this. The job will appear under the authenticated user's account.

        Args:
            request: script (full bash script text including #SBATCH headers) plus
                     optional overrides: job_name, partition, account, mem, cpus,
                     time_limit, gpus.
        """
        username = _auth(ctx)

        # Build sbatch CLI flags that override #SBATCH lines in the script
        flags: list[str] = []
        if request.job_name:   flags.append(f"--job-name={shlex.quote(request.job_name)}")
        if request.partition:  flags.append(f"--partition={shlex.quote(request.partition)}")
        if request.account:    flags.append(f"--account={shlex.quote(request.account)}")
        if request.mem:        flags.append(f"--mem={shlex.quote(request.mem)}")
        if request.cpus:       flags.append(f"--cpus-per-task={int(request.cpus)}")
        if request.time_limit: flags.append(f"--time={shlex.quote(request.time_limit)}")
        if request.gpus:       flags.append(f"--gres=gpu:{int(request.gpus)}")

        flags_str = " ".join(flags)

        # Pipe the script via stdin to avoid creating temp files on the remote
        safe_script = request.script.replace("'", "'\\''")
        cmd = f"echo '{safe_script}' | sbatch {flags_str}"
        out = run(username, cmd)

        success = "Submitted batch job" in out
        job_id  = None
        if success:
            parts  = out.split()
            job_id = parts[-1] if parts else None

        return {
            "success": success,
            "job_id":  job_id,
            "message": out,
            "note": None,
        }

    @app.tool(tags=["job", "cancel"])
    async def cancel_slurm_job(ctx: Context, job_id: str) -> dict:
        """
        Cancel a Slurm job by ID (equivalent to scancel).
        You can only cancel your own jobs (Slurm enforces this via the --user flag).

        Args:
            job_id: The numeric Slurm job ID to cancel
        """
        username = _auth(ctx)
        jid = shlex.quote(job_id)
        cmd = f"scancel --user={shlex.quote(username)} {jid}"
        out = run(username, cmd)

        # scancel outputs nothing on success
        success = not out.startswith("[") and "error" not in out.lower()
        return {
            "success": success,
            "job_id":  job_id,
            "message": f"Job {job_id} cancelled successfully." if success else out,
        }

    @app.tool(tags=["cluster", "summary"])
    async def get_cluster_summary(ctx: Context, partition: Optional[str] = None) -> str:
        """
        Get a full cluster overview: partition status, node states, and job counts.
        Optionally focus on a specific partition.

        Args:
            partition: Optional partition name to filter on (e.g. "gpu", "highmem")
        """
        username = _auth(ctx)

        sinfo_cmd  = f"sinfo --format='%.15P %.5a %.10l %.6D %.6t %N'"
        squeue_cmd = f"squeue {_squeue_format()} -a"
        if partition:
            p = shlex.quote(partition)
            sinfo_cmd  += f" -p {p}"
            squeue_cmd += f" -p {p}"

        sinfo_out  = run(username, sinfo_cmd)
        squeue_out = run(username, squeue_cmd)

        # Count job states from squeue
        state_counts: dict[str, int] = {}
        for line in squeue_out.splitlines()[1:]:   # skip header
            cols = line.split()
            if len(cols) >= 5:
                state = cols[4]
                state_counts[state] = state_counts.get(state, 0) + 1

        job_summary = "\n".join(
            f"- {s}: {c}" for s, c in sorted(state_counts.items())
        ) or "- (no jobs)"

        return (
            f"## IGS Cluster Summary{f' — {partition}' if partition else ''}\n\n"
            f"### Jobs\n{job_summary}\n\n"
            f"### Partitions / Nodes\n{sinfo_out}"
        )
