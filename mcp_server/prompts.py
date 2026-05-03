"""
MCP Prompts — guided workflow templates for common Slurm tasks.
These generate structured instructions for the LLM to follow.
"""
from typing import Optional
from fastmcp import FastMCP


def register_prompts(app: FastMCP):

    @app.prompt()
    def check_running_jobs() -> str:
        """Guide the user through checking their current jobs and diagnosing issues."""
        return """You are helping a researcher check their IGS cluster jobs.

Steps:
1. Call `list_cluster_jobs` with their username to get the current queue.
2. For any job in RUNNING state: call `get_job_details` with its job_id for full resource usage.
3. For any job in PENDING state: explain the reason field (e.g. Resources, Priority, QOSMaxJobsPerUser).
4. Flag any jobs in FAILED or TIMEOUT state and suggest next steps (check .err file, increase walltime).
5. Summarize: N running, N pending, N failed.

Keep responses concise and actionable. Always show the job_id so the user can reference it."""

    @app.prompt()
    def submit_gpu_training_job(project_directory: str, script_name: str) -> str:
        """Generate a GPU job submission for a training workload."""
        return f"""Help the user submit a GPU training job on the IGS cluster.

Context:
- Project directory: {project_directory}
- Script: {script_name}
- Target partition: gpu
- Typical resources: 1 GPU, 4 CPUs, 30G memory

Steps:
1. Confirm the working directory and script exist (ask if unsure).
2. Build a `JobSubmissionRequest` with:
   - partition: "gpu"
   - tres_per_job: "gres/gpu=1"
   - cpus_per_task: 4
   - memory_per_node: "30G"
   - current_working_directory: "{project_directory}"
   - standard_output: "{project_directory}/logs/%j.out"
   - standard_error: "{project_directory}/logs/%j.err"
3. Call `submit_slurm_job` with the request.
4. Report the assigned job_id and tell the user how to monitor it with `list_cluster_jobs`.
"""

    @app.prompt()
    def cluster_status_overview() -> str:
        """Full cluster diagnostic — queue, nodes, partitions, and recommendations."""
        return """Run a full IGS cluster status check.

Steps (run in this order):
1. Call `ping` — confirm the API is reachable and show the cluster URL.
2. Call `get_cluster_summary` — get the full picture: job states, node states, partition load.
3. Highlight:
   - Any partition with >80% nodes in use (congested)
   - Any nodes in DRAIN or DOWN state
   - Estimated wait time based on pending job count
4. Give a 3-bullet recommendation: best partition for a new job right now, expected queue time, any issues to watch.
"""

    @app.prompt()
    def job_submission_guide(job_type: Optional[str] = None) -> str:
        """Produce step-by-step job submission guidance, optionally for a specific job type."""
        if job_type:
            t = job_type.lower()
            if "gpu" in t:
                guidance = "GPU job: use --partition=gpu, --gres=gpu:1, typically 4 CPUs and 30G memory."
            elif "array" in t:
                guidance = "Job array: use --array=1-N%concurrent. Access sample with $SLURM_ARRAY_TASK_ID."
            elif "highmem" in t or "memory" in t or "mem" in t:
                guidance = "High-memory job: use --partition=highmem. Request memory in G (e.g. --mem=512G)."
            elif "mpi" in t:
                guidance = "MPI job: use --ntasks=N (not --cpus-per-task). Load the MPI module before running."
            elif "interactive" in t:
                guidance = "Interactive session: use `srun --pty bash` with your resource flags."
            else:
                guidance = f"For '{job_type}' jobs: check the job templates resource for a starting point."
        else:
            guidance = "General job submission: start from the templates in slurm://docs/job-templates."

        return f"""Help the user submit a Slurm job on the IGS cluster.

Job type guidance: {guidance}

Steps:
1. Ask for: script location, partition preference, CPUs, memory, time limit (if not given).
2. Read the script if the user provides a path (use `read_user_file`).
3. Build a `JobSubmissionRequest` from what you learn.
4. Call `submit_slurm_job`.
5. Report the job_id and show the command to monitor it.

Always remind the user to check their .err file first if the job fails.
"""
