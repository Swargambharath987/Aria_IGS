"""
MCP Resources — static reference data and live cluster info.
Resources are read by the LLM at context-building time, not on-demand like tools.
"""
import asyncio

from fastmcp import FastMCP

from client import SlurmClient


def register_resources(app: FastMCP):

    @app.resource("slurm://docs/job-templates")
    def get_job_templates() -> str:
        """Job script templates for common workloads on the IGS cluster."""
        return """# IGS Slurm Job Templates

## CPU job (defq partition)
```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load <software>
<your commands>
```

## GPU job (gpu partition)
```bash
#!/bin/bash
#SBATCH --job-name=gpu_train
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load cuda
python train.py
```

## High-memory job (highmem partition)
```bash
#!/bin/bash
#SBATCH --job-name=assembly
#SBATCH --partition=highmem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

<genome assembly command>
```

## Job array (process multiple samples)
```bash
#!/bin/bash
#SBATCH --job-name=sample_array
#SBATCH --partition=defq
#SBATCH --array=1-100%20
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%A_%a.out

SAMPLE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" samples.txt)
<process $SAMPLE>
```
"""

    @app.resource("slurm://docs/usage-guide")
    def get_slurm_guide() -> str:
        """IGS Slurm quick-reference guide."""
        return """# IGS Slurm Quick Reference

## Submit a job
```bash
sbatch myjob.sh
```

## Check your queue
```bash
squeue -u $USER
```

## Cancel a job
```bash
scancel <job_id>
```

## Check job efficiency after completion
```bash
seff <job_id>
```

## Check detailed accounting
```bash
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

## Interactive session
```bash
srun --pty --partition=defq --cpus-per-task=4 --mem=16G bash
```

## Partitions
| Partition | Use case | Max time |
|-----------|----------|----------|
| defq      | General CPU jobs | 7 days |
| highmem   | Large memory (>256G) | 7 days |
| gpu       | GPU training / inference | 7 days |

## Common states
- `PD` — Pending (waiting for resources)
- `R`  — Running
- `CG` — Completing
- `F`  — Failed
- `TO` — Timeout
"""

    @app.resource("slurm://cluster/tres")
    def get_cluster_tres() -> str:
        """TRES (Trackable RESources) types available on the IGS cluster."""
        return """# IGS Cluster TRES Reference

| TRES | Description | Example |
|------|-------------|---------|
| cpu | CPU cores | --cpus-per-task=4 |
| mem | Memory per node | --mem=16G |
| node | Node count | --nodes=2 |
| billing | Billing units | (auto-calculated) |
| gres/gpu | GPU units | --gres=gpu:1 |
| gres/gpu:a100 | Specific GPU model | --gres=gpu:a100:1 |
"""

    @app.resource("slurm://cluster/nodes")
    async def get_cluster_nodes(slurm_token: str = "", slurm_user: str = "") -> str:
        """Live node status from the IGS cluster."""
        if not slurm_token or not slurm_user:
            return "[Authentication required — provide your Slurm JWT token]"
        try:
            client = SlurmClient(slurm_user, slurm_token)
            data   = await client.get("/nodes")
            nodes  = data.get("nodes", [])
            if not nodes:
                return "No node data available."
            rows = ["| Node | State | CPUs | Memory | Partitions |",
                    "|------|-------|------|--------|------------|"]
            for n in nodes:
                rows.append(
                    f"| {n.get('name','')} "
                    f"| {n.get('state',[''])[0]} "
                    f"| {n.get('cpus','')} "
                    f"| {n.get('real_memory','')}M "
                    f"| {', '.join(n.get('partitions', []))} |"
                )
            return "\n".join(rows)
        except Exception as e:
            return f"[Error fetching nodes: {e}]"

    @app.resource("slurm://cluster/partitions")
    async def get_cluster_partitions(slurm_token: str = "", slurm_user: str = "") -> str:
        """Live partition info from the IGS cluster."""
        if not slurm_token or not slurm_user:
            return "[Authentication required — provide your Slurm JWT token]"
        try:
            client = SlurmClient(slurm_user, slurm_token)
            data   = await client.get("/partitions")
            parts  = data.get("partitions", [])
            rows   = ["| Partition | State | Nodes | Max time | Default mem |",
                      "|-----------|-------|-------|----------|-------------|"]
            for p in parts:
                rows.append(
                    f"| {p.get('name','')} "
                    f"| {p.get('state','')} "
                    f"| {p.get('nodes', {}).get('total','')} "
                    f"| {p.get('maximum_time', {}).get('number','')} min "
                    f"| {p.get('defaults', {}).get('memory_per_node', {}).get('number','')}M |"
                )
            return "\n".join(rows)
        except Exception as e:
            return f"[Error fetching partitions: {e}]"
