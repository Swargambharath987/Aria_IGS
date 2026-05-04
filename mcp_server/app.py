from fastmcp import FastMCP

from resources import register_resources
from prompts import register_prompts
from tools import register_tools

app = FastMCP(
    name="slurm-igs",
    instructions=(
        "MCP server for the IGS HPC cluster at University of Maryland. "
        "Provides Slurm job management via SSH: list jobs, view details, check efficiency, "
        "get cluster status, and (with admin approval) submit and cancel jobs. "
        "Authenticate by passing your Slurm JWT as a Bearer token — generate one on the "
        "cluster with: scontrol token lifespan=3600"
    ),
)

register_tools(app)
register_resources(app)
register_prompts(app)
