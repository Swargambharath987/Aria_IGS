from fastmcp import FastMCP

from resources import register_resources
from prompts import register_prompts
from tools import register_tools

app = FastMCP(
    name="slurm-igs",
    instructions=(
        "MCP server for the IGS HPC cluster at University of Maryland. "
        "Provides full Slurm job management: submit, cancel, monitor, and cluster status. "
        "Each call requires a valid Slurm JWT — generate one on the cluster with: "
        "`scontrol token lifespan=3600`"
    ),
)

register_tools(app)
register_resources(app)
register_prompts(app)
