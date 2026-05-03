"""
Entry point: python -m mcp_server [--stdio] [--host HOST] [--port PORT]

Default: SSE transport on port 8001 (Docker service, HTTP clients)
--stdio: stdio transport for Claude Desktop direct integration
"""
import argparse
from app import app

parser = argparse.ArgumentParser(description="Slurm IGS MCP Server")
parser.add_argument("--stdio", action="store_true", help="Use stdio transport (Claude Desktop)")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8001)
args = parser.parse_args()

if args.stdio:
    app.run(transport="stdio")
else:
    app.run(transport="sse", host=args.host, port=args.port)
