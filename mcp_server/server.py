"""
Entry point for the Slurm MCP server.

Usage:
  # SSE transport (Docker / HTTP clients):
  python server.py

  # stdio transport (Claude Desktop):
  python server.py --stdio
"""
import argparse
import sys

from slurm_mcp import mcp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdio", action="store_true", help="Use stdio transport (Claude Desktop)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.stdio:
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=args.host, port=args.port)
