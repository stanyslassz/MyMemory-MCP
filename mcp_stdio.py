#!/usr/bin/env python3
"""Claude Desktop stdio launcher for memory-ai MCP server.

Usage in Claude Desktop config:
  "command": "/absolute/path/to/memory-ai/.venv/bin/python",
  "args": ["/absolute/path/to/memory-ai/mcp_stdio.py"]

Self-contained: sets sys.path AND cwd to project root so config.yaml,
memory/, and prompts/ resolve correctly regardless of how Claude Desktop
spawns the process.
"""
import os
import sys
from pathlib import Path

# Anchor everything to this script's directory (project root)
_PROJECT_ROOT = Path(__file__).resolve().parent

# 1. Make `import src.*` work without pip install
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 2. Make load_config() find config.yaml and resolve relative paths
os.chdir(_PROJECT_ROOT)

from src.mcp.server import run_server  # noqa: E402

if __name__ == "__main__":
    run_server(transport_override="stdio")
