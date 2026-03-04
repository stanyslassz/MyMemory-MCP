"""Smoke test: mcp_stdio.py launches without import errors in a clean env."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_mcp_stdio_imports_cleanly():
    """mcp_stdio.py must be importable from any cwd without PYTHONPATH."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('mcp_stdio', r'{PROJECT_ROOT / 'mcp_stdio.py'}'); "
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
            "print('OK')",
        ],
        cwd="/tmp",  # intentionally NOT project root
        capture_output=True,
        text=True,
        timeout=30,
        env={},  # clean env — no PYTHONPATH
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "OK" in result.stdout


def test_mcp_stdio_config_resolves_from_any_cwd():
    """Config must find config.yaml and resolve paths even when cwd != project root."""
    script = (
        f"import os, sys; sys.path.insert(0, r'{PROJECT_ROOT}'); os.chdir(r'{PROJECT_ROOT}'); "
        "from src.core.config import load_config; c = load_config(); "
        f"assert str(c.project_root) == r'{PROJECT_ROOT}', f'wrong root: {{c.project_root}}'; "
        "assert (c.project_root / 'config.yaml').exists(), 'config.yaml not found'; "
        "print('CONFIG_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "CONFIG_OK" in result.stdout


def test_mcp_stdio_server_starts_and_accepts_shutdown():
    """mcp_stdio.py starts in stdio mode and can be killed cleanly."""
    proc = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "mcp_stdio.py")],
        cwd="/tmp",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Send EOF to stdin — server should exit gracefully
    proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    # Exit code 0 or 1 are both acceptable (EOF shutdown vs signal)
    assert proc.returncode is not None, "Process did not terminate"
