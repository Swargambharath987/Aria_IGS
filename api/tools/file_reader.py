"""
File reader tool — Phase 1.2 (read-only)
Reads job scripts and output files from the user's home directory via SSH.

Security contract:
- Paths are restricted to the user's own home directory (~/).
- Shell metacharacters are rejected outright.
- Directory traversal sequences (../) are rejected.
- Maximum file size enforced server-side with `head`.

All functions return plain strings for LLM context injection.
"""
import re

from tools.slurm_ssh import SSH_AVAILABLE, _run

# Allowed file extensions for reading
_ALLOWED_EXTENSIONS = {".sh", ".slurm", ".sbatch", ".out", ".err", ".log", ".py", ".R", ".txt", ".yaml", ".yml"}

# Max lines returned to keep context window sane
_MAX_LINES = 200

# Characters that could enable command injection
_UNSAFE_CHARS = re.compile(r"[;&|`$><\(\)\{\}\[\]\\]")

# Intent: user is asking to read/show/analyze a specific file or log
_INTENT_READ_FILE = re.compile(
    r"(?:"
    r"(?:read|show|open|display|cat|check|look at|view|analyze|analyse|print)\s+(?:the\s+)?"
    r"(?:file|script|log|output|error(?: log)?|job script)?\s*"
    r"(?P<path1>~?/[\w./\-]+\.(?:sh|slurm|sbatch|out|err|log|py|R|txt|yaml|yml))"
    r"|"
    r"(?P<path2>~?/[\w./\-]+\.(?:sh|slurm|sbatch|out|err|log|py|R|txt|yaml|yml))"
    r"\s+(?:file|script|log|output)?"
    r")",
    re.IGNORECASE,
)

# Intent: user wants to list files in a directory
_INTENT_LIST_FILES = re.compile(
    r"\b(?:list|show|what(?:'s| is| are)?|ls)\s+"
    r"(?:files?|scripts?|logs?|outputs?|jobs?)\s+(?:in\s+)?(?P<dir>~?/[\w./\-]+)",
    re.IGNORECASE,
)


def needs_file_read(message: str) -> dict | None:
    """
    Return {'action': 'read', 'path': ...} or {'action': 'list', 'dir': ...}
    if the message is asking to read a file or list directory contents.
    Returns None if no file intent is detected or SSH is unavailable.
    """
    if not SSH_AVAILABLE:
        return None

    m = _INTENT_READ_FILE.search(message)
    if m:
        path = m.group("path1") or m.group("path2")
        if path:
            return {"action": "read", "path": path}

    m = _INTENT_LIST_FILES.search(message)
    if m:
        return {"action": "list", "dir": m.group("dir")}

    return None


def _validate_path(path: str, username: str) -> str | None:
    """
    Validate that path is safe to read. Returns normalized path or None if rejected.
    - Must be in user's home directory
    - No shell metacharacters
    - No directory traversal
    - Allowed extensions only
    """
    if _UNSAFE_CHARS.search(path):
        return None

    if ".." in path:
        return None

    # Expand ~ to /home/<username>
    if path.startswith("~/"):
        path = f"/home/{username}/{path[2:]}"
    elif path.startswith("~"):
        path = f"/home/{username}/{path[1:]}"

    # Must be within user's home directory
    expected_prefix = f"/home/{username}/"
    if not path.startswith(expected_prefix):
        return None

    # Check extension
    dot = path.rfind(".")
    if dot == -1:
        return None
    ext = path[dot:].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return None

    return path


def read_file(username: str, path: str) -> str:
    """Read a file from the user's home directory and return its contents."""
    safe_path = _validate_path(path, username)
    if safe_path is None:
        return (
            f"[File read denied: '{path}' is not in your home directory, "
            f"uses an unsupported extension, or contains unsafe characters. "
            f"Readable types: .sh .slurm .sbatch .out .err .log .py .R .txt .yaml .yml]"
        )

    # Run head directly — stderr is captured so permission denied and missing file
    # errors surface clearly to the LLM rather than being swallowed.
    out = _run(username, f"head -n {_MAX_LINES} '{safe_path}' 2>&1")
    if not out or out.startswith("[Slurm"):
        return out or "[File read returned no output]"
    if "Permission denied" in out:
        return f"[Permission denied reading '{safe_path}' — this file is not accessible to Aria on this deployment]"
    if "No such file" in out:
        return f"[File not found: '{safe_path}' — check the path is correct]"

    line_count_out = _run(username, f"wc -l < '{safe_path}' 2>/dev/null")
    try:
        total_lines = int(line_count_out.strip())
        truncated = total_lines > _MAX_LINES
    except ValueError:
        truncated = False

    header = f"File: {safe_path}"
    if truncated:
        header += f" (showing first {_MAX_LINES} of {total_lines} lines)"
    return f"{header}\n{'─' * 60}\n{out}"


def list_job_files(username: str, directory: str) -> str:
    """List job-related files in a directory within the user's home."""
    # Validate directory path (reuse path validator with a dummy extension)
    if _UNSAFE_CHARS.search(directory):
        return "[Directory listing denied: path contains unsafe characters]"
    if ".." in directory:
        return "[Directory listing denied: directory traversal not allowed]"

    if directory.startswith("~/"):
        directory = f"/home/{username}/{directory[2:]}"
    elif directory.startswith("~"):
        directory = f"/home/{username}/{directory[1:]}"
    elif not directory.startswith(f"/home/{username}/"):
        return f"[Directory listing denied: '{directory}' is not in your home directory]"

    exts = "|".join(e.lstrip(".") for e in _ALLOWED_EXTENSIONS)
    out = _run(
        username,
        f"ls -lhrt '{directory}' 2>/dev/null | grep -E '\\.({exts})$' | tail -30",
    )
    if not out or out.startswith("[Slurm"):
        return out or f"[No job files found in {directory}]"

    return f"Files in {directory}:\n{out}"


def get_file_data(action: dict, username: str) -> str:
    """Dispatch to read_file or list_job_files based on the action dict from needs_file_read()."""
    if action["action"] == "read":
        return read_file(username, action["path"])
    if action["action"] == "list":
        return list_job_files(username, action["dir"])
    return ""
