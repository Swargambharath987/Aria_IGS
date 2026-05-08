"""
error_patterns.py — Pattern library for Slurm/HPC job failure log diagnosis.

Each pattern has:
  - name: short label
  - signatures: list of regex strings (case-insensitive)
  - explanation: plain-English description of what went wrong
  - fix: actionable fix template

diagnose(log_text) returns a dict with match results and evidence lines.
"""
import re
from typing import Optional

PATTERNS = [
    {
        "name": "OOM Kill",
        "signatures": [
            r"oom-kill event",
            r"\bKilled\b",
            r"oom_kill_process",
            r"Out of memory: Kill process",
        ],
        "explanation": "Job used more memory than requested and was killed by the OS kernel's out-of-memory manager.",
        "fix": (
            "Increase --mem. Your job requested {requested_mem} but needed more. "
            "Try doubling it."
        ),
    },
    {
        "name": "Time Limit Exceeded",
        "signatures": [
            r"CANCELLED AT .* DUE TO TIME LIMIT",
            r"DUE TO TIME LIMIT",
            r"\bTIME LIMIT\b",
        ],
        "explanation": "Job ran longer than the walltime requested and was cancelled by Slurm.",
        "fix": "Increase --time. Add 50% to your current time limit as a starting point.",
    },
    {
        "name": "CUDA Out of Memory",
        "signatures": [
            r"CUDA out of memory",
            r"RuntimeError: CUDA",
            r"torch\.cuda\.OutOfMemoryError",
            r"cudaMalloc failed",
        ],
        "explanation": "The GPU ran out of VRAM — the model or batch did not fit in GPU memory.",
        "fix": (
            "Reduce batch size, use gradient checkpointing, or request a GPU with more VRAM "
            "(--gres=gpu:a100:1)."
        ),
    },
    {
        "name": "Missing Module / Command Not Found",
        "signatures": [
            r"command not found",
            r"No module named",
            r"ModuleNotFoundError",
            r"ImportError",
            r"cannot find",
            r"module: command not found",
        ],
        "explanation": "A required software module or Python package is not loaded or installed.",
        "fix": (
            "Add the missing module to your script with `module load <name>` or install "
            "the Python package."
        ),
    },
    {
        "name": "Disk Quota Exceeded",
        "signatures": [
            r"Disk quota exceeded",
            r"No space left on device",
            r"\bquota\b",
            r"disk full",
        ],
        "explanation": "The filesystem is full or your quota is exceeded — the job could not write output.",
        "fix": (
            "Free up disk space with `du -sh ~/scratch/*` or move large files to /scratch."
        ),
    },
    {
        "name": "Segmentation Fault",
        "signatures": [
            r"Segmentation fault",
            r"segfault",
            r"signal 11",
            r"core dumped",
        ],
        "explanation": "The program crashed with a memory access error (segfault).",
        "fix": (
            "This is often a bug in the tool or a version mismatch. Check the tool version "
            "and if the error is reproducible on a subset of your data."
        ),
    },
    {
        "name": "MPI / Parallel Communication Error",
        "signatures": [
            r"Fatal error in PMPI",
            r"MPI_Abort",
            r"mpirun noticed",
            r"Abort\(1\)",
            r"ORTE_ERROR",
        ],
        "explanation": (
            "A parallel MPI job failed — usually one process crashed and killed the rest."
        ),
        "fix": (
            "Check which rank failed (look for the first error above the MPI abort). "
            "Reduce the number of MPI processes or check memory per rank."
        ),
    },
    {
        "name": "Permission Denied",
        "signatures": [
            r"Permission denied",
            r"cannot open",
            r"\bEACCES\b",
            r"\bEPERM\b",
        ],
        "explanation": "The job tried to read or write a file it doesn't have access to.",
        "fix": (
            "Check file permissions with `ls -la <path>`. Use `chmod` or contact the file owner."
        ),
    },
    {
        "name": "Node Failure / Preempted",
        "signatures": [
            r"\bNODE_FAIL\b",
            r"node failure",
            r"\bpreempted\b",
            r"\bPREEMPTED\b",
            r"Job step aborted",
        ],
        "explanation": (
            "The compute node failed or the job was preempted by a higher-priority job."
        ),
        "fix": (
            "This is not your fault — resubmit the job. Consider using `--requeue` in your "
            "sbatch header to auto-requeue on node failure."
        ),
    },
]

_MAX_EVIDENCE_LINES = 3
_FALLBACK_LINES = 20


def _extract_evidence(lines: list[str], matched_indices: list[int]) -> str:
    """Return up to _MAX_EVIDENCE_LINES lines that triggered the match."""
    seen = set()
    evidence = []
    for idx in matched_indices:
        if idx not in seen:
            seen.add(idx)
            evidence.append(lines[idx].rstrip())
        if len(evidence) >= _MAX_EVIDENCE_LINES:
            break
    return "\n".join(evidence)


def _fallback_lines(log_text: str) -> str:
    """Return first + last 20 lines of log for manual review."""
    lines = log_text.splitlines()
    if len(lines) <= _FALLBACK_LINES * 2:
        return "\n".join(lines)
    first = lines[:_FALLBACK_LINES]
    last = lines[-_FALLBACK_LINES:]
    return "\n".join(first) + "\n...\n" + "\n".join(last)


def _infer_requested_mem(log_text: str) -> str:
    """Try to extract the requested memory from the log (e.g. from #SBATCH --mem=)."""
    match = re.search(r"--mem[= ](\S+)", log_text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "your requested amount"


def diagnose(log_text: str) -> dict:
    """
    Match log_text against known HPC/Slurm failure patterns.

    Returns:
        {
            "matched": bool,
            "pattern_name": str,      # empty string if no match
            "explanation": str,
            "fix": str,
            "evidence": str,          # matching lines (up to 3), or first+last 20 if no match
        }
    """
    lines = log_text.splitlines()

    for pattern in PATTERNS:
        matched_indices: list[int] = []
        for sig in pattern["signatures"]:
            for i, line in enumerate(lines):
                if re.search(sig, line, re.IGNORECASE):
                    matched_indices.append(i)
        if matched_indices:
            requested_mem = _infer_requested_mem(log_text)
            fix = pattern["fix"].replace("{requested_mem}", requested_mem)
            evidence = _extract_evidence(lines, matched_indices)
            return {
                "matched": True,
                "pattern_name": pattern["name"],
                "explanation": pattern["explanation"],
                "fix": fix,
                "evidence": evidence,
            }

    # No pattern matched
    return {
        "matched": False,
        "pattern_name": "",
        "explanation": "",
        "fix": "",
        "evidence": _fallback_lines(log_text),
    }
