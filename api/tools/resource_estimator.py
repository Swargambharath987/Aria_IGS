"""
resource_estimator.py — Rule-based Slurm resource estimator (Module 2.1).

Pure Python, no LLM calls. Uses regex/string matching against a knowledge table
of bioinformatics tool resource profiles to produce SBATCH header recommendations.
"""
import math
import re

# ---------------------------------------------------------------------------
# Knowledge table — community-norm resource profiles per tool
# ---------------------------------------------------------------------------

TOOL_PROFILES = {
    "bwa":            {"mem_gb": 8,   "cpus": 8,  "partition": "defq"},
    "bwa-mem2":       {"mem_gb": 32,  "cpus": 16, "partition": "defq"},
    "star":           {"mem_gb": 40,  "cpus": 12, "partition": "defq"},
    "hisat2":         {"mem_gb": 8,   "cpus": 8,  "partition": "defq"},
    "bowtie2":        {"mem_gb": 4,   "cpus": 8,  "partition": "defq"},
    "samtools":       {"mem_gb": 8,   "cpus": 4,  "partition": "defq"},
    "gatk":           {"mem_gb": 16,  "cpus": 4,  "partition": "defq"},
    "haplotypecaller":{"mem_gb": 16,  "cpus": 2,  "partition": "defq"},
    "deseq2":         {"mem_gb": 16,  "cpus": 4,  "partition": "defq"},
    "salmon":         {"mem_gb": 8,   "cpus": 8,  "partition": "defq"},
    "kallisto":       {"mem_gb": 4,   "cpus": 8,  "partition": "defq"},
    "flye":           {"mem_gb": 128, "cpus": 16, "partition": "highmem"},
    "spades":         {"mem_gb": 64,  "cpus": 16, "partition": "highmem"},
    "trinity":        {"mem_gb": 64,  "cpus": 16, "partition": "highmem"},
    "nextflow":       {"mem_gb": 8,   "cpus": 4,  "partition": "defq"},
    "snakemake":      {"mem_gb": 8,   "cpus": 4,  "partition": "defq"},
    "torch":          {"mem_gb": 32,  "cpus": 8,  "partition": "gpu", "gpu": 1},
    "tensorflow":     {"mem_gb": 32,  "cpus": 8,  "partition": "gpu", "gpu": 1},
    "cuda":           {"mem_gb": 32,  "cpus": 8,  "partition": "gpu", "gpu": 1},
}

# Human-readable display names for detected tools
TOOL_DISPLAY = {
    "bwa":            "BWA (aligner)",
    "bwa-mem2":       "BWA-MEM2 (aligner)",
    "star":           "STAR (aligner)",
    "hisat2":         "HISAT2 (aligner)",
    "bowtie2":        "Bowtie2 (aligner)",
    "samtools":       "samtools",
    "gatk":           "GATK",
    "haplotypecaller":"GATK HaplotypeCaller",
    "deseq2":         "DESeq2",
    "salmon":         "Salmon (quant)",
    "kallisto":       "kallisto (quant)",
    "flye":           "Flye (assembler)",
    "spades":         "SPAdes (assembler)",
    "trinity":        "Trinity (assembler)",
    "nextflow":       "Nextflow (workflow)",
    "snakemake":      "Snakemake (workflow)",
    "torch":          "PyTorch (GPU)",
    "tensorflow":     "TensorFlow (GPU)",
    "cuda":           "CUDA",
}

# Regex patterns to find each tool name in script text (case-insensitive)
TOOL_PATTERNS: dict[str, str] = {
    "bwa-mem2":       r"\bbwa-mem2\b",
    "bwa":            r"\bbwa\b",
    "star":           r"\bSTAR\b|\bstar\b",
    "hisat2":         r"\bhisat2\b",
    "bowtie2":        r"\bbowtie2\b",
    "samtools":       r"\bsamtools\b",
    "haplotypecaller":r"\bHaplotypeCaller\b|\bhaplotypecaller\b",
    "gatk":           r"\bgatk\b|\bGATK\b",
    "deseq2":         r"\bDESeq2\b|\bdeseq2\b",
    "salmon":         r"\bsalmon\b|\bSalmon\b",
    "kallisto":       r"\bkallisto\b",
    "flye":           r"\bflye\b|\bFlye\b",
    "spades":         r"\bspades\.py\b|\bspades\b|\bSPAdes\b",
    "trinity":        r"\bTrinity\b|\btrinity\b",
    "nextflow":       r"\bnextflow\b|\bNextflow\b",
    "snakemake":      r"\bsnakemake\b|\bSnakemake\b",
    "torch":          r"\btorch\b|\bpytorch\b|\bimport torch\b|\.to\(['\"]cuda['\"]\)",
    "tensorflow":     r"\btensorflow\b|\bimport tensorflow\b|\btf\.\b",
    "cuda":           r"\bcuda\b|\bCUDA\b|nvcc\b",
}

# GPU-specific additional signals
GPU_PATTERNS = [
    r"\.to\(['\"]cuda['\"]\)",
    r"\bimport torch\b",
    r"\bimport tensorflow\b",
    r"\bcuda\b",
    r"\bgpu\b",
    r"#SBATCH\s+--gres=gpu",
]

# Thread/parallelism flag patterns — extract numeric value
THREAD_PATTERNS = [
    r"-t\s+(\d+)",
    r"-p\s+(\d+)",
    r"--threads[= ](\d+)",
    r"--nthreads[= ](\d+)",
    r"nthreads\s*=\s*(\d+)",
    r"SLURM_CPUS_PER_TASK\s*[=:]\s*(\d+)",
    r"--runThreadN\s+(\d+)",      # STAR
    r"-@\s+(\d+)",                # samtools
    r"--cpu\s+(\d+)",
    r"--cores\s+(\d+)",
]

# Workload type categories for walltime decisions
ASSEMBLER_TOOLS  = {"flye", "spades", "trinity"}
ALIGNMENT_TOOLS  = {"bwa", "bwa-mem2", "star", "hisat2", "bowtie2"}
GPU_TOOLS        = {"torch", "tensorflow", "cuda"}


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_tools(script: str) -> list[str]:
    """Return list of matched tool keys (order: bwa-mem2 before bwa, etc.)."""
    found = []
    for tool, pattern in TOOL_PATTERNS.items():
        if re.search(pattern, script, re.IGNORECASE):
            found.append(tool)
    return found


def _detect_explicit_threads(script: str) -> int | None:
    """Return the highest explicit thread count found in the script, or None."""
    counts = []
    for pattern in THREAD_PATTERNS:
        for m in re.finditer(pattern, script, re.IGNORECASE):
            try:
                counts.append(int(m.group(1)))
            except (IndexError, ValueError):
                pass
    return max(counts) if counts else None


def _detect_language(script: str) -> str:
    """Best-guess programming language / framework."""
    first_line = script.strip().split("\n")[0] if script.strip() else ""
    if "python" in first_line.lower():
        return "Python"
    if re.search(r"\bRscript\b|\b\.R\b|library\(", script):
        return "R"
    if re.search(r"\bnextflow\b", script, re.IGNORECASE):
        return "Nextflow"
    if re.search(r"\bsnakemake\b", script, re.IGNORECASE):
        return "Snakemake"
    if first_line.startswith("#!") and "bash" in first_line:
        return "Bash"
    if re.search(r"#!/usr/bin/env\s+python", script):
        return "Python"
    return "Bash"


def _workload_type(tools: list[str]) -> str:
    """Classify workload for human-readable description."""
    tool_set = set(tools)
    has_align    = bool(tool_set & ALIGNMENT_TOOLS)
    has_assemble = bool(tool_set & ASSEMBLER_TOOLS)
    has_gpu      = bool(tool_set & GPU_TOOLS)
    has_variant  = "haplotypecaller" in tool_set or "gatk" in tool_set
    has_quant    = "salmon" in tool_set or "kallisto" in tool_set or "deseq2" in tool_set
    has_workflow = "nextflow" in tool_set or "snakemake" in tool_set

    parts = []
    if has_gpu:         parts.append("GPU/ML training")
    if has_assemble:    parts.append("de novo assembly")
    if has_align:       parts.append("short-read alignment")
    if has_variant:     parts.append("variant calling")
    if has_quant:       parts.append("quantification/DE analysis")
    if has_workflow:    parts.append("workflow management")
    return ", ".join(parts).capitalize() if parts else "General compute"


def _choose_walltime(tools: list[str], partition: str) -> str:
    tool_set = set(tools)
    if tool_set & ASSEMBLER_TOOLS:
        return "72:00:00"
    if tool_set & GPU_TOOLS:
        return "24:00:00"
    if tool_set & ALIGNMENT_TOOLS:
        return "24:00:00"
    if tool_set:
        return "12:00:00"   # bioinformatics but no heavy alignment
    return "04:00:00"       # nothing detected → conservative


def _round_mem(mem_gb: float) -> str:
    """Round up to nearest 'nice' boundary and return as string with G suffix."""
    # Boundaries: 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512
    nice = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    for boundary in nice:
        if mem_gb <= boundary:
            return f"{boundary}G"
    return f"{math.ceil(mem_gb)}G"


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------

def estimate(script_content: str) -> str:
    """
    Analyze a job script and return a formatted resource recommendation string.

    Parameters
    ----------
    script_content : str
        Full text of a job script (bash, Python, R, or workflow file).

    Returns
    -------
    str
        Human-readable markdown-style recommendation with a ready-to-paste
        #SBATCH header block.
    """
    script = script_content or ""

    # 1. Detect tools, language, explicit threads
    detected_tools   = _detect_tools(script)
    language         = _detect_language(script)
    explicit_threads = _detect_explicit_threads(script)

    # 2. Aggregate resource requirements across detected tools
    if detected_tools:
        max_mem_gb  = max(TOOL_PROFILES[t]["mem_gb"] for t in detected_tools)
        max_cpus    = max(TOOL_PROFILES[t]["cpus"]   for t in detected_tools)
        # Partition precedence: gpu > highmem > defq
        partitions  = [TOOL_PROFILES[t]["partition"] for t in detected_tools]
        if "gpu" in partitions:
            partition = "gpu"
        elif "highmem" in partitions:
            partition = "highmem"
        else:
            partition = "defq"
        needs_gpu   = any(TOOL_PROFILES[t].get("gpu") for t in detected_tools)
    else:
        # Conservative defaults when nothing is detected
        max_mem_gb  = 8
        max_cpus    = 4
        partition   = "defq"
        needs_gpu   = False

    # 3. Honour explicit thread flags from the script (take the higher value)
    if explicit_threads:
        max_cpus = max(max_cpus, explicit_threads)

    # 4. Add 20% memory headroom, then round to a nice boundary
    mem_with_headroom = max_mem_gb * 1.20
    mem_str           = _round_mem(mem_with_headroom)

    # 5. Walltime
    walltime = _choose_walltime(detected_tools, partition)

    # 6. Workload description
    workload = _workload_type(detected_tools)

    # 7. Build reasoning bullets
    reasoning_lines = []

    if detected_tools:
        mem_breakdown = ", ".join(
            f"{TOOL_DISPLAY.get(t, t)} ~{TOOL_PROFILES[t]['mem_gb']}GB"
            for t in detected_tools
        )
        reasoning_lines.append(
            f"Memory: peak tool requirement is {max_mem_gb}GB ({mem_breakdown}); "
            f"with 20% headroom → {mem_str}"
        )
    else:
        reasoning_lines.append(
            "No recognised bioinformatics tools detected — using conservative defaults "
            "(8GB, 4 CPUs, defq, 4h). Adjust after profiling your workload."
        )

    if explicit_threads:
        reasoning_lines.append(
            f"CPUs: explicit thread flag found ({explicit_threads} threads) — "
            f"using {max_cpus} (whichever was higher: tool default vs script flag)"
        )
    else:
        reasoning_lines.append(
            f"CPUs: {max_cpus} — based on highest tool default across detected tools"
        )

    partition_reason = {
        "gpu":     "GPU partition required (PyTorch / TensorFlow / CUDA detected)",
        "highmem": "highmem partition required (assembler needs >32GB RAM)",
        "defq":    "defq — no GPU needed, memory under highmem threshold",
    }
    reasoning_lines.append(f"Partition: {partition} — {partition_reason[partition]}")

    walltime_map = {
        "72:00:00": "72h for de novo assembly (Flye / SPAdes / Trinity are memory- and time-intensive)",
        "24:00:00": "24h for alignment or GPU training pipelines on a single sample",
        "12:00:00": "12h for bioinformatics without heavy alignment",
        "04:00:00": "4h conservative default — adjust based on your actual run time",
    }
    reasoning_lines.append(f"Walltime: {walltime_map.get(walltime, walltime)}")

    # 8. Build the #SBATCH header
    sbatch_lines = [
        f"#SBATCH --partition={partition}",
        f"#SBATCH --cpus-per-task={max_cpus}",
        f"#SBATCH --mem={mem_str}",
        f"#SBATCH --time={walltime}",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
    ]
    if needs_gpu:
        sbatch_lines.insert(3, "#SBATCH --gres=gpu:1")

    sbatch_block = "\n".join(sbatch_lines)

    # 9. Detected tool display list
    if detected_tools:
        tool_list_str = ", ".join(TOOL_DISPLAY.get(t, t) for t in detected_tools)
    else:
        tool_list_str = "None recognised"

    # 10. Assemble final output
    reasoning_str = "\n".join(f"- {line}" for line in reasoning_lines)

    output = f"""\
## Resource Recommendation

**Detected language/framework:** {language}
**Detected tools:** {tool_list_str}
**Workload type:** {workload}

### Recommended #SBATCH header:
```bash
{sbatch_block}
```

**Reasoning:**
{reasoning_str}

**Note:** These are estimates based on community norms. After your first run, check \
actual usage with `seff <job_id>` and tune accordingly.\
"""
    return output
