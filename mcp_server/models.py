from typing import List, Optional
from pydantic import BaseModel, Field


class JobSubmissionRequest(BaseModel):
    """
    Represents a job to submit via sbatch.
    The script must be a complete bash script (including #!/bin/bash and #SBATCH directives).
    Extra SBATCH directives can be added via sbatch_args (these override anything in the script).
    """
    script:      str            # full bash script content including #SBATCH headers
    job_name:    Optional[str]  = Field(None, max_length=64, pattern=r"^[a-zA-Z0-9_\-]*$")
    partition:   Optional[str]  = Field(None, max_length=32)
    account:     Optional[str]  = Field(None, max_length=32)
    # These override #SBATCH directives already in the script
    mem:         Optional[str]  = None   # e.g. "8G", "1024M"
    cpus:        Optional[int]  = Field(None, ge=1, le=128)
    time_limit:  Optional[str]  = None   # e.g. "24:00:00"
    gpus:        Optional[int]  = Field(None, ge=0, le=8)


class JobListFilters(BaseModel):
    state:     Optional[List[str]] = None
    user:      Optional[str]       = None
    partition: Optional[str]       = None
    limit:     Optional[int]       = Field(20, ge=1, le=1000)
    offset:    Optional[int]       = Field(0, ge=0)
