from typing import List, Optional
from pydantic import BaseModel, Field


class JobSpec(BaseModel):
    name:                      Optional[str]       = Field(None, max_length=64, pattern=r"^[a-zA-Z0-9_\-]*$")
    nodes:                     Optional[str]       = None   # e.g. "1" or "1-4"
    current_working_directory: str                 = "/tmp"
    partition:                 Optional[str]       = Field(None, max_length=32)
    environment:               List[str]           = ["PATH=/bin:/usr/bin:/usr/local/bin:."]
    tres_per_job:              Optional[str]       = None   # e.g. "gres/gpu=1,mem=30G"
    standard_output:           Optional[str]       = None
    standard_error:            Optional[str]       = None
    account:                   Optional[str]       = Field(None, max_length=32)
    time_limit:                Optional[int]       = Field(None, ge=1, le=10080)  # minutes, max 1 week
    memory_per_node:           Optional[str]       = None   # e.g. "8G", "1024M"
    cpus_per_task:             Optional[int]       = Field(None, ge=1, le=128)


class JobSubmissionRequest(BaseModel):
    job:    JobSpec
    script: str   # full bash script content


class JobListFilters(BaseModel):
    state:     Optional[List[str]] = None
    user:      Optional[str]       = None
    partition: Optional[str]       = None
    limit:     Optional[int]       = Field(20, ge=1, le=1000)
    offset:    Optional[int]       = Field(0, ge=0)
