""" 
Configuration for the local-dataset variant of the API. 
 
DATASET_ROOT is the one folder on this machine the API is allowed to read 
MRI/mask series from. Every path a client sends in is resolved relative to 
this root and checked to make sure it can't escape it (no "..", no absolute 
paths, no symlink tricks) - otherwise any client with a valid API key could 
ask the server to read arbitrary files off disk. 
"""

import os
from pathlib import Path
from fastapi import HTTPException

# Set this in your .env, e.g.:
#   DATASET_ROOT=/data/Brain-Tumor-Progression
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "./data")).resolve()


def resolve_dataset_path(relative_path: str) -> Path:
    """ 
    Turns a client-supplied relative path (e.g. "PGBM-001/.../11.000000-T1post-03326") 
    into an absolute path, guaranteed to stay inside DATASET_ROOT. 
    """
    candidate = (DATASET_ROOT / relative_path).resolve()

    if not candidate.is_relative_to(DATASET_ROOT):
        raise HTTPException(
            status_code=400,
            detail="Path escapes the configured dataset root - not allowed.",
        )

    if not candidate.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No such path under dataset root: {relative_path}",
        )

    return candidate
