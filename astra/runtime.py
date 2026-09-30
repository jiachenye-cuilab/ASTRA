"""Release-relative assets, content identities, and CUDA resource limits."""
import hashlib
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from astra.assets import ROOT


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()



def configure_cuda(device, memory_gib):
    properties = torch.cuda.get_device_properties(device)
    total_gib = properties.total_memory / 2**30
    budget_gib = min(memory_gib, .75 * total_gib)
    fraction = budget_gib / total_gib
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    torch.cuda.reset_peak_memory_stats(device)
    return dict(device_name=properties.name, total_gpu_memory_gib=total_gib,
                gpu_memory_limit_gib=budget_gib, cuda_memory_fraction=fraction)
