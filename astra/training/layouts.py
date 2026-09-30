"""Whole-section pseudo-Visium lattice, cropped using complete spot support."""

import math
import json
from pathlib import Path

import numpy as np
import torch

from astra.training.generator import OwnerMapSample, canonical_hd16_owner
from astra.training.ownership import masks_to_owner
from astra.training.utils import stable_seed


def section_lattice_origin(resource_id, seed):
    """One fixed phase in a primitive lattice cell, independent of expression."""
    u, v = np.random.default_rng(stable_seed(seed, resource_id, "v030_visium_lattice")).random(2)
    return (float(v * 50 * math.sqrt(3)), float(u * 100 + v * 50))


def core_supervision(*, batch_size=1, device="cpu", core_um=192):
    if type(core_um) is not int or not 0 < core_um <= 256 or core_um % 16:
        raise ValueError("centered supervision must align to the 8um output grid")
    mask = torch.zeros((batch_size, 128, 128), dtype=torch.bool, device=device)
    margin = (256 - core_um) // 4
    mask[:, margin:128-margin, margin:128-margin] = True
    return mask


def complete_spot_owner(centers_yx_um, *, fov_origin_yx_um, diameter_um=55.0, device="cpu"):
    """Bin centers discretize circles only after continuous full-support filtering.

    Global physical bin edges are 2*i um; centers are 2*i+1 um. Counts are
    aggregated by the existing collator over each retained complete support.
    """
    centers = np.asarray(centers_yx_um, dtype=np.float64).reshape(-1, 2)
    origin = np.asarray(fov_origin_yx_um, dtype=np.float64)
    if origin.shape != (2,) or not np.isfinite(centers).all() or not np.isfinite(origin).all() or not 0 < diameter_um <= 256:
        raise ValueError("invalid registered spot geometry")
    local = centers - origin
    radius = diameter_um / 2
    keep = np.all((local >= radius) & (local <= 256 - radius), axis=1)
    retained = np.flatnonzero(keep)
    if not len(retained):
        return OwnerMapSample(torch.full((128, 128), -1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.bool, device=device), dict(task="Spot55", retained_indices=[]))
    c = torch.tensor(local[keep], dtype=torch.float64, device=device)
    axis = (torch.arange(128, dtype=torch.float64, device=device) + 0.5) * 2
    masks = ((axis[None, :, None] - c[:, 0, None, None]).square()
             + (axis[None, None, :] - c[:, 1, None, None]).square()) <= radius**2
    owner, valid = masks_to_owner(masks, check_connectivity=True)
    return OwnerMapSample(owner, valid, dict(task="Spot55", retained_indices=retained.tolist(),
        centers_yx_um=centers[keep].tolist(), diameter_um=diameter_um,
        boundary_policy="complete_spots_only", count_scope="full_retained_spot"))


def lattice_spot_owner(start_yx_2um, *, lattice_origin_yx_um, device="cpu"):
    start = np.asarray(start_yx_2um)
    if start.shape != (2,) or not np.issubdtype(start.dtype, np.integer) or np.any(start < 0) or np.any(start % 8):
        raise ValueError("FOV must use the frozen native 16um-aligned coordinates")
    origin = start * 2
    ay, ax = map(float, lattice_origin_yx_um)
    if not math.isfinite(ay + ax):
        raise ValueError("nonfinite lattice origin")
    spacing_y = 50 * math.sqrt(3)
    centers, ids = [], []
    for row in range(math.floor((origin[0] - ay - 27.5) / spacing_y),
                     math.ceil((origin[0] + 256 - ay + 27.5) / spacing_y) + 1):
        offset_x = ax + row * 50
        for col in range(math.floor((origin[1] - offset_x - 27.5) / 100),
                         math.ceil((origin[1] + 256 - offset_x + 27.5) / 100) + 1):
            centers.append((ay + row * spacing_y, offset_x + col * 100))
            ids.append((row, col))
    sample = complete_spot_owner(centers, fov_origin_yx_um=origin, device=device)
    sample.parameters.update(lattice_origin_yx_um=[ay, ax], pitch_um=100,
        spot_ids=[list(ids[i]) for i in sample.parameters["retained_indices"]],
        fov_origin_yx_um=origin.tolist(), program="whole_section_hexagonal_lattice")
    return sample


def evaluation_owner(task, start_yx_2um, *, lattice_origin_yx_um, device="cpu"):
    if task == "HD16":
        return canonical_hd16_owner(device=device)
    if task == "Spot55":
        return lattice_spot_owner(start_yx_2um, lattice_origin_yx_um=lattice_origin_yx_um, device=device)
    raise ValueError("only HD16 and Spot55 are primary evaluation tasks")


def load_validation_owner(config, root, resource_id, field_index, expected_start, *, task, device="cpu"):
    if resource_id not in config["wt"]["validation"]:
        raise PermissionError("layout resource is outside validation")
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    record = json.loads((root / resource_id / "section.json").read_text())
    if summary["status"] != "complete" or record["role"] != "validation" or record["resource_id"] != resource_id:
        raise ValueError("validation layout identity differs")
    if type(field_index) is not int or not 0 <= field_index < record["fields"]:
        raise ValueError("validation field index out of bounds")
    starts = np.load(root / resource_id / "starts_yx.npy", mmap_mode="r")
    np.testing.assert_array_equal(starts[field_index], expected_start)
    if task == "HD16":
        return canonical_hd16_owner(device=device)
    if task != "Spot55":
        raise ValueError("unknown validation task")
    owners = np.load(root / resource_id / "spot55_owner.npy", mmap_mode="r")
    owner = torch.tensor(owners[field_index], dtype=torch.long, device=device)
    return OwnerMapSample(owner, torch.ones(record["parents_per_fov"][field_index], dtype=torch.bool, device=device),
                          record["fovs"][field_index])
