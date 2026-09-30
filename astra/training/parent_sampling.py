"""Target-free, deterministic random/HD16/Spot55 training observations."""

from collections import defaultdict
import hashlib
import math

import numpy as np
import torch

from astra.training.cache import authorized_ids
from astra.training.generator import GeneratorConfig, OwnerMapSample, axis_aligned_square_owner, canonical_hd16_owner, generate_random_owner
from astra.training.parent_views import PHASES, validate_view_policy, view_sampling_report
from astra.training.layouts import lattice_spot_owner, section_lattice_origin
from astra.training.utils import stable_seed

TASKS = ("random", "HD16", "Spot55")


def random_owner_with_resampling(seed, generator, stratum):
    """Retain the historical deterministic retry policy without dropping a FOV."""
    for retry in range(8):
        effective = seed if retry == 0 else int(hashlib.sha256(
            f"{seed}:{retry}:v012_resample".encode()).hexdigest()[:16], 16) % (2**63 - 1)
        try:
            sample = generate_random_owner(effective, config=generator, coverage_stratum=stratum,
                                           device="cpu", audit_connectivity=True)
        except RuntimeError as error:
            if str(error) != f"v033 generator failed after {generator.maximum_attempts} attempts for seed {effective}":
                raise
            if retry == 7:
                raise RuntimeError(f"random owner generator exhausted requested seed {seed}") from error
        else:
            sample.parameters.update(requested_seed=seed, effective_seed=effective, seed_resample_index=retry)
            return sample


def assign_parent_tasks(items, config, epoch, *, smoke=False):
    view_policy = validate_view_policy(config["training"])
    if view_policy is not None:
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("parent sampling epoch must be a positive integer")
        for item in items:
            if item["sample"] not in authorized_ids(config, "train"):
                raise PermissionError("parent views require a training section")
            if tuple(item["hd16_phase_um"]) not in PHASES:
                raise ValueError("unknown HD16 phase")
        return ["HD16"] * len(items), view_sampling_report(items, view_policy, smoke=smoke)
    policy = config["training"]["parent_probabilities"]
    if (set(policy) != set(TASKS) or any(not math.isfinite(v) or v < 0 for v in policy.values())
            or not math.isclose(sum(policy.values()), 1.0, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("parent probabilities must explicitly sum to one over random, HD16 and Spot55")
    if type(epoch) is not int or epoch <= 0:
        raise ValueError("parent sampling epoch must be a positive integer")
    groups = defaultdict(list)
    for index, item in enumerate(items):
        if item["sample"] not in authorized_ids(config, "train"):
            raise PermissionError("parent fitting observations require a training section")
        groups[item["sample"]].append(index)
    tasks = ["random"] * len(items)
    for rid, indices in groups.items():
        order = np.random.default_rng(stable_seed(config["training"]["seed"], rid, epoch,
            "v030_parent_task_order")).permutation(indices)
        offset = 0
        for task in TASKS[1:]:
            quota = len(indices) * policy[task]
            count = math.floor(epoch * quota + .5 + 1e-9) - math.floor((epoch - 1) * quota + .5 + 1e-9)
            if offset + count > len(order):
                raise ValueError("rounded observation quotas exceed the section's field budget")
            for index in order[offset:offset + count]:
                tasks[index] = task
            offset += count
    if smoke:
        if len(tasks) < len(TASKS):
            raise ValueError("training smoke needs at least three fields to cover all observation layouts")
        tasks[:3] = TASKS
    return tasks, dict(probabilities=policy, fields={task: tasks.count(task) for task in TASKS},
                       smoke_layout_override=smoke)


def training_parent_masks(items, tasks, fields, config, *, device=None, cuda_batch_size=96):
    """Preserve presentation order while batching random CUDA geometry for an epoch.

    Explicit ``device="cpu"`` retains the reference generator for comparisons and
    CPU-only preparation. CUDA generation must run on the training thread before
    passing the resulting masks to the batch prefetcher.
    """
    fields._keys(items, "train")
    if len(items) != len(tasks) or set(tasks) - set(TASKS):
        raise ValueError("training fields and parent layouts differ")
    result, origins = [None] * len(items), {}
    settings = config["training"]
    device = torch.device(fields.device if device is None else device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("training parent masks require CPU or CUDA")
    generator = GeneratorConfig(**settings["generator"])
    random_indices = [index for index, task in enumerate(tasks) if task == "random"]
    if random_indices and device.type == "cuda":
        from astra.training.cuda_masks import generate_cuda_epoch_masks
        samples = generate_cuda_epoch_masks([items[index] for index in random_indices], generator,
            device=device, batch_size=cuda_batch_size)
        for index, sample in zip(random_indices, samples, strict=True):
            result[index] = sample
    lattice_seed = stable_seed(settings["seed"], "v030_training_spot55")
    hd16 = canonical_hd16_owner(device=device) if "HD16" in tasks else None
    phased = {(0, 0): hd16}
    for index, (item, task) in enumerate(zip(items, tasks, strict=True)):
        if task == "HD16":
            phase = tuple(item.get("hd16_phase_um", (0, 0)))
            if phase != (0, 0) and (settings.get("fov_parent_views") is None or phase not in PHASES):
                raise ValueError("phased HD16 observations require the configured view policy")
            if phase not in phased:
                phased[phase] = axis_aligned_square_owner(side_um=16., phase_um=phase, device=device)
            result[index] = phased[phase]
            continue
        if task == "random":
            if device.type == "cuda":
                continue
            sample = random_owner_with_resampling(item["mask_seed"], generator, item["coverage_stratum"])
        else:
            rid = item["sample"]
            if rid not in origins:
                origins[rid] = section_lattice_origin(rid, lattice_seed)
            sample = lattice_spot_owner(fields.starts[rid][item["field_index"]],
                                        lattice_origin_yx_um=origins[rid])
        result[index] = OwnerMapSample(sample.owner_map.to(device), sample.parent_valid.to(device), sample.parameters)
    return result
