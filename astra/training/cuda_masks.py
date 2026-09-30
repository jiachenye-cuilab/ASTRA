"""Self-contained batched random-parent geometry with the reference CPU RNG.

Copied only the mask-generation closure of the historical GPU pipeline. Sparse
expression loading and GPU-resident data stores are deliberately not included.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any

import torch

from astra.training.generator import GeneratorConfig, OwnerMapSample


@dataclass
class _MaskState:
    output_index: int
    requested_seed: int
    coverage_stratum: int
    effective_seed: int
    resample_index: int
    attempt: int
    rng: torch.Generator

    @classmethod
    def create(
        cls, output_index: int, requested_seed: int, coverage_stratum: int
    ) -> "_MaskState":
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(requested_seed))
        return cls(
            output_index=int(output_index),
            requested_seed=int(requested_seed),
            coverage_stratum=int(coverage_stratum),
            effective_seed=int(requested_seed),
            resample_index=0,
            attempt=0,
            rng=rng,
        )

    def advance_resample(self) -> None:
        self.resample_index += 1
        if self.resample_index >= 8:
            raise RuntimeError(
                f"random owner generator exhausted requested seed {self.requested_seed}"
            )
        self.effective_seed = int(
            hashlib.sha256(
                f"{self.requested_seed}:{self.resample_index}:v012_resample".encode()
            ).hexdigest()[:16],
            16,
        ) % (2**63 - 1)
        self.rng.manual_seed(self.effective_seed)
        self.attempt = 0


@dataclass(frozen=True)
class _Attempt:
    state: _MaskState
    target_coverage: float
    target_area: float
    regularity: float
    parents: int
    seeds: torch.Tensor
    power: torch.Tensor
    q: torch.Tensor
    angles: torch.Tensor
    half_y: torch.Tensor
    half_x: torch.Tensor
    kept: torch.Tensor
    dropout_probability: float


def _uniform(
    shape: tuple[int, ...] | tuple[()], *, rng: torch.Generator
) -> torch.Tensor:
    return torch.rand(shape, generator=rng, dtype=torch.float64)


def _log_uniform(
    lower: float, upper: float, shape: tuple[int, ...] | tuple[()], *, rng: torch.Generator
) -> torch.Tensor:
    unit = _uniform(shape, rng=rng)
    return torch.exp(math.log(float(lower)) + unit * math.log(float(upper) / float(lower)))


def _prepare_attempt(state: _MaskState, config: GeneratorConfig) -> _Attempt:
    state.attempt += 1
    rows, columns = config.field_shape
    cells = rows * columns
    lower, upper = config.coverage_strata[state.coverage_stratum]
    target_coverage = float(lower + (upper - lower) * _uniform((), rng=state.rng))
    target_area = float(
        _log_uniform(
            config.minimum_parent_area_children,
            config.maximum_parent_area_children,
            (),
            rng=state.rng,
        )
    )
    regularity = float(_uniform((), rng=state.rng))
    estimate = max(1, int(round(target_coverage * cells / target_area)))
    capacity = max(
        1,
        ((rows - 2) * (columns - 2)) // config.minimum_parent_area_children,
    )
    parents = min(config.maximum_parents, capacity, estimate)
    grid_rows = max(1, int(math.floor(math.sqrt(parents * rows / columns))))
    grid_columns = int(math.ceil(parents / grid_rows))
    while grid_rows * grid_columns < parents:
        grid_rows += 1
    grid_y = (torch.arange(grid_rows, dtype=torch.float64) + 0.5) * (
        rows / grid_rows
    )
    grid_x = (torch.arange(grid_columns, dtype=torch.float64) + 0.5) * (
        columns / grid_columns
    )
    lattice = torch.stack(
        torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=-1
    ).reshape(-1, 2)
    lattice = lattice[
        torch.randperm(lattice.shape[0], generator=state.rng)[:parents]
    ]
    random_seed = torch.stack(
        (
            1.5 + (rows - 3.0) * _uniform((parents,), rng=state.rng),
            1.5 + (columns - 3.0) * _uniform((parents,), rng=state.rng),
        ),
        dim=-1,
    )
    seeds = regularity * lattice + (1.0 - regularity) * random_seed
    spacing = math.sqrt(cells / parents)
    power = (
        _uniform((parents,), rng=state.rng) - 0.5
    ) * (1.0 - regularity) * 0.2 * spacing**2
    q = _log_uniform(config.q_range[0], config.q_range[1], (parents,), rng=state.rng)
    ratios = _log_uniform(
        config.axis_ratio_range[0],
        config.axis_ratio_range[1],
        (parents,),
        rng=state.rng,
    )
    angles = math.pi * _uniform((parents,), rng=state.rng)
    area_jitter = torch.exp(
        0.20 * torch.randn((parents,), generator=state.rng, dtype=torch.float64)
    ).clamp(0.70, 1.35)
    desired_area = (target_area * area_jitter).clamp(
        config.minimum_parent_area_children, config.maximum_parent_area_children
    )
    coefficient = (
        4.0
        * torch.exp(2.0 * torch.lgamma(1.0 + 1.0 / q))
        / torch.exp(torch.lgamma(1.0 + 2.0 / q))
    )
    half_y = torch.sqrt(desired_area * ratios / coefficient)
    half_x = torch.sqrt(desired_area / ratios / coefficient)
    dropout_probability = (
        min(0.08, max(0.0, (1.0 - target_coverage) * 0.10))
        if target_coverage < 0.80
        else 0.0
    )
    kept = _uniform((parents,), rng=state.rng) >= dropout_probability
    if not bool(kept.any()):
        kept[0] = True
    return _Attempt(
        state=state,
        target_coverage=target_coverage,
        target_area=target_area,
        regularity=regularity,
        parents=parents,
        seeds=seeds,
        power=power,
        q=q,
        angles=angles,
        half_y=half_y,
        half_x=half_x,
        kept=kept,
        dropout_probability=dropout_probability,
    )


def _connectivity_ok(
    owner: torch.Tensor, *, maximum_parent_area: int, maximum_parents: int
) -> torch.Tensor:
    """Exact 4-connectivity audit by same-owner label propagation on CUDA."""

    batch, rows, columns = owner.shape
    cells = rows * columns
    covered = owner >= 0
    labels = torch.arange(cells, dtype=torch.int32, device=owner.device).reshape(
        1, rows, columns
    ).expand(batch, -1, -1).clone()
    sentinel = int(cells)
    labels.masked_fill_(~covered, sentinel)
    same_y = covered[:, 1:, :] & covered[:, :-1, :] & (
        owner[:, 1:, :] == owner[:, :-1, :]
    )
    same_x = covered[:, :, 1:] & covered[:, :, :-1] & (
        owner[:, :, 1:] == owner[:, :, :-1]
    )
    for iteration in range(int(maximum_parent_area) + 1):
        updated = labels.clone()
        updated[:, 1:, :] = torch.minimum(
            updated[:, 1:, :],
            torch.where(same_y, labels[:, :-1, :], sentinel),
        )
        updated[:, :-1, :] = torch.minimum(
            updated[:, :-1, :],
            torch.where(same_y, labels[:, 1:, :], sentinel),
        )
        updated[:, :, 1:] = torch.minimum(
            updated[:, :, 1:],
            torch.where(same_x, labels[:, :, :-1], sentinel),
        )
        updated[:, :, :-1] = torch.minimum(
            updated[:, :, :-1],
            torch.where(same_x, labels[:, :, 1:], sentinel),
        )
        if ((iteration + 1) % 16 == 0 or iteration == maximum_parent_area) and torch.equal(updated, labels):
            labels = updated
            break
        labels = updated
    else:
        raise RuntimeError("v033 CUDA connectivity propagation did not converge")

    flat_owner = owner.reshape(batch, cells)
    flat_labels = labels.reshape(batch, cells)
    offsets = torch.arange(batch, device=owner.device)[:, None] * maximum_parents
    segments = offsets + flat_owner.clamp_min(0)
    minima = torch.full(
        (batch * maximum_parents,),
        sentinel,
        dtype=torch.int32,
        device=owner.device,
    )
    minima.scatter_reduce_(
        0,
        segments[covered.reshape(batch, cells)],
        flat_labels[covered.reshape(batch, cells)],
        reduce="amin",
        include_self=True,
    )
    expected = minima[segments]
    return ((~covered.reshape(batch, cells)) | (flat_labels == expected)).all(dim=1)


def _rasterize_attempts(
    attempts: list[_Attempt], config: GeneratorConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(attempts)
    rows, columns = config.field_shape
    cells = rows * columns
    maximum = max(item.parents for item in attempts)
    seeds = torch.zeros((batch, maximum, 2), dtype=torch.float64)
    power = torch.zeros((batch, maximum), dtype=torch.float64)
    q = torch.ones((batch, maximum), dtype=torch.float64)
    angles = torch.zeros((batch, maximum), dtype=torch.float64)
    half_y = torch.ones((batch, maximum), dtype=torch.float64)
    half_x = torch.ones((batch, maximum), dtype=torch.float64)
    kept = torch.zeros((batch, maximum), dtype=torch.bool)
    parent_valid = torch.zeros((batch, maximum), dtype=torch.bool)
    for index, item in enumerate(attempts):
        width = item.parents
        seeds[index, :width] = item.seeds
        power[index, :width] = item.power
        q[index, :width] = item.q
        angles[index, :width] = item.angles
        half_y[index, :width] = item.half_y
        half_x[index, :width] = item.half_x
        kept[index, :width] = item.kept
        parent_valid[index, :width] = True
    seeds, power, q, angles, half_y, half_x, kept, parent_valid = (
        value.to(device=device, non_blocking=False)
        for value in (seeds, power, q, angles, half_y, half_x, kept, parent_valid)
    )
    row_axis = torch.arange(rows, device=device, dtype=torch.float64) + 0.5
    column_axis = torch.arange(columns, device=device, dtype=torch.float64) + 0.5
    yy, xx = torch.meshgrid(row_axis, column_axis, indexing="ij")
    points = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
    interior = (
        (points[:, 0] > 1.0)
        & (points[:, 0] < rows - 1.0)
        & (points[:, 1] > 1.0)
        & (points[:, 1] < columns - 1.0)
    )
    distance = (
        (points[None, :, None, 0] - seeds[:, None, :, 0]).square()
        + (points[None, :, None, 1] - seeds[:, None, :, 1]).square()
        - power[:, None, :]
    )
    distance.masked_fill_(~parent_valid[:, None, :], torch.inf)
    candidate = distance.argmin(dim=2)

    def select(values: torch.Tensor) -> torch.Tensor:
        return torch.gather(values, 1, candidate)

    chosen_y = select(seeds[:, :, 0])
    chosen_x = select(seeds[:, :, 1])
    delta_y = points[None, :, 0] - chosen_y
    delta_x = points[None, :, 1] - chosen_x
    chosen_angle = select(angles)
    rotated_y = torch.cos(chosen_angle) * delta_y + torch.sin(chosen_angle) * delta_x
    rotated_x = -torch.sin(chosen_angle) * delta_y + torch.cos(chosen_angle) * delta_x
    chosen_q = select(q)
    support_score = (
        (rotated_y.abs() / select(half_y)).pow(chosen_q)
        + (rotated_x.abs() / select(half_x)).pow(chosen_q)
    )
    eligible = interior[None, :] & select(kept)
    required_scale = torch.exp(
        torch.log(support_score.clamp_min(torch.finfo(torch.float64).tiny)) / chosen_q
    )
    desired = torch.tensor(
        [
            min(
                int(round(item.target_coverage * cells)),
                int((rows - 2) * (columns - 2)),
            )
            for item in attempts
        ],
        dtype=torch.long,
        device=device,
    )
    enough = eligible.sum(dim=1) >= desired
    ordered = torch.sort(required_scale.masked_fill(~eligible, torch.inf), dim=1).values
    threshold = torch.gather(ordered, 1, (desired - 1).clamp_min(0)[:, None]).squeeze(1)
    return candidate, required_scale, eligible, enough, threshold


def _finish_attempts(
    attempts: list[_Attempt],
    candidate: torch.Tensor,
    required_scale: torch.Tensor,
    eligible: torch.Tensor,
    enough: torch.Tensor,
    threshold: torch.Tensor,
    config: GeneratorConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    device = candidate.device
    batch, cells = candidate.shape
    erosions = []
    enough_cpu = enough.cpu().tolist()
    for item, available in zip(attempts, enough_cpu):
        erosions.append(
            float(0.97 + 0.03 * _uniform((), rng=item.state.rng))
            if available
            else 1.0
        )
    scale = threshold * torch.tensor(erosions, dtype=torch.float64, device=device)
    owner = torch.where(
        eligible & (required_scale <= scale[:, None]),
        candidate,
        torch.full_like(candidate, -1),
    )
    maximum = config.maximum_parents
    covered = owner >= 0
    offsets = torch.arange(batch, device=device)[:, None] * maximum
    segments = offsets + owner.clamp_min(0)
    counts = torch.zeros(batch * maximum, dtype=torch.int64, device=device)
    counts.scatter_add_(
        0,
        segments[covered],
        torch.ones_like(segments[covered], dtype=torch.int64),
    )
    counts = counts.reshape(batch, maximum)
    valid_area = (counts >= config.minimum_parent_area_children) & (
        counts <= config.maximum_parent_area_children
    )
    keep_cell = covered & torch.gather(valid_area, 1, owner.clamp_min(0))
    owner = torch.where(keep_cell, owner, torch.full_like(owner, -1))
    lookup = torch.cumsum(valid_area.to(dtype=torch.long), dim=1) - 1
    compressed = torch.gather(lookup, 1, owner.clamp_min(0))
    owner = torch.where(owner >= 0, compressed, torch.full_like(compressed, -1))
    owner = owner.reshape(batch, *config.field_shape)
    parent_count = valid_area.sum(dim=1)
    actual_coverage = (owner >= 0).to(dtype=torch.float64).mean(dim=(1, 2))
    requested_coverage = torch.tensor(
        [item.target_coverage for item in attempts], dtype=torch.float64, device=device
    )
    coverage_ok = (
        (actual_coverage - requested_coverage).abs() <= config.coverage_tolerance
    ) & (parent_count > 0)
    connectivity = _connectivity_ok(
        owner,
        maximum_parent_area=config.maximum_parent_area_children,
        maximum_parents=config.maximum_parents,
    )
    accepted = enough & coverage_ok & connectivity
    return owner, parent_count, actual_coverage, accepted, erosions


def _generate_masks(
    items: list[dict[str, Any]],
    generator_config: GeneratorConfig,
    *,
    device: torch.device,
    batch_size: int = 96,
) -> list[OwnerMapSample]:
    """Generate an ordered epoch of masks with CPU RNG and batched CUDA geometry."""

    config = generator_config.validate()
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or type(batch_size) is not int or batch_size <= 0 or not items:
        raise ValueError("v033 CUDA generator boundary differs")
    if any(type(item["mask_seed"]) is not int or type(item["coverage_stratum"]) is not int
           or not 0 <= item["coverage_stratum"] < len(config.coverage_strata) for item in items):
        raise ValueError("mask seeds and coverage strata must be explicit valid integers")
    result: list[OwnerMapSample | None] = [None] * len(items)
    states = [
        _MaskState.create(
            index,
            int(item["mask_seed"]),
            int(item["coverage_stratum"]),
        )
        for index, item in enumerate(items)
    ]
    for chunk_start in range(0, len(states), int(batch_size)):
        active = states[chunk_start : chunk_start + int(batch_size)]
        while active:
            for state in active:
                if state.attempt >= config.maximum_attempts:
                    state.advance_resample()
            attempts = [_prepare_attempt(state, config) for state in active]
            candidate, required, eligible, enough, threshold = _rasterize_attempts(
                attempts, config, device
            )
            owner, parent_count, coverage, accepted, erosions = _finish_attempts(
                attempts, candidate, required, eligible, enough, threshold, config
            )
            accepted_cpu = accepted.cpu().tolist()
            parent_cpu = parent_count.cpu().tolist()
            coverage_cpu = coverage.cpu().tolist()
            scale_cpu = (
                threshold
                * torch.tensor(erosions, dtype=torch.float64, device=device)
            ).cpu().tolist()
            remaining: list[_MaskState] = []
            for index, (attempt, passed) in enumerate(zip(attempts, accepted_cpu)):
                state = attempt.state
                if not passed:
                    remaining.append(state)
                    continue
                parents = int(parent_cpu[index])
                result[state.output_index] = OwnerMapSample(
                    owner_map=owner[index],
                    parent_valid=torch.ones(parents, dtype=torch.bool, device=device),
                    parameters={
                        "program": "power_lq_exclusive_v1",
                        "seed": state.effective_seed,
                        "attempt": state.attempt,
                        "coverage_stratum": state.coverage_stratum,
                        "target_coverage": attempt.target_coverage,
                        "actual_coverage": float(coverage_cpu[index]),
                        "target_parent_area_children": attempt.target_area,
                        "parents": parents,
                        "regularity": attempt.regularity,
                        "support_scale": float(scale_cpu[index]),
                        "erosion_multiplier": float(erosions[index]),
                        "dropout_probability": attempt.dropout_probability,
                        "config": asdict(config),
                        "requested_seed": state.requested_seed,
                        "effective_seed": state.effective_seed,
                        "seed_resample_index": state.resample_index,
                    },
                )
            active = remaining
    if any(value is None for value in result):
        raise RuntimeError("v033 CUDA generator left an empty result")
    return [value for value in result if value is not None]


def generate_cuda_epoch_masks(items, generator_config, *, device, batch_size=96):
    """Generate ordered masks on one CUDA device; CPU RNG preserves seed semantics."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CUDA epoch masks require a CUDA device; use the CPU reference otherwise")
    return _generate_masks(items, generator_config, device=device, batch_size=batch_size)


__all__ = ["generate_cuda_epoch_masks"]
