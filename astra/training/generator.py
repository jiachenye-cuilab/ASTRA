"""Target-free exclusive connected-parent observation programs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch

from astra.training.ownership import is_single_4_connected, masks_to_owner, parent_child_counts


@dataclass(frozen=True)
class GeneratorConfig:
    field_shape: tuple[int, int] = (128, 128)
    cell_um: float = 2.0
    minimum_parent_area_children: int = 64
    maximum_parent_area_children: int = 768
    maximum_parents: int = 256
    q_range: tuple[float, float] = (2.0, 16.0)
    axis_ratio_range: tuple[float, float] = (0.75, 1.33)
    coverage_strata: tuple[tuple[float, float], ...] = (
        (0.15, 0.45),
        (0.45, 0.80),
        (0.80, 1.00),
    )
    coverage_tolerance: float = 0.05
    maximum_attempts: int = 32

    def validate(self) -> "GeneratorConfig":
        rows, columns = (int(value) for value in self.field_shape)
        if (
            min(rows, columns) < 8
            or self.cell_um <= 0
            or self.minimum_parent_area_children <= 0
            or self.maximum_parent_area_children < self.minimum_parent_area_children
            or self.maximum_parent_area_children > rows * columns
            or self.maximum_parents <= 0
            or self.maximum_parents > rows * columns
            or not 1.0 <= self.q_range[0] <= self.q_range[1]
            or not 0 < self.axis_ratio_range[0] <= self.axis_ratio_range[1]
            or self.coverage_tolerance <= 0
            or self.maximum_attempts <= 0
        ):
            raise ValueError("v033 generator configuration differs")
        previous = 0.0
        for lower, upper in self.coverage_strata:
            if not 0.0 <= lower < upper <= 1.0 or lower < previous:
                raise ValueError("v033 coverage strata differ")
            previous = upper
        return self


@dataclass(frozen=True)
class OwnerMapSample:
    owner_map: torch.Tensor
    parent_valid: torch.Tensor
    parameters: dict[str, Any]

    @property
    def parents(self) -> int:
        return int(self.parent_valid.sum())

    @property
    def coverage(self) -> float:
        return float((self.owner_map >= 0).to(dtype=torch.float64).mean())


def _uniform(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device=device, dtype=dtype)


def _log_uniform(
    lower: float,
    upper: float,
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    unit = _uniform(shape, generator=generator, device=device, dtype=torch.float64)
    return torch.exp(math.log(float(lower)) + unit * math.log(float(upper) / float(lower)))


def _compress_owner(owner: torch.Tensor, parents: int) -> tuple[torch.Tensor, torch.Tensor]:
    counts = parent_child_counts(owner, parents)[0]
    keep = counts > 0
    old_ids = torch.nonzero(keep, as_tuple=False).flatten()
    if old_ids.numel() == 0:
        raise ValueError("v033 generated no non-empty parent")
    lookup = torch.full((parents,), -1, dtype=torch.long, device=owner.device)
    lookup[old_ids] = torch.arange(old_ids.numel(), dtype=torch.long, device=owner.device)
    result = torch.where(owner >= 0, lookup[owner.clamp_min(0)], owner)
    return result, torch.ones(old_ids.numel(), dtype=torch.bool, device=owner.device)


def _connectivity_ok(owner: torch.Tensor, parents: int) -> bool:
    cpu = owner.detach().cpu()
    return all(is_single_4_connected(cpu == index) for index in range(parents))


def generate_random_owner(
    seed: int,
    *,
    config: GeneratorConfig | None = None,
    coverage_stratum: int | None = None,
    device: torch.device | str = "cpu",
    audit_connectivity: bool = True,
) -> OwnerMapSample:
    """Generate one target/H&E-free owner map from a single continuous program.

    A power diagram first gives each child at most one candidate owner. Each
    candidate cell is intersected with a rotated Lq support. A scalar support
    threshold and light parent dropout control coverage without ever creating
    overlapping parents.
    """

    cfg = (config or GeneratorConfig()).validate()
    target_device = torch.device(device)
    rng = torch.Generator(device=target_device)
    rng.manual_seed(int(seed))
    rows, columns = cfg.field_shape
    cells = rows * columns
    if coverage_stratum is None:
        stratum = int(
            torch.randint(
                len(cfg.coverage_strata),
                (),
                generator=rng,
                device=target_device,
            )
        )
    else:
        stratum = int(coverage_stratum)
    if stratum < 0 or stratum >= len(cfg.coverage_strata):
        raise ValueError("v033 coverage stratum index differs")
    coverage_lower, coverage_upper = cfg.coverage_strata[stratum]

    row_axis = torch.arange(rows, device=target_device, dtype=torch.float64) + 0.5
    column_axis = torch.arange(columns, device=target_device, dtype=torch.float64) + 0.5
    yy, xx = torch.meshgrid(row_axis, column_axis, indexing="ij")
    points = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
    # Random supports do not touch the FOV boundary; aligned fixed evaluation
    # layouts have their own exact rasterizers below.
    interior = (
        (points[:, 0] > 1.0)
        & (points[:, 0] < rows - 1.0)
        & (points[:, 1] > 1.0)
        & (points[:, 1] < columns - 1.0)
    )

    for attempt in range(cfg.maximum_attempts):
        target_coverage = float(
            coverage_lower
            + (coverage_upper - coverage_lower)
            * _uniform((), generator=rng, device=target_device, dtype=torch.float64)
        )
        target_area = float(
            _log_uniform(
                cfg.minimum_parent_area_children,
                cfg.maximum_parent_area_children,
                (),
                generator=rng,
                device=target_device,
            )
        )
        regularity = float(
            _uniform((), generator=rng, device=target_device, dtype=torch.float64)
        )
        estimate = max(1, int(round(target_coverage * cells / target_area)))
        capacity = max(1, int(interior.sum()) // cfg.minimum_parent_area_children)
        parents = min(cfg.maximum_parents, capacity, estimate)
        grid_rows = max(1, int(math.floor(math.sqrt(parents * rows / columns))))
        grid_columns = int(math.ceil(parents / grid_rows))
        while grid_rows * grid_columns < parents:
            grid_rows += 1
        grid_y = (torch.arange(grid_rows, device=target_device, dtype=torch.float64) + 0.5) * (
            rows / grid_rows
        )
        grid_x = (
            torch.arange(grid_columns, device=target_device, dtype=torch.float64) + 0.5
        ) * (columns / grid_columns)
        lattice = torch.stack(torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=-1).reshape(-1, 2)
        permutation = torch.randperm(
            lattice.shape[0], generator=rng, device=target_device
        )[:parents]
        lattice = lattice[permutation]
        random_seed = torch.stack(
            (
                1.5
                + (rows - 3.0)
                * _uniform(
                    (parents,), generator=rng, device=target_device, dtype=torch.float64
                ),
                1.5
                + (columns - 3.0)
                * _uniform(
                    (parents,), generator=rng, device=target_device, dtype=torch.float64
                ),
            ),
            dim=-1,
        )
        seeds = regularity * lattice + (1.0 - regularity) * random_seed
        spacing = math.sqrt(cells / parents)
        power = (
            _uniform(
                (parents,), generator=rng, device=target_device, dtype=torch.float64
            )
            - 0.5
        ) * (1.0 - regularity) * 0.2 * spacing**2
        distance = (
            (points[:, None, 0] - seeds[None, :, 0]).square()
            + (points[:, None, 1] - seeds[None, :, 1]).square()
            - power[None]
        )
        candidate = distance.argmin(dim=1)

        q = _log_uniform(
            cfg.q_range[0],
            cfg.q_range[1],
            (parents,),
            generator=rng,
            device=target_device,
        )
        ratios = _log_uniform(
            cfg.axis_ratio_range[0],
            cfg.axis_ratio_range[1],
            (parents,),
            generator=rng,
            device=target_device,
        )
        angles = math.pi * _uniform(
            (parents,), generator=rng, device=target_device, dtype=torch.float64
        )
        area_jitter = torch.exp(
            0.20
            * torch.randn(
                (parents,), generator=rng, device=target_device, dtype=torch.float64
            )
        ).clamp(0.70, 1.35)
        desired_area = (target_area * area_jitter).clamp(
            cfg.minimum_parent_area_children, cfg.maximum_parent_area_children
        )
        area_coefficient = (
            4.0
            * torch.exp(2.0 * torch.lgamma(1.0 + 1.0 / q))
            / torch.exp(torch.lgamma(1.0 + 2.0 / q))
        )
        half_y = torch.sqrt(desired_area * ratios / area_coefficient)
        half_x = torch.sqrt(desired_area / ratios / area_coefficient)
        chosen_seed = seeds[candidate]
        delta_y = points[:, 0] - chosen_seed[:, 0]
        delta_x = points[:, 1] - chosen_seed[:, 1]
        chosen_angle = angles[candidate]
        rotated_y = torch.cos(chosen_angle) * delta_y + torch.sin(chosen_angle) * delta_x
        rotated_x = -torch.sin(chosen_angle) * delta_y + torch.cos(chosen_angle) * delta_x
        chosen_q = q[candidate]
        support_score = (
            (rotated_y.abs() / half_y[candidate]).pow(chosen_q)
            + (rotated_x.abs() / half_x[candidate]).pow(chosen_q)
        )
        dropout_probability = (
            min(0.08, max(0.0, (1.0 - target_coverage) * 0.10))
            if target_coverage < 0.80
            else 0.0
        )
        kept = _uniform(
            (parents,), generator=rng, device=target_device, dtype=torch.float64
        ) >= dropout_probability
        if not bool(kept.any()):
            kept[0] = True
        eligible = interior & kept[candidate]
        # One order statistic replaces a 24-iteration host-synchronized search.
        # It is the exact support multiplier at which the desired number of
        # eligible children enters the Lq supports.
        required_scale = torch.exp(
            torch.log(support_score.clamp_min(torch.finfo(support_score.dtype).tiny))
            / chosen_q
        )
        desired_children = min(
            int(round(target_coverage * cells)),
            int((rows - 2) * (columns - 2)),
        )
        eligible_scale = required_scale[eligible]
        if eligible_scale.numel() < desired_children:
            continue
        scale_tensor = eligible_scale.kthvalue(desired_children).values
        erosion = float(
            0.97
            + 0.03
            * _uniform((), generator=rng, device=target_device, dtype=torch.float64)
        )
        scale_tensor = scale_tensor * erosion
        owned = eligible & (required_scale <= scale_tensor)
        owner = torch.where(owned, candidate, torch.full_like(candidate, -1)).reshape(
            rows, columns
        )
        areas = parent_child_counts(owner, parents)[0]
        valid_area = (areas >= cfg.minimum_parent_area_children) & (
            areas <= cfg.maximum_parent_area_children
        )
        owner = torch.where(
            (owner >= 0) & valid_area[owner.clamp_min(0)], owner, torch.full_like(owner, -1)
        )
        try:
            owner, parent_valid = _compress_owner(owner, parents)
        except ValueError:
            continue
        actual_coverage = float((owner >= 0).double().mean())
        if abs(actual_coverage - target_coverage) > cfg.coverage_tolerance:
            continue
        if audit_connectivity and not _connectivity_ok(owner, int(parent_valid.numel())):
            continue
        return OwnerMapSample(
            owner_map=owner,
            parent_valid=parent_valid,
            parameters={
                "program": "power_lq_exclusive_v1",
                "seed": int(seed),
                "attempt": attempt + 1,
                "coverage_stratum": stratum,
                "target_coverage": target_coverage,
                "actual_coverage": actual_coverage,
                "target_parent_area_children": target_area,
                "parents": int(parent_valid.numel()),
                "regularity": regularity,
                "support_scale": float(scale_tensor),
                "erosion_multiplier": erosion,
                "dropout_probability": dropout_probability,
                "config": asdict(cfg),
            },
        )
    raise RuntimeError(
        f"v033 generator failed after {cfg.maximum_attempts} attempts for seed {seed}"
    )


def axis_aligned_square_owner(
    *,
    side_um: float,
    phase_um: tuple[float, float] = (0.0, 0.0),
    field_shape: tuple[int, int] = (128, 128),
    cell_um: float = 2.0,
    device: torch.device | str = "cpu",
) -> OwnerMapSample:
    """Rasterize all full, non-overlapping phased square parents."""

    rows, columns = (int(value) for value in field_shape)
    side = float(side_um)
    phase_y, phase_x = (float(value) for value in phase_um)
    if side <= 0 or cell_um <= 0 or not 0 <= phase_y < side or not 0 <= phase_x < side:
        raise ValueError("v033 square side/phase differs")
    fov_y, fov_x = rows * cell_um, columns * cell_um
    starts_y = torch.arange(phase_y, fov_y - side + 1e-12, side, dtype=torch.float64)
    starts_x = torch.arange(phase_x, fov_x - side + 1e-12, side, dtype=torch.float64)
    if starts_y.numel() == 0 or starts_x.numel() == 0:
        raise ValueError("v033 phased square program has no full parent")
    yy = (torch.arange(rows, dtype=torch.float64) + 0.5) * cell_um
    xx = (torch.arange(columns, dtype=torch.float64) + 0.5) * cell_um
    masks = []
    for start_y in starts_y:
        for start_x in starts_x:
            masks.append(
                ((yy[:, None] >= start_y) & (yy[:, None] < start_y + side))
                & ((xx[None, :] >= start_x) & (xx[None, :] < start_x + side))
            )
    mask_tensor = torch.stack(masks).to(device=device)
    owner, valid = masks_to_owner(mask_tensor, check_connectivity=True)
    return OwnerMapSample(
        owner_map=owner,
        parent_valid=valid,
        parameters={
            "program": "axis_aligned_square",
            "side_um": side,
            "phase_um": [phase_y, phase_x],
            "parents": int(valid.sum()),
            "actual_coverage": float((owner >= 0).double().mean()),
        },
    )


def canonical_hd16_owner(
    *,
    field_shape: tuple[int, int] = (128, 128),
    cell_um: float = 2.0,
    device: torch.device | str = "cpu",
) -> OwnerMapSample:
    return axis_aligned_square_owner(
        side_um=16.0,
        phase_um=(0.0, 0.0),
        field_shape=field_shape,
        cell_um=cell_um,
        device=device,
    )


def sparse_spot55_owner(
    *,
    seed: int = 0,
    diameter_um: float = 55.0,
    pitch_um: float = 100.0,
    translation_um: tuple[float, float] = (0.0, 0.0),
    orientation_radians: float = 0.0,
    dropout_probability: float = 0.0,
    field_shape: tuple[int, int] = (128, 128),
    cell_um: float = 2.0,
    device: torch.device | str = "cpu",
) -> OwnerMapSample:
    """Seven full, disjoint center-in-disc spot masks for fixed evaluation."""

    rows, columns = (int(value) for value in field_shape)
    fov_y, fov_x = rows * cell_um, columns * cell_um
    diameter, pitch = float(diameter_um), float(pitch_um)
    translate_y, translate_x = (float(value) for value in translation_um)
    if (
        diameter <= 0
        or pitch <= diameter
        or not 0 <= dropout_probability < 1
        or cell_um <= 0
    ):
        raise ValueError("v033 sparse spot geometry differs")
    centers = [(translate_y, translate_x)]
    for index in range(6):
        angle = float(orientation_radians) + index * math.pi / 3.0
        centers.append(
            (translate_y + pitch * math.sin(angle), translate_x + pitch * math.cos(angle))
        )
    centers_tensor = torch.tensor(centers, dtype=torch.float64, device=device)
    radius = diameter / 2.0
    if bool(
        torch.any(centers_tensor[:, 0].abs() + radius > fov_y / 2.0 + 1e-12)
        or torch.any(centers_tensor[:, 1].abs() + radius > fov_x / 2.0 + 1e-12)
    ):
        raise ValueError("v033 sparse spot support is clipped by the FOV")
    yy = (torch.arange(rows, device=device, dtype=torch.float64) + 0.5) * cell_um - fov_y / 2.0
    xx = (torch.arange(columns, device=device, dtype=torch.float64) + 0.5) * cell_um - fov_x / 2.0
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    masks = (
        (grid_y[None] - centers_tensor[:, 0, None, None]).square()
        + (grid_x[None] - centers_tensor[:, 1, None, None]).square()
        <= radius**2
    )
    rng = torch.Generator(device=torch.device(device))
    rng.manual_seed(int(seed))
    keep = _uniform(
        (7,), generator=rng, device=torch.device(device), dtype=torch.float64
    ) >= dropout_probability
    if not bool(keep.any()):
        keep[0] = True
    masks = masks[keep]
    owner, valid = masks_to_owner(masks, check_connectivity=True)
    return OwnerMapSample(
        owner_map=owner,
        parent_valid=valid,
        parameters={
            "program": "spot55_seven_disc",
            "seed": int(seed),
            "diameter_um": diameter,
            "pitch_um": pitch,
            "translation_um": [translate_y, translate_x],
            "orientation_radians": float(orientation_radians),
            "dropout_probability": float(dropout_probability),
            "parents": int(valid.sum()),
            "actual_coverage": float((owner >= 0).double().mean()),
        },
    )


__all__ = [
    "GeneratorConfig",
    "OwnerMapSample",
    "axis_aligned_square_owner",
    "canonical_hd16_owner",
    "generate_random_owner",
    "sparse_spot55_owner",
]
