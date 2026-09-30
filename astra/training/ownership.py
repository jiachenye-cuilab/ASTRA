"""Binary exclusive-ownership semantics on the native 2 µm grid."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import torch


STRATUM_INVALID = 0
STRATUM_SINGLE_PARENT = 1
STRATUM_MULTI_PARENT = 2
STRATUM_BOUNDARY = 3
STRATUM_GAP_ONLY = 4


def _spatial_owner(owner_map: torch.Tensor) -> torch.Tensor:
    owner = torch.as_tensor(owner_map)
    if owner.ndim == 2:
        owner = owner.unsqueeze(0)
    if owner.ndim != 3 or owner.dtype != torch.long:
        raise TypeError("v033 owner_map must be int64 [batch,rows,columns]")
    return owner


def _spatial_valid(field_valid: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    valid = torch.as_tensor(field_valid)
    if valid.ndim == 2:
        valid = valid.unsqueeze(0)
    if valid.shape != shape or valid.dtype != torch.bool:
        raise TypeError("v033 field_valid must be bool and match owner_map")
    return valid


def _segment_ids(owner: torch.Tensor, parents: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch = owner.shape[0]
    flat = owner.reshape(batch, -1)
    covered = flat >= 0
    offsets = torch.arange(batch, device=owner.device, dtype=torch.long)[:, None] * parents
    return torch.where(covered, flat + offsets, torch.zeros_like(flat)), covered


def parent_child_counts(owner_map: torch.Tensor, parents: int) -> torch.Tensor:
    owner = _spatial_owner(owner_map)
    if parents <= 0:
        raise ValueError("v033 requires at least one padded parent slot")
    segments, covered = _segment_ids(owner, int(parents))
    result = torch.zeros(
        owner.shape[0] * int(parents), dtype=torch.int64, device=owner.device
    )
    result.scatter_add_(
        0,
        segments[covered],
        torch.ones_like(segments[covered], dtype=torch.int64),
    )
    return result.reshape(owner.shape[0], int(parents))


def is_single_4_connected(mask: torch.Tensor) -> bool:
    """CPU audit for one non-empty binary parent mask."""

    value = torch.as_tensor(mask, dtype=torch.bool, device="cpu")
    if value.ndim != 2:
        return False
    rows, columns = value.shape
    # Traversal is scalar CPU work. A private NumPy bitmap avoids a torch
    # dispatch for each neighbour and leaves the caller's mask untouched.
    remaining = value.numpy().copy().reshape(-1)
    total = int(np.count_nonzero(remaining))
    if not total:
        return False
    start = int(remaining.argmax())
    queue: deque[int] = deque([start])
    remaining[start] = False
    visited = 1
    while queue:
        row, column = divmod(queue.popleft(), columns)
        for next_row, next_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if 0 <= next_row < rows and 0 <= next_column < columns:
                index = next_row * columns + next_column
                if remaining[index]:
                    remaining[index] = False
                    visited += 1
                    queue.append(index)
    return visited == total


def validate_owner_batch(
    owner_map: torch.Tensor,
    parent_valid: torch.Tensor,
    field_valid: torch.Tensor,
    *,
    check_connectivity: bool = False,
) -> None:
    """Reject overlap-equivalent ids, invalid ownership and optional islands."""

    owner = _spatial_owner(owner_map)
    valid_field = _spatial_valid(field_valid, tuple(owner.shape))
    valid_parent = torch.as_tensor(parent_valid)
    if valid_parent.ndim == 1:
        valid_parent = valid_parent.unsqueeze(0)
    if (
        valid_parent.ndim != 2
        or valid_parent.shape[0] != owner.shape[0]
        or valid_parent.dtype != torch.bool
        or valid_parent.device != owner.device
        or valid_field.device != owner.device
    ):
        raise TypeError("v033 parent_valid must be bool [batch,parents] on one device")
    parents = valid_parent.shape[1]
    if bool(torch.any(owner < -1)) or bool(torch.any(owner >= parents)):
        raise ValueError("v033 owner id lies outside {-1, ..., parents-1}")
    if bool(torch.any((owner >= 0) & ~valid_field)):
        raise ValueError("v033 invalid field cell cannot have an owner")
    counts = parent_child_counts(owner, parents)
    if bool(torch.any(valid_parent & (counts == 0))):
        raise ValueError("v033 valid parent must own at least one child")
    if bool(torch.any(~valid_parent & (counts != 0))):
        raise ValueError("v033 invalid padded parent owns children")
    if check_connectivity:
        owner_cpu = owner.detach().cpu()
        valid_cpu = valid_parent.detach().cpu()
        for batch_index in range(owner.shape[0]):
            for parent_index in torch.nonzero(valid_cpu[batch_index], as_tuple=False).flatten():
                if not is_single_4_connected(
                    owner_cpu[batch_index] == int(parent_index)
                ):
                    raise ValueError("v033 parent mask is not one 4-connected region")


def masks_to_owner(
    masks: torch.Tensor,
    *,
    field_valid: torch.Tensor | None = None,
    check_connectivity: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert external binary masks to canonical owner ids, rejecting overlap."""

    value = torch.as_tensor(masks)
    squeeze = value.ndim == 3
    if squeeze:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.dtype != torch.bool:
        raise TypeError("v033 masks must be bool [batch,parents,rows,columns]")
    batch, parents, rows, columns = value.shape
    valid_field = (
        torch.ones((batch, rows, columns), dtype=torch.bool, device=value.device)
        if field_valid is None
        else _spatial_valid(field_valid, (batch, rows, columns))
    )
    if bool(torch.any(value.sum(dim=1) > 1)):
        raise ValueError("v033 parent masks overlap")
    if bool(torch.any(value & ~valid_field[:, None])):
        raise ValueError("v033 parent mask includes invalid field cells")
    parent_valid = value.flatten(2).any(dim=-1)
    ids = torch.arange(parents, dtype=torch.long, device=value.device)
    owner = torch.where(
        value,
        ids[None, :, None, None],
        torch.full((), -1, dtype=torch.long, device=value.device),
    ).amax(dim=1)
    validate_owner_batch(
        owner,
        parent_valid,
        valid_field,
        check_connectivity=check_connectivity,
    )
    return (owner[0], parent_valid[0]) if squeeze else (owner, parent_valid)


def aggregate_parent_counts(
    target_count_2um: torch.Tensor,
    owner_map: torch.Tensor,
    parent_valid: torch.Tensor,
) -> torch.Tensor:
    """Direct integer segment sum from native 2 µm UMI to parents."""

    owner = _spatial_owner(owner_map)
    target = torch.as_tensor(target_count_2um, device=owner.device)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if target.ndim != 4 or target.shape[:3] != owner.shape:
        raise ValueError("v033 target must be [batch,rows,columns,genes]")
    if target.dtype == torch.bool or target.is_floating_point():
        raise TypeError("v033 synthetic parent counts require integer native UMI")
    if bool(torch.any(target < 0)):
        raise ValueError("v033 native UMI must be nonnegative")
    valid_parent = torch.as_tensor(parent_valid, dtype=torch.bool, device=owner.device)
    if valid_parent.ndim == 1:
        valid_parent = valid_parent.unsqueeze(0)
    if valid_parent.shape[0] != owner.shape[0]:
        raise ValueError("v033 parent_valid batch differs")
    parents = valid_parent.shape[1]
    segments, covered = _segment_ids(owner, parents)
    genes = target.shape[-1]
    result = torch.zeros(
        (owner.shape[0] * parents, genes), dtype=torch.int64, device=owner.device
    )
    flat_target = target.to(dtype=torch.int64).reshape(owner.shape[0], -1, genes)
    covered_segments = segments[covered]
    result.scatter_add_(
        0,
        covered_segments[:, None].expand(-1, genes),
        flat_target[covered],
    )
    result = result.reshape(owner.shape[0], parents, genes)
    result = torch.where(valid_parent[..., None], result, torch.zeros_like(result))
    return result[0] if target_count_2um.ndim == 3 else result


@dataclass(frozen=True)
class ParentGeometry:
    """Mask-derived geometry; the owner map remains the sole authority."""

    features: torch.Tensor
    centers_um: torch.Tensor
    sizes_um: torch.Tensor
    area_children: torch.Tensor
    perimeter_edges: torch.Tensor


def derive_parent_geometry(
    owner_map: torch.Tensor,
    parent_valid: torch.Tensor,
    *,
    cell_um: float = 2.0,
) -> ParentGeometry:
    owner = _spatial_owner(owner_map)
    valid_parent = torch.as_tensor(parent_valid, dtype=torch.bool, device=owner.device)
    if valid_parent.ndim == 1:
        valid_parent = valid_parent.unsqueeze(0)
    batch, rows, columns = owner.shape
    if cell_um <= 0 or valid_parent.shape[0] != batch:
        raise ValueError("v033 geometry dimensions differ")
    parents = valid_parent.shape[1]
    segments, covered = _segment_ids(owner, parents)
    flat_segments = segments[covered]
    counts = parent_child_counts(owner, parents)
    dtype = torch.float32
    row_axis = torch.arange(rows, device=owner.device, dtype=dtype)
    column_axis = torch.arange(columns, device=owner.device, dtype=dtype)
    yy, xx = torch.meshgrid(row_axis, column_axis, indexing="ij")
    yy = yy[None].expand(batch, -1, -1).reshape(batch, -1)
    xx = xx[None].expand(batch, -1, -1).reshape(batch, -1)
    size = batch * parents

    def scatter_sum(values: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(size, dtype=dtype, device=owner.device)
        out.scatter_add_(0, flat_segments, values[covered])
        return out.reshape(batch, parents)

    count_float = counts.to(dtype=dtype).clamp_min(1.0)
    center_row_cell = scatter_sum(yy) / count_float
    center_column_cell = scatter_sum(xx) / count_float

    minimum_row = torch.full((size,), float(rows), dtype=dtype, device=owner.device)
    maximum_row = torch.full((size,), -1.0, dtype=dtype, device=owner.device)
    minimum_column = torch.full((size,), float(columns), dtype=dtype, device=owner.device)
    maximum_column = torch.full((size,), -1.0, dtype=dtype, device=owner.device)
    minimum_row.scatter_reduce_(0, flat_segments, yy[covered], reduce="amin", include_self=True)
    maximum_row.scatter_reduce_(0, flat_segments, yy[covered], reduce="amax", include_self=True)
    minimum_column.scatter_reduce_(0, flat_segments, xx[covered], reduce="amin", include_self=True)
    maximum_column.scatter_reduce_(0, flat_segments, xx[covered], reduce="amax", include_self=True)
    height = (maximum_row - minimum_row + 1.0).reshape(batch, parents).clamp_min(1.0)
    width = (maximum_column - minimum_column + 1.0).reshape(batch, parents).clamp_min(1.0)

    boundary_edges = torch.zeros_like(owner, dtype=torch.int64)
    boundary_edges[:, 0] += owner[:, 0] >= 0
    boundary_edges[:, -1] += owner[:, -1] >= 0
    boundary_edges[:, :, 0] += owner[:, :, 0] >= 0
    boundary_edges[:, :, -1] += owner[:, :, -1] >= 0
    vertical = owner[:, 1:] != owner[:, :-1]
    boundary_edges[:, 1:] += vertical & (owner[:, 1:] >= 0)
    boundary_edges[:, :-1] += vertical & (owner[:, :-1] >= 0)
    horizontal = owner[:, :, 1:] != owner[:, :, :-1]
    boundary_edges[:, :, 1:] += horizontal & (owner[:, :, 1:] >= 0)
    boundary_edges[:, :, :-1] += horizontal & (owner[:, :, :-1] >= 0)
    perimeter = scatter_sum(boundary_edges.reshape(batch, -1).to(dtype=dtype))
    compactness = 4.0 * math.pi * count_float / perimeter.clamp_min(1.0).square()
    fill_ratio = count_float / (height * width)

    fov_y_um = rows * float(cell_um)
    fov_x_um = columns * float(cell_um)
    centers_um = torch.stack(
        (
            (center_row_cell + 0.5) * float(cell_um) - fov_y_um / 2.0,
            (center_column_cell + 0.5) * float(cell_um) - fov_x_um / 2.0,
        ),
        dim=-1,
    )
    sizes_um = torch.stack((height, width), dim=-1) * float(cell_um)
    features = torch.stack(
        (
            centers_um[..., 0] / (fov_y_um / 2.0),
            centers_um[..., 1] / (fov_x_um / 2.0),
            torch.log(count_float),
            torch.log(height),
            torch.log(width),
            compactness,
            fill_ratio,
        ),
        dim=-1,
    )
    mask = valid_parent[..., None]
    return ParentGeometry(
        features=torch.where(mask, features, torch.zeros_like(features)),
        centers_um=torch.where(mask, centers_um, torch.zeros_like(centers_um)),
        sizes_um=torch.where(mask, sizes_um, torch.zeros_like(sizes_um)),
        area_children=torch.where(valid_parent, counts, torch.zeros_like(counts)),
        perimeter_edges=torch.where(
            valid_parent, perimeter.to(dtype=torch.int64), torch.zeros_like(counts)
        ),
    )


def classify_8um_ownership(
    owner_map: torch.Tensor,
    field_valid: torch.Tensor,
) -> torch.Tensor:
    """Classify complete canonical 4×4 groups without assigning an 8 µm owner."""

    owner = _spatial_owner(owner_map)
    valid = _spatial_valid(field_valid, tuple(owner.shape))
    batch, rows, columns = owner.shape
    if rows % 4 or columns % 4:
        raise ValueError("v033 field cannot form canonical 8um groups")
    grouped_owner = (
        owner.reshape(batch, rows // 4, 4, columns // 4, 4)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, rows // 4, columns // 4, 16)
    )
    grouped_valid = (
        valid.reshape(batch, rows // 4, 4, columns // 4, 4)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, rows // 4, columns // 4, 16)
    )
    result = torch.full(
        grouped_owner.shape[:3], STRATUM_INVALID, dtype=torch.int8, device=owner.device
    )
    complete = grouped_valid.all(dim=-1)
    covered = grouped_owner >= 0
    all_covered = covered.all(dim=-1)
    all_gap = (~covered).all(dim=-1)
    safe = torch.where(covered, grouped_owner, torch.zeros_like(grouped_owner))
    single = all_covered & (safe.amin(dim=-1) == safe.amax(dim=-1))
    result[complete & single] = STRATUM_SINGLE_PARENT
    result[complete & all_covered & ~single] = STRATUM_MULTI_PARENT
    result[complete & ~all_covered & ~all_gap] = STRATUM_BOUNDARY
    result[complete & all_gap] = STRATUM_GAP_ONLY
    return result


__all__ = [
    "ParentGeometry",
    "STRATUM_BOUNDARY",
    "STRATUM_GAP_ONLY",
    "STRATUM_INVALID",
    "STRATUM_MULTI_PARENT",
    "STRATUM_SINGLE_PARENT",
    "aggregate_parent_counts",
    "classify_8um_ownership",
    "derive_parent_geometry",
    "is_single_4_connected",
    "masks_to_owner",
    "parent_child_counts",
    "validate_owner_batch",
]
