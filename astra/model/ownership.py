"""Observation ownership, spatial validity, and parent geometry."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import torch


def _spatial_owner(owner_map: torch.Tensor) -> torch.Tensor:
    owner = torch.as_tensor(owner_map)
    if owner.ndim == 2:
        owner = owner.unsqueeze(0)
    if owner.ndim != 3 or owner.dtype != torch.long:
        raise TypeError("v012 owner_map must be int64 [batch,rows,columns]")
    return owner


def _spatial_valid(field_valid: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    valid = torch.as_tensor(field_valid)
    if valid.ndim == 2:
        valid = valid.unsqueeze(0)
    if valid.shape != shape or valid.dtype != torch.bool:
        raise TypeError("v012 field_valid must be bool and match owner_map")
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
        raise ValueError("v012 requires at least one padded parent slot")
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
    if value.ndim != 2 or not bool(value.any()):
        return False
    rows, columns = value.shape
    start = tuple(int(v) for v in torch.nonzero(value, as_tuple=False)[0])
    queue: deque[tuple[int, int]] = deque([start])
    visited = {start}
    while queue:
        row, column = queue.popleft()
        for next_row, next_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            point = (next_row, next_column)
            if (
                0 <= next_row < rows
                and 0 <= next_column < columns
                and point not in visited
                and bool(value[next_row, next_column])
            ):
                visited.add(point)
                queue.append(point)
    return len(visited) == int(value.sum())


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
        raise TypeError("v012 parent_valid must be bool [batch,parents] on one device")
    parents = valid_parent.shape[1]
    if bool(torch.any(owner < -1)) or bool(torch.any(owner >= parents)):
        raise ValueError("v012 owner id lies outside {-1, ..., parents-1}")
    if bool(torch.any((owner >= 0) & ~valid_field)):
        raise ValueError("v012 invalid field cell cannot have an owner")
    counts = parent_child_counts(owner, parents)
    if bool(torch.any(valid_parent & (counts == 0))):
        raise ValueError("v012 valid parent must own at least one child")
    if bool(torch.any(~valid_parent & (counts != 0))):
        raise ValueError("v012 invalid padded parent owns children")
    if check_connectivity:
        owner_cpu = owner.detach().cpu()
        valid_cpu = valid_parent.detach().cpu()
        for batch_index in range(owner.shape[0]):
            for parent_index in torch.nonzero(valid_cpu[batch_index], as_tuple=False).flatten():
                if not is_single_4_connected(
                    owner_cpu[batch_index] == int(parent_index)
                ):
                    raise ValueError("v012 parent mask is not one 4-connected region")


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
        raise ValueError("v012 geometry dimensions differ")
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
