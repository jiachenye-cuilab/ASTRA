"""Canonical observation-by-output-cell segment layouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SegmentLayout:
    cell_to_segment: torch.Tensor
    valid_cell_indices: torch.Tensor
    segment_batch: torch.Tensor
    segment_owner: torch.Tensor
    segment_group: torch.Tensor
    segment_area_children: torch.Tensor
    groups_per_axis: tuple[int, int]

    @property
    def segments(self) -> int:
        return int(self.segment_batch.numel())


def build_segment_layout(
    owner_map: torch.Tensor,
    field_valid: torch.Tensor,
    *,
    parent_slots: int,
    validate: bool = True,
) -> SegmentLayout:
    owner = torch.as_tensor(owner_map)
    valid = torch.as_tensor(field_valid, dtype=torch.bool, device=owner.device)
    if owner.ndim == 2:
        owner = owner.unsqueeze(0)
    if valid.ndim == 2:
        valid = valid.unsqueeze(0)
    if (
        owner.dtype != torch.long
        or valid.shape != owner.shape
        or owner.ndim != 3
        or owner.shape[1] % 4
        or owner.shape[2] % 4
    ):
        raise ValueError("ASTRA owner/valid layout differs")
    batch, rows, columns = owner.shape
    groups_y, groups_x = rows // 4, columns // 4
    groups = groups_y * groups_x
    if parent_slots <= 0:
        raise ValueError("ASTRA parent slot count must be positive")
    if validate and bool(torch.any((owner < -1) | (owner >= parent_slots))):
        raise ValueError("ASTRA owner index lies outside padded parent slots")
    owner_slots = int(parent_slots) + 1
    yy = torch.arange(rows, device=owner.device)[:, None]
    xx = torch.arange(columns, device=owner.device)[None, :]
    group = (yy // 4) * groups_x + xx // 4
    group = group[None].expand(batch, -1, -1)
    batch_index = torch.arange(batch, device=owner.device)[:, None, None]
    keys = ((batch_index * owner_slots + owner + 1) * groups + group)
    flat_valid = valid.reshape(-1)
    valid_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
    unique, inverse = torch.unique(
        keys.reshape(-1)[flat_valid], sorted=True, return_inverse=True
    )
    temporary = unique // groups
    segment_group = unique % groups
    segment_batch = temporary // owner_slots
    segment_owner = temporary % owner_slots - 1
    area = torch.zeros(unique.numel(), dtype=torch.int64, device=owner.device)
    area.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.int64))
    cell_to_segment = torch.full(
        (batch * rows * columns,), -1, dtype=torch.long, device=owner.device
    )
    cell_to_segment[valid_indices] = inverse
    return SegmentLayout(
        cell_to_segment=cell_to_segment.reshape(batch, rows, columns),
        valid_cell_indices=valid_indices,
        segment_batch=segment_batch,
        segment_owner=segment_owner,
        segment_group=segment_group,
        segment_area_children=area,
        groups_per_axis=(groups_y, groups_x),
    )
