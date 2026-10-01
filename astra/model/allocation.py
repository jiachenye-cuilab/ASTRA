"""Unchanged direct-8 pooling and allocator used by the historical control."""

from __future__ import annotations

import torch

from astra.model.segment import SegmentLayout


def pool_segment_mass(values_2um: torch.Tensor, layout: SegmentLayout) -> torch.Tensor:
    values = torch.as_tensor(values_2um)
    if values.ndim != 4 or values.shape[:3] != layout.cell_to_segment.shape:
        raise ValueError("ASTRA marginal field/layout shape differs")
    selected = (
        values.reshape(-1, values.shape[-1])
        .index_select(0, layout.valid_cell_indices)
        .to(dtype=torch.float64)
    )
    inverse = layout.cell_to_segment.reshape(-1).index_select(
        0, layout.valid_cell_indices
    )
    result = torch.zeros(
        (layout.segments, values.shape[-1]),
        dtype=torch.float64,
        device=values.device,
    )
    result.scatter_add_(0, inverse[:, None].expand_as(selected), selected)
    return result


def aggregate_segments(mass, layout, batch_size):
    gy, gx = layout.groups_per_axis
    index = layout.segment_batch * (gy * gx) + layout.segment_group
    output = mass.new_zeros((batch_size * gy * gx, mass.shape[-1]))
    output.index_add_(0, index, mass)
    return output.reshape(batch_size, gy, gx, mass.shape[-1])


def uniform_bridge(mass, layout):
    """Optional compatibility raster; it does not contain learned 2um detail."""
    inverse = layout.cell_to_segment.flatten()[layout.valid_cell_indices]
    values = mass[inverse] / layout.segment_area_children[inverse, None]
    bridge = mass.new_zeros((layout.cell_to_segment.numel(), mass.shape[-1]))
    bridge[layout.valid_cell_indices] = values
    return bridge.reshape(*layout.cell_to_segment.shape, mass.shape[-1])


def allocate(prior, residual, layout, counts, parent_valid, *, tolerance, validate):
    counts = counts.to(torch.float64)
    batch, parents, genes = counts.shape
    weights = prior.to(torch.float64) * residual.to(torch.float64).exp()
    covered = layout.segment_owner >= 0
    index = layout.segment_batch[covered] * parents + layout.segment_owner[covered]
    denominator = weights.new_zeros((batch * parents, genes))
    denominator.index_add_(0, index, weights[covered])
    positive = parent_valid[..., None] & (counts > 0)
    denominator = denominator.reshape_as(counts)
    if validate and bool(torch.any(positive & (denominator <= 0))):
        raise ValueError("positive parent count has no allocation support")
    scale = torch.where(
        positive, counts / denominator.clamp_min(torch.finfo(torch.float64).tiny),
        torch.zeros_like(counts),
    )
    mass = weights.clone()
    mass[covered] = weights[covered] * scale.reshape(-1, genes)[index]
    parent_sum = mass.new_zeros((batch * parents, genes))
    parent_sum.index_add_(0, index, mass[covered])
    parent_sum = parent_sum.reshape_as(counts)
    error = torch.where(parent_valid[..., None], parent_sum - counts, 0.0)
    if validate:
        if not bool(torch.isfinite(mass).all()) or bool(torch.any(mass < 0)):
            raise ValueError("allocation must be finite and nonnegative")
        if bool(torch.any(error.abs() > tolerance * (1 + counts.abs()))):
            raise ValueError("parent count conservation failed")
    return mass, parent_sum, error
