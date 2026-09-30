"""Unchanged expanded segment head used by the historical control."""

from __future__ import annotations

import math

import torch
from torch import nn

from astra.model.segment import SegmentLayout


class ResidualMlpBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(int(width))
        self.first = nn.Linear(int(width), int(width))
        self.second = nn.Linear(int(width), int(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.normalization(value)
        update = torch.nn.functional.gelu(self.first(update))
        return value + self.second(update)


class ExpandedSegmentHead(nn.Module):
    """Preserve cell-level variation before a wider segment residual tower."""

    geometry_features = 8

    def __init__(
        self,
        *,
        input_width: int,
        cell_width: int,
        segment_width: int,
        blocks: int,
        genes: int,
    ) -> None:
        super().__init__()
        self.cell_adapter = nn.Sequential(
            nn.LayerNorm(int(input_width)),
            nn.Linear(int(input_width), int(cell_width)),
            nn.GELU(),
            nn.Linear(int(cell_width), int(cell_width)),
            nn.LayerNorm(int(cell_width)),
        )
        pooled_width = 3 * int(cell_width) + self.geometry_features
        self.segment_input = nn.Sequential(
            nn.LayerNorm(pooled_width),
            nn.Linear(pooled_width, int(segment_width)),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            ResidualMlpBlock(int(segment_width)) for _ in range(int(blocks))
        )
        self.output_normalization = nn.LayerNorm(int(segment_width))
        self.output = nn.Linear(int(segment_width), int(genes))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _statistics(
        cell_features: torch.Tensor, layout: SegmentLayout
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = cell_features.reshape(-1, cell_features.shape[-1]).index_select(
            0, layout.valid_cell_indices
        )
        inverse = layout.cell_to_segment.reshape(-1).index_select(
            0, layout.valid_cell_indices
        )
        shape = (layout.segments, values.shape[-1])
        total = torch.zeros(shape, dtype=values.dtype, device=values.device)
        square = torch.zeros_like(total)
        total.scatter_add_(0, inverse[:, None].expand_as(values), values)
        square.scatter_add_(0, inverse[:, None].expand_as(values), values.square())
        area = layout.segment_area_children[:, None].to(dtype=values.dtype)
        mean = total / area
        variance = (square / area - mean.square()).clamp_min(0)
        standard_deviation = torch.sqrt(variance + 1e-6)
        maximum = torch.full_like(total, -torch.inf)
        maximum.scatter_reduce_(
            0,
            inverse[:, None].expand_as(values),
            values,
            reduce="amax",
            include_self=True,
        )
        return mean, standard_deviation, maximum

    @staticmethod
    def _geometry(
        layout: SegmentLayout,
        *,
        parent_area: torch.Tensor,
        field_shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows, columns = field_shape
        groups_y, groups_x = layout.groups_per_axis
        flat = layout.valid_cell_indices
        spatial = flat % (rows * columns)
        yy = spatial // columns
        xx = spatial % columns
        local_y = ((yy % 4).to(dtype=dtype) + 0.5) / 2.0 - 1.0
        local_x = ((xx % 4).to(dtype=dtype) + 0.5) / 2.0 - 1.0
        inverse = layout.cell_to_segment.reshape(-1).index_select(0, flat)
        centroid = torch.zeros(
            (layout.segments, 2), dtype=dtype, device=flat.device
        )
        centroid.scatter_add_(
            0,
            inverse[:, None].expand(-1, 2),
            torch.stack((local_y, local_x), dim=1),
        )
        area = layout.segment_area_children.to(dtype=dtype)
        centroid = centroid / area[:, None]
        group_y = layout.segment_group // groups_x
        group_x = layout.segment_group % groups_x
        group_y = 2.0 * (group_y.to(dtype=dtype) + 0.5) / groups_y - 1.0
        group_x = 2.0 * (group_x.to(dtype=dtype) + 0.5) / groups_x - 1.0
        covered = layout.segment_owner >= 0
        owner_area = torch.ones_like(area)
        owner_area[covered] = parent_area[
            layout.segment_batch[covered], layout.segment_owner[covered]
        ].to(dtype=dtype)
        fraction_of_parent = torch.where(
            covered, area / owner_area.clamp_min(1.0), torch.zeros_like(area)
        )
        return torch.stack(
            (
                torch.log1p(area) / math.log(17.0),
                area / 16.0,
                centroid[:, 0],
                centroid[:, 1],
                group_y,
                group_x,
                (~covered).to(dtype=dtype),
                fraction_of_parent,
            ),
            dim=1,
        )

    def forward(
        self,
        hidden_2um: torch.Tensor,
        layout: SegmentLayout,
        *,
        parent_area: torch.Tensor,
        field_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adapted = self.cell_adapter(hidden_2um)
        mean, standard_deviation, maximum = self._statistics(adapted, layout)
        geometry = self._geometry(
            layout,
            parent_area=parent_area,
            field_shape=field_shape,
            dtype=adapted.dtype,
        )
        pooled = torch.cat((mean, standard_deviation, maximum, geometry), dim=1)
        value = self.segment_input(pooled)
        for block in self.blocks:
            value = block(value)
        residual = self.output(self.output_normalization(value))
        return residual, pooled
