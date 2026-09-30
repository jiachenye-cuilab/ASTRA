"""Shared image and expression encoders with protocol conditioning."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from astra.model.head import ExpandedSegmentHead
from astra.model.image import ExpandedMaskedImageEncoder


class DilatedImageEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        hidden_channels: int,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0 or not dilations:
            raise ValueError("v012 image encoder dimensions differ")
        channels = (int(input_channels),) + (int(hidden_channels),) * len(dilations)
        self.layers = nn.ModuleList(
            nn.Conv2d(
                channels[index],
                channels[index + 1],
                kernel_size=3,
                padding=int(dilation),
                dilation=int(dilation),
                bias=False,
            )
            for index, dilation in enumerate(dilations)
        )

    def forward(self, image: torch.Tensor, field_valid: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or field_valid.shape != image.shape[:1] + image.shape[2:]:
            raise ValueError("v012 image/valid field shapes differ")
        mask = field_valid[:, None].to(dtype=image.dtype)
        hidden = image * mask
        for layer in self.layers:
            hidden = F.gelu(layer(hidden)) * mask
        return hidden


class ExclusiveOwnerFieldModel(nn.Module):
    """Predict one positive field, then normalize within each disjoint parent."""

    def __init__(
        self,
        *,
        num_genes: int,
        image_feature_dimension: int = 6,
        field_shape: tuple[int, int] = (128, 128),
        cell_um: float = 2.0,
        image_channels: int = 32,
        parent_channels: int = 32,
        width: int = 64,
        rank: int = 32,
        image_dilations: tuple[int, ...] = (1, 2, 4),
        residual_bound: float = 4.0,
        gene_exposure_log_bound: float = 20.0,
        minimum_prior_mass: float = 1e-10,
        conservation_tolerance: float = 1e-10,
    ) -> None:
        super().__init__()
        rows, columns = (int(value) for value in field_shape)
        if (
            min(
                int(num_genes),
                int(image_feature_dimension),
                rows,
                columns,
                int(image_channels),
                int(parent_channels),
                int(width),
                int(rank),
            )
            <= 0
            or rows % 4
            or columns % 4
            or min(
                float(cell_um),
                float(residual_bound),
                float(gene_exposure_log_bound),
                float(minimum_prior_mass),
                float(conservation_tolerance),
            )
            <= 0
        ):
            raise ValueError("v012 model configuration differs")
        self.num_genes = int(num_genes)
        self.image_feature_dimension = int(image_feature_dimension)
        self.field_shape = (rows, columns)
        self.cell_um = float(cell_um)
        self.parent_channels = int(parent_channels)
        self.residual_bound = float(residual_bound)
        self.gene_exposure_log_bound = float(gene_exposure_log_bound)
        self.minimum_prior_mass = float(minimum_prior_mass)
        self.conservation_tolerance = float(conservation_tolerance)

        self.image_encoder = DilatedImageEncoder(
            input_channels=self.image_feature_dimension,
            hidden_channels=int(image_channels),
            dilations=tuple(int(value) for value in image_dilations),
        )
        self.parent_geometry_encoder = nn.Sequential(
            nn.Linear(7, self.parent_channels),
            nn.GELU(),
            nn.Linear(self.parent_channels, self.parent_channels),
        )
        self.parent_count_encoder = nn.Linear(
            self.num_genes, self.parent_channels, bias=False
        )
        self.parent_normalization = nn.LayerNorm(self.parent_channels)
        query_dimensions = (
            int(image_channels)
            + 2 * self.parent_channels
            + 2  # absolute coordinates
            + 2  # within-parent relative coordinates
            + 4  # coverage at four local scales
            + 1  # field-valid flag
        )
        self.query_lifting = nn.Sequential(
            nn.Linear(query_dimensions, int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width)),
            nn.LayerNorm(int(width)),
        )
        self.query_projection = nn.Linear(int(width), int(rank), bias=False)
        self.gene_embedding = nn.Embedding(self.num_genes, int(rank))
        self.gene_log_exposure = nn.Parameter(torch.zeros(self.num_genes))
        nn.init.zeros_(self.query_projection.weight)
        nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)

        row_axis = 2.0 * (torch.arange(rows, dtype=torch.float32) + 0.5) / rows - 1.0
        column_axis = (
            2.0 * (torch.arange(columns, dtype=torch.float32) + 0.5) / columns
            - 1.0
        )
        yy, xx = torch.meshgrid(row_axis, column_axis, indexing="ij")
        self.register_buffer(
            "query_coordinates", torch.stack((yy, xx), dim=-1), persistent=True
        )

    def _normalize_image_layout(self, image: torch.Tensor) -> torch.Tensor:
        rows, columns = self.field_shape
        if image.ndim == 3 and image.shape[1:] == (
            rows * columns,
            self.image_feature_dimension,
        ):
            image = image.reshape(
                image.shape[0], rows, columns, self.image_feature_dimension
            ).permute(0, 3, 1, 2)
        if image.ndim != 4 or image.shape[1:] != (
            self.image_feature_dimension,
            rows,
            columns,
        ):
            raise ValueError("v012 dense H&E feature layout differs")
        if image.dtype != torch.float32:
            raise TypeError("v012 dense H&E features must use float32")
        return image

    @staticmethod
    def _gather_parent(values: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
        batch, parents = values.shape[:2]
        tail = values.shape[2:]
        flat_owner = owner.reshape(batch, -1)
        offsets = torch.arange(batch, device=owner.device)[:, None] * parents
        indices = flat_owner.clamp_min(0) + offsets
        gathered = values.reshape(batch * parents, *tail)[indices]
        return gathered

    @staticmethod
    def _coverage_features(owner: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        covered = (owner >= 0).to(dtype=dtype)[:, None]
        values = [covered]
        for kernel in (3, 7, 15):
            values.append(
                F.avg_pool2d(covered, kernel_size=kernel, stride=1, padding=kernel // 2)
            )
        return torch.cat(values, dim=1).permute(0, 2, 3, 1)


class DirectGeneHeadFieldModel(ExclusiveOwnerFieldModel):
    """Use one zero-initialized width-to-gene map instead of a learned factorization."""

    def __init__(
        self,
        *,
        num_genes: int,
        image_feature_dimension: int = 6,
        field_shape: tuple[int, int] = (128, 128),
        cell_um: float = 2.0,
        image_channels: int = 32,
        parent_channels: int = 32,
        width: int = 64,
        image_dilations: tuple[int, ...] = (1, 2, 4),
        residual_bound: float = 4.0,
        gene_exposure_log_bound: float = 20.0,
        minimum_prior_mass: float = 1e-10,
        conservation_tolerance: float = 1e-10,
    ) -> None:
        super().__init__(
            num_genes=num_genes,
            image_feature_dimension=image_feature_dimension,
            field_shape=field_shape,
            cell_um=cell_um,
            image_channels=image_channels,
            parent_channels=parent_channels,
            width=width,
            rank=width,
            image_dilations=image_dilations,
            residual_bound=residual_bound,
            gene_exposure_log_bound=gene_exposure_log_bound,
            minimum_prior_mass=minimum_prior_mass,
            conservation_tolerance=conservation_tolerance,
        )
        self.query_projection = nn.Identity()
        self.gene_embedding = nn.Embedding(self.num_genes, int(width))
        nn.init.zeros_(self.gene_embedding.weight)


class Direct8Model(DirectGeneHeadFieldModel):
    """Reuse the field encoder and heads, without the historical training states."""

    def __init__(self, *, expanded_cell_width=128, expanded_segment_width=256,
                 expanded_blocks=2, segment_residual_bound=2.0,
                 image_local_residual_blocks=2, image_group_norm_groups=8, **kwargs):
        super().__init__(**kwargs)
        if segment_residual_bound <= 0:
            raise ValueError("segment residual bound must be positive")
        self.segment_residual_bound = float(segment_residual_bound)
        self.image_encoder = ExpandedMaskedImageEncoder(
            input_channels=self.image_feature_dimension,
            hidden_channels=int(kwargs.get("image_channels", 32)),
            dilations=tuple(kwargs.get("image_dilations", (1, 2, 4))),
            residual_blocks=image_local_residual_blocks,
            normalization_groups=image_group_norm_groups,
        )
        self.expanded_head = ExpandedSegmentHead(
            input_width=self.gene_embedding.embedding_dim, cell_width=expanded_cell_width,
            segment_width=expanded_segment_width, blocks=expanded_blocks, genes=self.num_genes,
        )

    def _prior_residual(self, hidden, protocol_id=None):
        residual = torch.einsum(
            "bhwr,gr->bhwg", self.query_projection(hidden), self.gene_embedding.weight.to(hidden.dtype),
        ) / self.gene_embedding.embedding_dim**0.5
        return residual, self.gene_log_exposure.to(hidden.dtype)


class ProtocolModel:
    @staticmethod
    def _protocol_ids(protocol_id, *, batch_size, device):
        if protocol_id is None:
            raise ValueError("v027 requires protocol_id")
        ids = torch.as_tensor(protocol_id, device=device)
        if ids.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("protocol_id must contain integers")
        if ids.ndim == 0:
            ids = ids.expand(batch_size)
        elif tuple(ids.shape) != (batch_size,):
            raise ValueError("protocol_id must be scalar or have shape (batch,)")
        ids = ids.to(dtype=torch.long)
        if bool(torch.any((ids < 0) | (ids > 1))):
            raise ValueError("protocol_id values must be 0 or 1")
        return ids
