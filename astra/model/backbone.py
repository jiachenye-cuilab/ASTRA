"""Local spatial backbone with assay-aware expression and image routes."""

from copy import deepcopy

import torch
from torch import nn

from astra.model.ownership import derive_parent_geometry, validate_owner_batch
from astra.model.allocation import pool_segment_mass
from astra.model.segment import build_segment_layout
from astra.model.base import Direct8Model as PreviousModel
from astra.model.allocation import aggregate_segments, allocate, uniform_bridge
from astra.model.base import ProtocolModel
from astra.model.context import LocalParentReader
from astra.model.morphology import MultiScaleMorphology


_CONFIGURED_GAP_PRIOR = object()


class Direct8Model(PreviousModel):
    """The foundation encoder is external; only its frozen spatial features enter here."""

    def __init__(self, *, context_width=64, context_neighbors=8, context_radius_um=64.0,
                 query_chunk_size=256, pathology_channels=1024, private_head_width=32,
                 image_channels=64, parent_channels=32, width=64,
                 image_dilations=(1, 2, 4), image_local_residual_blocks=2,
                 image_group_norm_groups=8, gap_prior_mode="local", efficient_execution=False,
                 use_pathology=True, use_local_read=True, shared_protocol=False,
                 use_multiscale=True, parent_graph=False, context_heads=1, adaptive_fusion=False,
                 parent_image_context=True, image_guided_read=True,
                 interpolation_power=2.0, interpolation_epsilon_um=1.0,
                 context_position_scale_um=None, context_weighting="attention", **kwargs):
        super().__init__(image_channels=image_channels, parent_channels=parent_channels, width=width,
                         image_dilations=image_dilations, image_local_residual_blocks=image_local_residual_blocks,
                         image_group_norm_groups=image_group_norm_groups, **kwargs)
        if context_width <= 0 or private_head_width <= 0:
            raise ValueError("context and private residual widths must be positive")
        if gap_prior_mode not in ("local", "global", "interpolated"):
            raise ValueError("gap_prior_mode must be local or global")
        self.gap_prior_mode = gap_prior_mode
        self.interpolation_power = float(interpolation_power)
        self.interpolation_epsilon_um = float(interpolation_epsilon_um)
        self.efficient_execution = bool(efficient_execution)
        self.use_local_read = bool(use_local_read)
        self.shared_protocol = bool(shared_protocol)
        # Reuse the shared prediction heads and allocator, replacing the old own/global encoder.
        for name in ("image_encoder", "parent_count_encoder", "parent_geometry_encoder", "parent_normalization", "query_lifting"):
            delattr(self, name)
        self.morphology = MultiScaleMorphology(
            input_channels=self.image_feature_dimension, image_channels=image_channels,
            dilations=image_dilations, residual_blocks=image_local_residual_blocks,
            normalization_groups=image_group_norm_groups, pathology_channels=pathology_channels,
            use_pathology=use_pathology, use_multiscale=use_multiscale, adaptive_fusion=adaptive_fusion,
        )
        encoder = nn.Sequential(nn.Linear(self.num_genes, parent_channels, bias=False), nn.GELU(),
                                nn.Linear(parent_channels, parent_channels), nn.LayerNorm(parent_channels))
        self.expression_encoders = nn.ModuleList([encoder, deepcopy(encoder)])
        # Missing assay genes and measured zeroes must remain distinguishable to the encoder.
        self.availability_encoder = nn.Linear(self.num_genes, parent_channels, bias=False)
        self.parent_reader = LocalParentReader(image_channels=image_channels, parent_channels=parent_channels,
            context_width=context_width, neighbors=context_neighbors, radius_um=context_radius_um,
            query_chunk_size=query_chunk_size, heads=context_heads, parent_graph=parent_graph,
            parent_image_context=parent_image_context, image_guided_read=image_guided_read,
            position_scale_um=context_position_scale_um, weighting=context_weighting)
        if not self.use_local_read:
            for module in (self.parent_reader.query, self.parent_reader.key,
                           self.parent_reader.value, self.parent_reader.relative_bias):
                module.requires_grad_(False)
        query_dimensions = image_channels + 2*context_width + 2 + 2 + 4 + 1 + 2
        self.query_lifting = nn.Sequential(nn.Linear(query_dimensions, width), nn.GELU(),
                                           nn.Linear(width, width), nn.LayerNorm(width))
        pooled_width = self.expanded_head.segment_input[0].normalized_shape[0]
        private_head = nn.Sequential(nn.LayerNorm(pooled_width), nn.Linear(pooled_width, private_head_width),
                                     nn.GELU(), nn.Linear(private_head_width, self.num_genes, bias=False))
        nn.init.zeros_(private_head[-1].weight)
        self.protocol_residual_heads = nn.ModuleList([private_head, deepcopy(private_head)])
        if self.shared_protocol:
            # Retain both checkpoint prefixes. A baseline initializer must copy each .0
            # state into both aliases before loading, so a private .1 cannot overwrite it.
            for modules in (self.expression_encoders, self.morphology.protocol_adapters, self.protocol_residual_heads):
                modules[1] = modules[0]

    def _inputs(self, counts, owner, parent_valid, field_valid, image, protocol_id, available, parent_geometry, validate):
        input_genes = getattr(self, "num_input_genes", self.num_genes)
        owner = owner.unsqueeze(0) if owner.ndim == 2 else owner
        counts = counts.unsqueeze(0) if counts.ndim == 2 else counts
        parent_valid = parent_valid.unsqueeze(0) if parent_valid.ndim == 1 else parent_valid
        field_valid = field_valid.unsqueeze(0) if field_valid.ndim == 2 else field_valid
        image = self._normalize_image_layout(image)
        if (owner.ndim != 3 or tuple(owner.shape[1:]) != self.field_shape or owner.dtype != torch.long
                or parent_valid.ndim != 2 or parent_valid.dtype != torch.bool
                or parent_valid.shape[0] != owner.shape[0] or parent_valid.shape[1] < 1
                or counts.shape != (*parent_valid.shape, input_genes)
                or field_valid.shape != owner.shape or field_valid.dtype != torch.bool
                or image.shape[0] != owner.shape[0]
                or any(x.device != owner.device for x in (counts, parent_valid, field_valid, image))):
            raise ValueError("ASTRA observation shapes, dtypes or devices differ")
        ids = ProtocolModel._protocol_ids(protocol_id, batch_size=owner.shape[0], device=owner.device)
        if available is None:
            available = torch.ones((owner.shape[0], input_genes), dtype=torch.bool, device=owner.device)
        elif (available.dtype != torch.bool or available.device != owner.device
              or available.shape not in ((input_genes,), (owner.shape[0], input_genes))):
            raise ValueError("gene_available must be a boolean G or BxG assay mask")
        available = torch.broadcast_to(available, (owner.shape[0], input_genes))
        active = parent_valid[..., None] & available[:, None]
        safe_counts = torch.where(active, counts, 0).to(torch.float64)
        if validate:
            validate_owner_batch(owner, parent_valid, field_valid)
            if not bool(torch.isfinite(safe_counts).all()) or bool(torch.any(safe_counts < 0)):
                raise ValueError("observed available parent counts must be finite and nonnegative")
        geometry = (parent_geometry if self.efficient_execution and parent_geometry is not None and not validate
                    else derive_parent_geometry(owner, parent_valid, cell_um=self.cell_um))
        if parent_geometry is not None and validate:
            for key in ("features", "centers_um", "sizes_um", "area_children", "perimeter_edges"):
                if not torch.equal(getattr(parent_geometry, key), getattr(geometry, key)):
                    raise ValueError("supplied parent geometry differs from actual owner support")
        return safe_counts, owner, parent_valid, field_valid, image, ids, available, geometry

    def _encode_expression(self, density, image, ids, available, *, parent_area=None):
        expression = image.new_zeros((*density.shape[:2], self.parent_channels))
        log_density = torch.log1p(density).to(image.dtype)
        for protocol, encoder in enumerate(self.expression_encoders):
            indices = torch.nonzero(ids == protocol, as_tuple=False).flatten()
            if indices.numel():
                expression = expression.index_copy(0, indices, encoder(log_density.index_select(0, indices)))
        return expression + self.availability_encoder((~available).to(image.dtype))[:, None]

    def _output_observations(self, counts, density, available):
        return counts, density, available

    def forward(self, *, parent_counts, owner_map, parent_valid, image_features_2um, field_valid,
                protocol_id, gene_available=None, parent_geometry=None, pathology_features=None,
                pathology_valid=None, validate=True, return_diagnostics=False, return_uniform_bridge=False,
                gap_prior_mode=_CONFIGURED_GAP_PRIOR):
        gap_prior_mode = self.gap_prior_mode if gap_prior_mode is _CONFIGURED_GAP_PRIOR else gap_prior_mode
        if gap_prior_mode not in ("local", "global", "interpolated"):
            raise ValueError("gap_prior_mode must be local or global")
        counts, owner, valid_parent, valid_field, image, ids, available, geometry = self._inputs(
            parent_counts, owner_map, parent_valid, field_valid, image_features_2um, protocol_id,
            gene_available, parent_geometry, validate)
        shape = owner.shape
        morphology = self.morphology(image, valid_field, ids,
                                     pathology_features=pathology_features, pathology_valid=pathology_valid)
        area = geometry.area_children.clamp_min(1)
        density = counts / area[..., None]
        expression = self._encode_expression(density, image, ids, available, parent_area=area)
        expression = torch.where(valid_parent[..., None], expression, 0.0)
        counts, density, available = self._output_observations(counts, density, available)
        context = self.parent_reader(image_features=morphology, expression_features=expression,
            parent_density=density, owner_map=owner, parent_valid=valid_parent, field_valid=valid_field,
            geometry=geometry, cell_um=self.cell_um,
            compute_density=not (self.efficient_execution and gap_prior_mode == "global" and not return_diagnostics),
            density_mode="inverse_distance" if gap_prior_mode == "interpolated" else "attention",
            interpolation_power=self.interpolation_power, interpolation_epsilon_um=self.interpolation_epsilon_um)

        support = context["local_support"]
        support2 = support.repeat_interleave(4, 1).repeat_interleave(4, 2)
        # A fallback is used only where no local observed parent is available.
        global_context = context["parent_tokens"].sum(1) / valid_parent.sum(1).clamp_min(1)[:, None]
        local_context = torch.where(support2[..., None], context["local_context"], global_context[:, None, None])
        if not self.use_local_read:
            local_context = global_context[:, None, None].expand_as(local_context)
        coordinates = self.query_coordinates.to(image.dtype)[None].expand(shape[0], -1, -1, -1)
        physical_scale = image.new_tensor([self.field_shape[0], self.field_shape[1]]) * self.cell_um / 2
        physical = coordinates * physical_scale
        centers = self._gather_parent(geometry.centers_um.to(image.dtype), owner).reshape(*shape, 2)
        sizes = self._gather_parent(geometry.sizes_um.to(image.dtype), owner).reshape(*shape, 2)
        relative = torch.where((owner >= 0)[..., None],
            2*(physical-centers)/sizes.clamp_min(self.cell_um), 0.0)
        support_features = context["support_features"].repeat_interleave(4, 1).repeat_interleave(4, 2)
        query = torch.cat((morphology.permute(0, 2, 3, 1), context["owner_context"], local_context,
            coordinates, relative, self._coverage_features(owner, image.dtype),
            valid_field[..., None].to(image.dtype), support_features.to(image.dtype)), dim=-1)
        hidden = torch.where(valid_field[..., None], self.query_lifting(query), 0.0)

        global_density = counts.sum(1) / geometry.area_children.sum(1).clamp_min(1)[:, None]
        if self.efficient_execution and gap_prior_mode == "global":
            gap_density = global_density[:, None, None]
        else:
            local_gap_density = context["local_density"].repeat_interleave(4, 1).repeat_interleave(4, 2)
            gap_density = torch.where(support2[..., None], local_gap_density, global_density[:, None, None])
            if gap_prior_mode == "global":
                gap_density = global_density[:, None, None].expand_as(gap_density)
        owner_density = self._gather_parent(density, owner).reshape(*shape, self.num_genes)
        base_mass = torch.where((owner >= 0)[..., None], owner_density, gap_density).clamp_min(self.minimum_prior_mass)
        cell_residual, raw_exposure = self._prior_residual(hidden)
        cell_residual = self.residual_bound * torch.tanh(cell_residual / self.residual_bound)
        exposure = self.gene_exposure_log_bound * torch.tanh(raw_exposure / self.gene_exposure_log_bound)
        prior = base_mass * torch.exp(cell_residual.to(torch.float64) + exposure[None, None, None].to(torch.float64))
        prior = torch.where(valid_field[..., None] & available[:, None, None], prior, 0.0)
        layout = build_segment_layout(owner, valid_field, parent_slots=counts.shape[1], validate=validate)
        segment_prior = pool_segment_mass(prior, layout)
        shared_residual, pooled = self.expanded_head(hidden, layout, parent_area=geometry.area_children, field_shape=self.field_shape)
        private_residual = torch.zeros_like(shared_residual)
        segment_protocol = ids.index_select(0, layout.segment_batch)
        if getattr(self, "output_head_mode", "protocol") == "coverage":
            # Route each owner/gap segment separately, including mixed 8um cells.
            segment_protocol = (layout.segment_owner < 0).long()
        for protocol, head in enumerate(self.protocol_residual_heads):
            indices = torch.nonzero(segment_protocol == protocol, as_tuple=False).flatten()
            if indices.numel():
                private_residual = private_residual.index_copy(0, indices, head(pooled.index_select(0, indices)))
        residual = shared_residual + private_residual
        residual = self.segment_residual_bound * torch.tanh(residual / self.segment_residual_bound)
        mass, parent_sum, error = allocate(segment_prior, residual, layout, counts, valid_parent,
                                          tolerance=self.conservation_tolerance, validate=validate)
        result = dict(pred_count_8um=aggregate_segments(mass, layout, shape[0]), coarse_segment_mass=mass,
            segment_layout=layout, parent_reaggregated=parent_sum, parent_conservation_residual=error,
            fine_branch_present=False)
        if return_uniform_bridge:
            result["coarse_uniform_bridge_count_2um"] = uniform_bridge(mass, layout)
        if return_diagnostics:
            result.update(parent_tokens=context["parent_tokens"], local_support=support,
                local_context=context["local_context"] if self.use_local_read else local_context,
                local_gap_density=context["local_density"],
                global_fallback_density=global_density, private_segment_residual=private_residual,
                shared_segment_residual=shared_residual, morphology_features=morphology)
        return result
