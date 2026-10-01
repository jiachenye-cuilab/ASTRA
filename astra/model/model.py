"""Panel-independent ASTRA with a WT trunk and a 3prime expression adapter."""

from dataclasses import dataclass

import torch
from torch import nn

from astra.model.backbone import Direct8Model as BaseModel


@dataclass(frozen=True)
class GenePanels:
    input_gene_ids: tuple[str, ...]
    output_gene_ids: tuple[str, ...]

    def __post_init__(self):
        for name in ("input_gene_ids", "output_gene_ids"):
            genes = tuple(getattr(self, name))
            if not genes or any(not isinstance(g, str) or not g.strip() for g in genes):
                raise ValueError(f"{name} must contain nonempty gene IDs")
            if len(set(genes)) != len(genes):
                raise ValueError(f"{name} contains duplicate gene IDs")
            object.__setattr__(self, name, genes)
        if not set(self.output_gene_ids).issubset(self.input_gene_ids):
            raise ValueError("output genes must be a subset of input genes for exact conservation")

    @property
    def output_indices(self):
        positions = {gene: i for i, gene in enumerate(self.input_gene_ids)}
        return tuple(positions[gene] for gene in self.output_gene_ids)

    def as_dict(self):
        return dict(input_gene_ids=list(self.input_gene_ids), output_gene_ids=list(self.output_gene_ids))


class Direct8Model(BaseModel):
    """Inputs are ordered coarse counts; all spatial gene tensors use output genes only.

    Protocol IDs are 0 = 3prime and 1 = WT. The adapter acts on
    log density before the shared expression encoder. No case adaptation branch.
    """

    def __init__(self, *, input_gene_ids, output_gene_ids, adapter_width=8,
                 parent_channels=32, gap_prior_mode="global", efficient_execution=True,
                 use_extra_input_genes=True, use_three_prime_adapter=True,
                 output_head_mode="shared", expression_normalization="log_density",
                 context_gene_ids=None, **kwargs):
        panels = GenePanels(input_gene_ids, output_gene_ids)
        if adapter_width <= 0:
            raise ValueError("adapter_width must be positive")
        if "num_genes" in kwargs or "shared_protocol" in kwargs:
            raise ValueError("ASTRA derives gene dimensions and fixes asymmetric protocol sharing")
        if expression_normalization not in ("log_density", "cp10k"):
            raise ValueError("expression_normalization must be log_density or cp10k")
        super().__init__(num_genes=len(panels.output_gene_ids), parent_channels=parent_channels,
                         shared_protocol=False, gap_prior_mode=gap_prior_mode,
                         efficient_execution=efficient_execution, **kwargs)
        self.panels = panels
        self.num_input_genes = len(panels.input_gene_ids)
        self.use_extra_input_genes = bool(use_extra_input_genes)
        self.use_three_prime_adapter = bool(use_three_prime_adapter)
        # Mapping is derived from checkpoint-checked IDs, never mutable checkpoint state.
        self.register_buffer("output_gene_indices", torch.tensor(panels.output_indices), persistent=False)
        context_gene_mask = torch.ones(self.num_input_genes, dtype=torch.bool)
        if not self.use_extra_input_genes:
            context_gene_mask.zero_()
            context_gene_mask[list(panels.output_indices)] = True
        self.context_gene_ids = None
        if context_gene_ids is not None:
            if (not isinstance(context_gene_ids, (list, tuple)) or not context_gene_ids
                    or any(not isinstance(g, str) for g in context_gene_ids)
                    or len(set(context_gene_ids)) != len(context_gene_ids)):
                raise ValueError("context genes must be a nonempty list of unique gene IDs")
            selected = set(context_gene_ids)
            allowed = set(panels.input_gene_ids if self.use_extra_input_genes else panels.output_gene_ids)
            if not selected <= allowed:
                raise ValueError("context genes must belong to the allowed input panel")
            self.context_gene_ids = tuple(g for g in panels.input_gene_ids if g in selected)
            context_gene_mask &= torch.tensor([g in selected for g in panels.input_gene_ids])
        self.register_buffer("context_gene_mask", context_gene_mask, persistent=False)
        del self.expression_encoders
        self.expression_encoder = nn.Sequential(
            nn.Linear(self.num_input_genes, parent_channels, bias=False), nn.GELU(),
            nn.Linear(parent_channels, parent_channels), nn.LayerNorm(parent_channels))
        self.availability_encoder = nn.Linear(self.num_input_genes, parent_channels, bias=False)
        self.three_prime_adapter = nn.Sequential(
            nn.LayerNorm(self.num_input_genes), nn.Linear(self.num_input_genes, adapter_width), nn.GELU(),
            nn.Linear(adapter_width, self.num_input_genes, bias=False))
        nn.init.zeros_(self.three_prime_adapter[-1].weight)
        if not self.use_three_prime_adapter:
            self.three_prime_adapter.requires_grad_(False)
        self.morphology.protocol_adapters = nn.ModuleList()
        if output_head_mode not in ("shared", "protocol", "coverage"):
            raise ValueError("output_head_mode must be shared, protocol or coverage")
        self.output_head_mode = output_head_mode
        if output_head_mode == "shared":
            self.protocol_residual_heads = nn.ModuleList()
        self.expression_normalization = expression_normalization
        self.expression_scale_encoder = None
        if expression_normalization == "cp10k":
            self.expression_scale_encoder = nn.Linear(2, parent_channels, bias=False)
            nn.init.zeros_(self.expression_scale_encoder.weight)
        self.model_kwargs = dict(input_gene_ids=list(panels.input_gene_ids),
            output_gene_ids=list(panels.output_gene_ids), adapter_width=adapter_width,
            parent_channels=parent_channels, gap_prior_mode=gap_prior_mode,
            efficient_execution=efficient_execution, **kwargs)
        # Omitted defaults retain compatibility with the original checkpoints.
        if not self.use_extra_input_genes:
            self.model_kwargs["use_extra_input_genes"] = False
        if not self.use_three_prime_adapter:
            self.model_kwargs["use_three_prime_adapter"] = False
        if output_head_mode != "shared":
            self.model_kwargs["output_head_mode"] = output_head_mode
        if expression_normalization != "log_density":
            self.model_kwargs["expression_normalization"] = expression_normalization
        if self.context_gene_ids is not None:
            self.model_kwargs["context_gene_ids"] = list(self.context_gene_ids)

    def get_extra_state(self):
        state = dict(version="v030", adapter_position="before_expression_encoder", **self.panels.as_dict())
        if self.expression_normalization != "log_density":
            state["expression_normalization"] = self.expression_normalization
        if self.context_gene_ids is not None:
            state["context_gene_ids"] = list(self.context_gene_ids)
        return state

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("ASTRA checkpoint gene identities/order differ from this model")

    def _encode_expression(self, density, image, ids, available, *, parent_area=None):
        available_context = available & self.context_gene_mask
        density = torch.where(self.context_gene_mask, density, 0.0)
        if self.expression_normalization == "cp10k":
            if parent_area is None:
                raise ValueError("CP10K requires observed parent areas")
            density = torch.where(available_context[:, None], density, 0.0)
            total_density = density.sum(-1, keepdim=True)
            # No pseudocount in the denominator: zero-UMI parents stay zero and
            # low-density parents retain the same count proportions.
            denominator = torch.where(total_density > 0, total_density, 1.0)
            features = torch.log1p(10000.0 * density / denominator).to(image.dtype)
            total_umi = total_density.squeeze(-1) * parent_area
            scale = torch.stack((torch.log1p(total_umi), torch.log1p(parent_area)), dim=-1)
        else:
            features = torch.log1p(density).to(image.dtype)
        indices = torch.nonzero(ids == 0, as_tuple=False).flatten()
        if self.use_three_prime_adapter and indices.numel():
            selected = features.index_select(0, indices)
            residual = self.three_prime_adapter(selected)
            # An assay-missing gene must not be synthesized by the adapter.
            residual = residual * available_context.index_select(0, indices)[:, None].to(residual.dtype)
            features = features.index_copy(0, indices, selected + residual)
        expression = self.expression_encoder(features)
        if self.expression_scale_encoder is not None:
            expression = expression + self.expression_scale_encoder(scale.to(image.dtype))
        return expression + self.availability_encoder(((~available) & self.context_gene_mask).to(image.dtype))[:, None]

    def select_output_genes(self, tensor):
        if tensor.shape[-1] != self.num_input_genes:
            raise ValueError("tensor must follow the input gene panel order")
        return tensor.index_select(-1, self.output_gene_indices)

    def _output_observations(self, counts, density, available):
        return tuple(self.select_output_genes(value) for value in (counts, density, available))
