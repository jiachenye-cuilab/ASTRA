"""Parent-relative allocation and an independent absolute gap-density decoder."""

import math

import torch
from torch import nn

from astra.training.segment import pool_segment_hidden


def parent_centered_features(hidden, owner, parent_valid):
    batch, rows, cols, width = hidden.shape
    parents = parent_valid.shape[1]
    covered = owner >= 0
    index = owner + torch.arange(batch, device=owner.device)[:, None, None] * parents
    # Large parents exceed the exact counting range of BF16/FP16. Accumulate
    # statistics in FP32 (or preserve FP64), then return the feature dtype.
    statistics = hidden.float() if hidden.dtype in (torch.float16, torch.bfloat16) else hidden
    sums = statistics.new_zeros((batch * parents, width))
    sums.index_add_(0, index[covered], statistics[covered])
    area = torch.bincount(index[covered], minlength=batch * parents)
    means = (sums / area.clamp_min(1)[:, None]).to(hidden.dtype)
    return torch.where(covered[..., None], hidden - means[index.clamp_min(0)], 0)


class AllocationHead(nn.Module):
    def __init__(self, width, context_width, genes, rank, bound, *, post_center_norm="layernorm"):
        super().__init__()
        if type(post_center_norm) is not str or post_center_norm not in ("layernorm", "none"):
            raise ValueError("allocation_post_center_norm must be layernorm or none")
        self.bound = float(bound)
        self.rank = rank
        self.post_center_norm = post_center_norm
        normalization = nn.LayerNorm(width) if post_center_norm == "layernorm" else nn.Identity()
        self.shape = nn.Sequential(normalization, nn.Linear(width, width), nn.GELU(),
                                   nn.Linear(width, rank))
        self.context_scale = nn.Linear(context_width, rank)
        self.gene_embedding = nn.Parameter(torch.zeros(genes, rank))
        nn.init.zeros_(self.context_scale.weight)
        nn.init.zeros_(self.context_scale.bias)

    def forward(self, hidden, parent_context, owner, parent_valid, layout):
        # Center features before nonlinear decoding, rather than centering logits
        # (a parent-constant shift of logits would cancel in normalization).
        relative = parent_centered_features(hidden, owner, parent_valid)
        shape = pool_segment_hidden(self.shape(relative), layout)
        covered = layout.segment_owner >= 0
        context = parent_context[layout.segment_batch[covered], layout.segment_owner[covered]]
        modes = shape[covered] * (1 + torch.tanh(self.context_scale(context)))
        raw = modes @ self.gene_embedding.T / math.sqrt(self.rank)
        return self.bound * torch.tanh(raw / self.bound)


class GapHead(nn.Module):
    def __init__(self, input_width, hidden_width, genes, bound, exposure_bound=20.0):
        super().__init__()
        self.bound, self.exposure_bound = float(bound), float(exposure_bound)
        self.features = nn.Sequential(nn.LayerNorm(input_width), nn.Linear(input_width, hidden_width),
                                      nn.GELU(), nn.Linear(hidden_width, hidden_width), nn.GELU())
        self.output = nn.Linear(hidden_width, genes)
        self.log_exposure = nn.Parameter(torch.zeros(genes))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        raw = self.output(self.features(features))
        return (self.bound * torch.tanh(raw / self.bound)
                + self.exposure_bound * torch.tanh(self.log_exposure / self.exposure_bound))


def allocate(covered_logits, gap_mass, layout, counts, parent_valid, *, tolerance=1e-10, validate=True):
    """Normalize over owner x gene; gap segments never enter a parent denominator."""
    batch, parents, genes = counts.shape
    covered = layout.segment_owner >= 0
    index = layout.segment_batch[covered] * parents + layout.segment_owner[covered]
    logits = covered_logits.double() + layout.segment_area_children[covered, None].double().log()
    maximum = logits.new_full((batch * parents, genes), -torch.inf)
    maximum.scatter_reduce_(0, index[:, None].expand_as(logits), logits.detach(), reduce="amax", include_self=True)
    weights = (logits - maximum[index]).exp()
    denominator = logits.new_zeros((batch * parents, genes)).index_add(0, index, weights)
    probability = weights / denominator[index].clamp_min(torch.finfo(torch.float64).tiny)
    covered_mass = probability * counts.double().reshape(-1, genes)[index]
    mass = counts.new_zeros((layout.segments, genes), dtype=torch.float64)
    mass[covered] = covered_mass
    mass[~covered] = gap_mass.double()
    parent_sum = mass.new_zeros((batch * parents, genes)).index_add(0, index, covered_mass).reshape_as(counts)
    residual = torch.where(parent_valid[..., None], parent_sum - counts, 0)
    if validate and (not bool(torch.isfinite(mass).all()) or bool((mass < 0).any())
                     or bool((residual.abs() > tolerance * (1 + counts.abs())).any())):
        raise ValueError("nonfinite allocation or parent count conservation failure")
    return mass, parent_sum, residual


def aggregate_segments(mass, layout, batch_size):
    rows, cols = layout.groups_per_axis
    index = layout.segment_batch * rows * cols + layout.segment_group
    return mass.new_zeros((batch_size * rows * cols, mass.shape[-1])).index_add(0, index, mass).reshape(
        batch_size, rows, cols, mass.shape[-1])
