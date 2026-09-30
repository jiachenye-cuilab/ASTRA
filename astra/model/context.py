"""Local observation context and interpolation across unobserved gaps."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def pool_parent_image(image_features, owner_map, parent_valid, field_valid):
    """Pool only valid cells actually owned by each parent, never its bounding box."""
    batch, channels, rows, columns = image_features.shape
    parents = parent_valid.shape[1]
    covered = (owner_map >= 0) & field_valid
    covered = covered & parent_valid.gather(1, owner_map.clamp_min(0).flatten(1)).reshape_as(owner_map)
    indices = owner_map + torch.arange(batch, device=owner_map.device)[:, None, None] * parents
    indices = indices[covered]
    values = image_features.permute(0, 2, 3, 1)[covered]
    accumulator_dtype = torch.float32 if image_features.dtype in (torch.float16, torch.bfloat16) else image_features.dtype
    sums = torch.zeros((batch*parents, channels), dtype=accumulator_dtype, device=image_features.device)
    sums.index_add_(0, indices, values.to(accumulator_dtype))
    counts = torch.zeros(batch*parents, dtype=torch.long, device=image_features.device)
    counts.index_add_(0, indices, torch.ones_like(indices))
    pooled = sums / counts.clamp_min(1)[:, None]
    return pooled.reshape(batch, parents, channels).to(image_features.dtype), counts.reshape(batch, parents)


def parent_spatial_order(owner_map, valid_parent, field_valid):
    """Use actual support, rather than arbitrary slot IDs, to resolve distance ties."""
    batch, rows, columns = owner_map.shape
    flat_owner = owner_map.flatten(1)
    covered = (flat_owner >= 0) & field_valid.flatten(1)
    covered = covered & valid_parent.gather(1, flat_owner.clamp_min(0))
    cells = torch.arange(rows*columns, device=owner_map.device)[None].expand(batch, -1)
    first_cell = torch.full(valid_parent.shape, rows*columns, dtype=torch.long, device=owner_map.device)
    first_cell.scatter_reduce_(1, flat_owner.clamp_min(0), torch.where(covered, cells, rows*columns),
                               reduce="amin", include_self=True)
    return first_cell.argsort(dim=-1, stable=True)


class ParentGraphUpdate(nn.Module):
    """One residual message step between distinct, spatially nearby observed parents."""

    def __init__(self, width, neighbors=8, radius_um=64.0):
        super().__init__()
        self.width, self.neighbors, self.radius_um = width, neighbors, radius_um
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, tokens, valid_parent, centers, spatial_order):
        batch, parents, _ = tokens.shape
        distances = (centers[:, :, None] - centers[:, None]).square().sum(-1)
        eligible = valid_parent[:, :, None] & valid_parent[:, None]
        eligible = eligible & ~torch.eye(parents, dtype=torch.bool, device=tokens.device)[None]
        distances = distances.masked_fill(~eligible, torch.inf)
        ordered = spatial_order[:, None].expand_as(distances)
        rank = distances.gather(-1, ordered).argsort(dim=-1, stable=True)[..., :min(self.neighbors, parents)]
        index = ordered.gather(-1, rank)
        valid = distances.gather(-1, index) <= self.radius_um**2
        batch_index = torch.arange(batch, device=tokens.device)[:, None, None]
        keys = self.key(tokens)[batch_index, index]
        values = self.value(tokens)[batch_index, index]
        logits = (self.query(tokens)[:, :, None] * keys).sum(-1) / math.sqrt(self.width)
        safe = torch.where(valid.any(-1, keepdim=True), logits.masked_fill(~valid, -torch.inf), 0.0)
        attention = torch.softmax(safe, dim=-1) * valid.to(logits.dtype)
        updated = tokens + self.output((attention[..., None] * values).sum(-2))
        return torch.where(valid_parent[..., None], updated, 0.0)


class LocalParentReader(nn.Module):
    def __init__(self, image_channels, parent_channels, context_width=64, neighbors=8,
                 radius_um=64.0, query_chunk_size=256, heads=1, parent_graph=False,
                 parent_image_context=True, image_guided_read=True,
                 position_scale_um=None, weighting="attention"):
        super().__init__()
        if min(image_channels, parent_channels, context_width, neighbors, query_chunk_size) <= 0 or (
                radius_um is not None and (not math.isfinite(radius_um) or radius_um <= 0)):
            raise ValueError("local parent reader dimensions and radius must be positive")
        position_scale_um = radius_um if position_scale_um is None else position_scale_um
        if position_scale_um is None or not math.isfinite(position_scale_um) or position_scale_um <= 0:
            raise ValueError("an explicit positive position scale is required without a radius cutoff")
        if weighting not in ("attention", "uniform") or (parent_graph and radius_um is None):
            raise ValueError("invalid context weighting or unbounded parent graph")
        self.context_width = int(context_width)
        self.parent_image_context = bool(parent_image_context)
        self.image_guided_read = bool(image_guided_read)
        if int(heads) != heads or heads < 1 or self.context_width % heads:
            raise ValueError("context heads must divide the fixed context width")
        self.heads = int(heads)
        self.neighbors = int(neighbors)
        self.radius_um = None if radius_um is None else float(radius_um)
        self.position_scale_um = float(position_scale_um)
        self.weighting = weighting
        self.query_chunk_size = int(query_chunk_size)
        token_input = int(image_channels + parent_channels + 7)
        self.token_fusion = nn.Sequential(nn.LayerNorm(token_input), nn.Linear(token_input, self.context_width),
                                          nn.GELU(), nn.Linear(self.context_width, self.context_width),
                                          nn.LayerNorm(self.context_width))
        self.query = nn.Sequential(nn.LayerNorm(image_channels+2), nn.Linear(image_channels+2, self.context_width))
        self.key = nn.Linear(self.context_width, self.context_width, bias=False)
        self.value = nn.Linear(self.context_width, self.context_width, bias=False)
        self.relative_bias = nn.Sequential(nn.Linear(3, max(8, self.context_width//2)), nn.GELU(),
                                           nn.Linear(max(8, self.context_width//2), 1))
        self.graph = ParentGraphUpdate(self.context_width, neighbors, radius_um) if parent_graph else None

    def forward(self, *, image_features, expression_features, parent_density, owner_map,
                parent_valid, field_valid, geometry, cell_um=2.0, compute_density=True,
                density_mode="attention", interpolation_power=2.0, interpolation_epsilon_um=1.0):
        if density_mode not in ("attention", "inverse_distance") or not math.isfinite(interpolation_power) or interpolation_power <= 0 or not math.isfinite(interpolation_epsilon_um) or interpolation_epsilon_um <= 0:
            raise ValueError("invalid parent density interpolation settings")
        batch, channels, rows, columns = image_features.shape
        parents = parent_valid.shape[1]
        if rows % 4 or columns % 4 or parents < 1 or cell_um <= 0:
            raise ValueError("local queries require nonempty parent slots and complete 4x4 grid groups")
        if tuple(owner_map.shape) != (batch, rows, columns) or field_valid.shape != owner_map.shape:
            raise ValueError("local reader image and ownership grids differ")
        pooled, counts = pool_parent_image(image_features, owner_map, parent_valid, field_valid)
        if not self.parent_image_context:
            pooled = torch.zeros_like(pooled)
        valid_parent = parent_valid & (counts > 0)
        safe_expression = torch.where(valid_parent[..., None], expression_features, 0.0).to(image_features.dtype)
        safe_geometry = torch.where(valid_parent[..., None], geometry.features, 0.0).to(image_features.dtype)
        tokens = self.token_fusion(torch.cat((safe_expression, pooled, safe_geometry), dim=-1))
        tokens = torch.where(valid_parent[..., None], tokens, 0.0)
        centers = torch.where(valid_parent[..., None], geometry.centers_um, 0.0).float()
        spatial_order = parent_spatial_order(owner_map, valid_parent, field_valid)
        if self.graph is not None:
            tokens = self.graph(tokens, valid_parent, centers, spatial_order)
        batch_index = torch.arange(batch, device=image_features.device)
        owner_context = tokens[batch_index[:, None, None], owner_map.clamp_min(0)]
        owner_context = torch.where(((owner_map >= 0) & field_valid)[..., None], owner_context, 0.0)

        safe_image = torch.where(field_valid[:, None], image_features, 0.0)
        query_fraction = F.avg_pool2d(field_valid[:, None].to(image_features.dtype), 4, 4)
        query_image = F.avg_pool2d(safe_image, 4, 4) / query_fraction.clamp_min(1/16)
        query_valid = (query_fraction[:, 0] > 0).flatten(1)
        query_image = query_image.permute(0, 2, 3, 1).flatten(1, 2)
        if not self.image_guided_read:
            query_image = torch.zeros_like(query_image)
        query_rows, query_columns = rows//4, columns//4
        # ParentGeometry stores (y,x) in um relative to the FOV center.
        yy = (torch.arange(query_rows, device=image_features.device, dtype=torch.float32)*4+2)*cell_um - rows*cell_um/2
        xx = (torch.arange(query_columns, device=image_features.device, dtype=torch.float32)*4+2)*cell_um - columns*cell_um/2
        query_centers = torch.stack(torch.meshgrid(yy, xx, indexing="ij"), dim=-1).reshape(-1, 2)
        positions = (query_centers/self.position_scale_um).to(image_features.dtype)[None].expand(batch, -1, -1)
        queries = self.query(torch.cat((query_image, positions), dim=-1))
        keys, values = self.key(tokens), self.value(tokens)
        log_area = torch.log1p(torch.where(valid_parent, geometry.area_children, 0).to(image_features.dtype))
        density = torch.where(valid_parent[..., None], parent_density, 0.0) if compute_density else None
        k = min(self.neighbors, parents)
        contexts, densities, supports, support_features = [], [], [], []
        for start in range(0, query_centers.shape[0], self.query_chunk_size):
            stop = start+self.query_chunk_size
            offsets = centers[:, None] - query_centers[None, start:stop, None]
            distances = offsets.square().sum(-1).masked_fill(~valid_parent[:, None], torch.inf)
            ordered_slots = spatial_order[:, None].expand_as(distances)
            rank = distances.gather(-1, ordered_slots).argsort(dim=-1, stable=True)[..., :k]
            index = ordered_slots.gather(-1, rank)
            distance = distances.gather(-1, index)
            valid = torch.isfinite(distance) & query_valid[:, start:stop, None]
            if self.radius_um is not None:
                valid = valid & (distance <= self.radius_um**2)
            support = valid.any(-1)
            selected_keys = keys[batch_index[:, None, None], index]
            selected_values = values[batch_index[:, None, None], index]
            relative = centers[batch_index[:, None, None], index] - query_centers[None, start:stop, None]
            relative = torch.cat((relative[..., [1, 0]]/self.position_scale_um,
                                  log_area[batch_index[:, None, None], index, None]), dim=-1)
            relative_bias = self.relative_bias(relative.to(queries.dtype)).squeeze(-1)
            if self.heads == 1 or compute_density:
                # Keep the original one-head density weights for the fixed prior semantics.
                logits = (queries[:, start:stop, None]*selected_keys).sum(-1) / math.sqrt(self.context_width)
                logits = logits + relative_bias
                masked_logits = logits.masked_fill(~valid, -torch.inf)
                safe_logits = torch.where(support[..., None], masked_logits, 0.0)
                attention = torch.softmax(safe_logits, dim=-1) * valid.to(logits.dtype)
            if self.weighting == "uniform":
                context_weights = valid.to(selected_values.dtype) / valid.sum(-1, keepdim=True).clamp_min(1)
                contexts.append((context_weights[..., None]*selected_values).sum(-2))
            elif self.heads == 1:
                contexts.append((attention[..., None]*selected_values).sum(-2))
            else:
                head_width = self.context_width // self.heads
                q = queries[:, start:stop].unflatten(-1, (self.heads, head_width))
                keys_by_head = selected_keys.unflatten(-1, (self.heads, head_width))
                logits = (q[:, :, None] * keys_by_head).sum(-1).transpose(-1, -2) / math.sqrt(head_width)
                logits = logits + relative_bias[:, :, None]
                safe = torch.where(support[..., None, None], logits.masked_fill(~valid[:, :, None], -torch.inf), 0.0)
                head_attention = torch.softmax(safe, dim=-1) * valid[:, :, None].to(logits.dtype)
                values_by_head = selected_values.unflatten(-1, (self.heads, head_width)).transpose(-2, -3)
                contexts.append((head_attention[..., None] * values_by_head).sum(-2).flatten(-2))
            # Only the current 8um chunk materializes [B,chunk,K,G].
            if compute_density:
                selected_density = density[batch_index[:, None, None], index]
                if density_mode == "inverse_distance":
                    weights = distance.double().clamp_min(interpolation_epsilon_um**2).pow(-interpolation_power / 2)
                    weights = weights * valid
                    weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
                else:
                    weights = attention.to(density.dtype)
                densities.append((weights[..., None]*selected_density).sum(-2))
            supports.append(support)
            closest = torch.where(support, distance[..., 0], 0.0).sqrt()/self.position_scale_um
            support_features.append(torch.stack((valid.sum(-1).to(closest.dtype)/self.neighbors, closest), dim=-1))
        local_context = torch.cat(contexts, dim=1).reshape(batch, query_rows, query_columns, self.context_width)
        local_context = local_context.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)
        local_context = torch.where(field_valid[..., None], local_context, 0.0)
        return dict(owner_context=owner_context, local_context=local_context,
                    local_density=torch.cat(densities, dim=1).reshape(batch, query_rows, query_columns, -1) if compute_density else None,
                    local_support=torch.cat(supports, dim=1).reshape(batch, query_rows, query_columns),
                    support_features=torch.cat(support_features, dim=1).reshape(batch, query_rows, query_columns, 2),
                    parent_tokens=tokens)


__all__ = ["LocalParentReader", "pool_parent_image"]
