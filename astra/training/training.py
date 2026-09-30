"""Base training and evaluation, with independently weighted spatial contributions."""

import math

import torch

from astra.training.heads import aggregate_segments
from astra.training.segment import build_segment_layout


def validate_loss_config(config):
    if config is None:
        return
    if set(config) != {"umi_floor", "parent_kl_coefficient", "parent_kl_direction", "parent_scope"}:
        raise ValueError("unknown training loss setting")
    if config["parent_kl_direction"] != "area_to_prediction" or config["parent_scope"] != "fully_inside_core":
        raise ValueError("unsupported parent regularization semantics")
    for key in ("umi_floor", "parent_kl_coefficient"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("loss coefficients must be finite and nonnegative")


def parent_kl_count_sum(prediction, batch, supervision, available, *, output_gene_indices=None):
    """Sum C_pg KL(area || prediction) for measured, positive, full-core parents."""
    counts = batch.parent_counts.detach().double()
    if output_gene_indices is not None:
        counts = counts[..., list(output_gene_indices)]
    batch_size, parents, genes = counts.shape
    available = torch.broadcast_to(available, (batch_size, genes))
    owner = batch.owner_map
    valid = batch.field_valid & (owner >= 0)
    indices = torch.arange(batch_size, device=owner.device)[:, None, None] * parents + owner.clamp_min(0)
    inside = counts.new_zeros(batch_size * parents)
    inside.index_add_(0, indices[valid], supervision.expand_as(owner)[valid].to(counts.dtype))
    area = batch.parent_geometry.area_children
    eligible = batch.parent_valid & (area > 0) & (inside.reshape(batch_size, parents) == area)
    layout = prediction["segment_layout"]
    b, parent = layout.segment_batch, layout.segment_owner
    keep = (parent >= 0) & eligible[b, parent.clamp_min(0)]
    b, parent = b[keep], parent[keep]
    # Mask before division/log so missing NaN placeholders cannot poison gradients.
    c = torch.where(available[b], counts[b, parent], 0)
    positive = (c > 0) & available[b]
    u = layout.segment_area_children[keep].double() / area[b, parent]
    mass = torch.where(positive, prediction["coarse_segment_mass"][keep], 0).double()
    p = mass / c.clamp_min(1)
    terms = torch.where(positive, c * u[:, None] *
        (u[:, None].log() - p.clamp_min(torch.finfo(p.dtype).tiny).log()), 0.0).sum(-1)
    result = counts.new_zeros(batch_size)
    result.index_add_(0, b, terms)
    return result.clamp_min(0)


def training_objective(losses, prediction, batch, supervision, available, config, *,
                       output_gene_indices=None, region_weights=None):
    """Optional legacy weighting/KL objective; reported NLL remains unweighted."""
    validate_loss_config(config)
    if config is None:
        return losses["optimization_loss"]
    if region_weights is not None and any(value != 1 for value in region_weights.values()):
        raise ValueError("training loss settings cannot be combined with nonunit region weights")
    umi = losses["target_umis"]
    if not bool((torch.isfinite(umi) & (umi >= 0)).all()):
        raise ValueError("scored target UMIs must be finite and nonnegative")
    # A zero-count core is still an observed negative example. With no UMI
    # weighting, retain its Poisson rate term and the original FOV mean.
    weights = (torch.ones_like(umi) if config["umi_floor"] == 0
               else umi / umi.clamp_min(config["umi_floor"]))
    penalty = (parent_kl_count_sum(prediction, batch, supervision, available,
        output_gene_indices=output_gene_indices) / umi.clamp_min(1)
        if config["parent_kl_coefficient"] else torch.zeros_like(umi))
    weighted_data = weights * losses["loss_per_patch"]
    weighted_penalty = weights * penalty
    total = weighted_data + config["parent_kl_coefficient"] * weighted_penalty
    losses.update(optimization_loss=total.mean(), optimization_loss_per_patch=total,
        weighted_data_loss=weighted_data.mean(), weighted_parent_kl=weighted_penalty.mean(),
        mean_fov_weight=weights.mean(), downweighted_fov_fraction=(weights < 1).double().mean(),
        fov_weights=weights, parent_kl_per_patch=penalty)
    return losses["optimization_loss"]


def count_loss(prediction, target, field_valid, supervision, *, gene_available, owner_map=None,
               region_weights=None):
    batch, rows, cols = field_valid.shape
    if (prediction.shape != target.shape or prediction.shape[:3] != (batch, rows // 4, cols // 4)
            or supervision.shape != field_valid.shape or supervision.dtype != torch.bool):
        raise ValueError("8um prediction, target, validity and supervision grids differ")
    available = torch.broadcast_to(gene_available, (batch, prediction.shape[-1]))
    active = (field_valid & supervision).reshape(batch, rows // 4, 4, cols // 4, 4).all((2, 4))
    keep = active[..., None] & available[:, None, None]
    if not bool(keep.flatten(1).any(1).all()):
        raise ValueError("each FOV requires complete scored cells and available output genes")
    truth = torch.where(keep, target.detach(), 0).double()
    pred = torch.where(keep, prediction, 0).double()
    if (not bool(torch.isfinite(truth).all() & torch.isfinite(pred).all())
            or bool((truth < 0).any() | (pred < 0).any())):
        raise ValueError("scored counts must be finite and nonnegative")
    umis = truth.sum((1, 2, 3))
    terms = pred - truth * pred.clamp_min(1e-12).log()
    loss_per_patch = terms.sum((1, 2, 3)) / umis.clamp_min(1)
    result = dict(loss=loss_per_patch.mean(), loss_per_patch=loss_per_patch, target_umis=umis,
                  primary_8um_nll_per_umi=loss_per_patch)
    weights = dict(covered=1.0, gap=1.0, mixed=1.0)
    if region_weights is not None:
        if set(region_weights) != set(weights) or any(not math.isfinite(v) or v < 0 for v in region_weights.values()):
            raise ValueError("region weights must explicitly cover covered/gap/mixed and be nonnegative")
        weights.update(region_weights)
    if owner_map is not None:
        block = owner_map.reshape(batch, rows // 4, 4, cols // 4, 4)
        covered, gap = (block >= 0).all((2, 4)), (block < 0).all((2, 4))
        regions = dict(covered=covered, gap=gap, mixed=~(covered | gap))
        contributions = {name: torch.where(mask[..., None], terms, 0).sum((1, 2, 3)) / umis.clamp_min(1)
                         for name, mask in regions.items()}
        result["region_contributions"] = contributions
        result["optimization_loss"] = sum(weights[name] * value.mean() for name, value in contributions.items())
    elif region_weights is not None:
        raise ValueError("region weights require the owner map")
    else:
        result["optimization_loss"] = result["loss"]
    return result


def area_prediction8(model, batch, available=None):
    available = (torch.ones((batch.parent_counts.shape[0], model.num_input_genes),
                           dtype=torch.bool, device=batch.parent_counts.device)
                 if available is None else available)
    available = torch.broadcast_to(available, (batch.parent_counts.shape[0], model.num_input_genes))
    safe = torch.where(batch.parent_valid[..., None] & available[:, None], batch.parent_counts, 0)
    counts = model.select_output_genes(safe).double()
    geometry = batch.parent_geometry
    layout = build_segment_layout(batch.owner_map, batch.field_valid,
                                 parent_slots=counts.shape[1], validate=False)
    density = counts / geometry.area_children.clamp_min(1)[..., None]
    global_density = counts.sum(1) / geometry.area_children.sum(1).clamp_min(1)[:, None]
    covered = layout.segment_owner >= 0
    values = global_density[layout.segment_batch].clone()
    values[covered] = density[layout.segment_batch[covered], layout.segment_owner[covered]]
    return aggregate_segments(values * layout.segment_area_children[:, None], layout, counts.shape[0])


def objective(model, batch, available, protocol, semantic, supervision, *, validate=True,
              region_weights=None, loss_config=None):
    validate_loss_config(loss_config)
    if loss_config is not None and region_weights is not None and any(value != 1 for value in region_weights.values()):
        raise ValueError("training loss settings cannot be combined with nonunit region weights")
    if batch.output_gene_ids != model.panels.output_gene_ids:
        raise ValueError("batch target gene identities/order differ from the model")
    features, feature_valid = (None, None) if semantic is None else semantic
    arguments = dict(**batch.model_inputs(), gene_available=available, protocol_id=protocol,
                     pathology_features=features, pathology_valid=feature_valid, validate=validate)
    if hasattr(model, "use_fine_head"):
        arguments["return_fine"] = False
    result = model(**arguments)
    losses = count_loss(result["pred_count_8um"], batch.target_count_8um, batch.field_valid, supervision,
        gene_available=model.select_output_genes(available), owner_map=batch.owner_map, region_weights=region_weights)
    training_objective(losses, result, batch, supervision, model.select_output_genes(available),
        loss_config, output_gene_indices=model.panels.output_indices, region_weights=region_weights)
    return result, losses


def train_step(model, optimizer, batch, available, protocol, semantic, supervision, *,
               max_gradient_norm=1.0, validate=True, loss_weight=1.0, zero_grad=True,
               optimizer_step=True, region_weights=None, loss_config=None):
    if not math.isfinite(loss_weight) or not 0 < loss_weight <= 1:
        raise ValueError("loss_weight must preserve the FOV mean in an accumulated update")
    if max_gradient_norm is not None and (not math.isfinite(max_gradient_norm) or max_gradient_norm <= 0):
        raise ValueError("gradient clipping threshold must be positive")
    if getattr(model, "adaptation_mode", False):
        raise ValueError("use adaptation.train_step for a frozen base with optional components")
    model.train()
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    prediction, losses = objective(model, batch, available, protocol, semantic, supervision,
                                    validate=validate, region_weights=region_weights, loss_config=loss_config)
    (losses["optimization_loss"] * loss_weight).backward()
    losses["gradient_norm_before_clip"] = None
    if optimizer_step:
        losses["gradient_norm_before_clip"] = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            float("inf") if max_gradient_norm is None else max_gradient_norm, error_if_nonfinite=True)
        optimizer.step()
    return prediction, losses


def evaluate_batch(model, batch, available, protocol, semantic, supervision, *, validate=True):
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction, losses = objective(model, batch, available, protocol, semantic, supervision, validate=validate)
            area = count_loss(area_prediction8(model, batch, available), batch.target_count_8um, batch.field_valid, supervision,
                              gene_available=model.select_output_genes(available), owner_map=batch.owner_map)
            return prediction, losses, area
    finally:
        model.train(training)
