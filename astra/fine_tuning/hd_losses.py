import torch
from astra.fine_tuning.observations import target_adaptation_mask, target_observation_mask

def _select_genes(value, output_gene_indices):
    if output_gene_indices is None:
        return value
    index = torch.as_tensor(output_gene_indices, device=value.device)
    if index.ndim != 1 or index.dtype != torch.long or not index.numel():
        raise ValueError("output_gene_indices must be nonempty int64 indices in input-panel order")
    if bool((index < 0).any()) or bool((index >= value.shape[-1]).any()) or index.unique().numel() != index.numel():
        raise ValueError("output gene indices must be unique and within the supplied input panel")
    return value.index_select(-1, index)


def _availability(gene_available, prediction, output_gene_indices):
    shape = (prediction.shape[0], prediction.shape[-1])
    if gene_available is None:
        return torch.ones(shape, dtype=torch.bool, device=prediction.device)
    if gene_available.dtype != torch.bool or gene_available.device != prediction.device:
        raise ValueError("gene_available must be boolean and on the prediction device")
    available = _select_genes(gene_available, output_gene_indices)
    if available.shape not in ((shape[-1],), shape):
        raise ValueError("gene availability does not match the output panel")
    return torch.broadcast_to(available, shape)


def _poisson_per_umi(prediction, target, active, available):
    """Mean FOV NLL / max(observed UMIs,1), omitting the label-only constant.

    Zero-count FOVs still penalize all predicted mass; they are not discarded.
    Unmeasured genes and spatial positions are removed before arithmetic.
    """
    if target.shape != prediction.shape or target.device != prediction.device:
        raise ValueError("prediction and target must have matching spatial grids, gene order and device")
    keep = active[..., None] & available[:, None, None]
    truth = torch.where(keep, target.detach(), 0).double()
    pred = torch.where(keep, prediction, 0).double()
    if (not bool(torch.isfinite(truth).all() & torch.isfinite(pred).all())
            or bool((truth < 0).any() | (pred < 0).any())):
        raise ValueError("scored counts and predictions must be finite and nonnegative")
    terms = pred - truth * pred.clamp_min(1e-12).log()
    return (terms.sum((1, 2, 3)) / truth.sum((1, 2, 3)).clamp_min(1)).mean()


def target_self_supervision(result, target_counts_16um, *, gene_available=None,
                            observed_16um=None, output_gene_indices=None, spatial_mask_2um=None):
    """Score T's original 16um observations after a 32um-input presentation.

    There is deliberately no target 8um/2um argument. Aggregation uses the
    coarse output, so this objective cannot train fine-only allocation weights.
    observed_16um selects known training/validation labels, never measured zeroes.
    An optional spatial mask scores only complete 16um cells without changing observations.
    """
    prediction8 = result["pred_count_8um"]
    valid = result["segment_layout"].cell_to_segment >= 0
    active = target_adaptation_mask(target_counts_16um, valid, observed_16um)
    batch, rows, columns, _ = target_counts_16um.shape
    if spatial_mask_2um is not None:
        if (not isinstance(spatial_mask_2um, torch.Tensor)
                or spatial_mask_2um.shape != valid.shape or spatial_mask_2um.dtype != torch.bool
                or spatial_mask_2um.device != valid.device):
            raise ValueError("spatial_mask_2um must be boolean BxHxW on the layout device")
        active = active & spatial_mask_2um.reshape(batch, rows, 8, columns, 8).all(dim=(2, 4))
    if prediction8.shape[:3] != (batch, rows * 2, columns * 2):
        raise ValueError("target self-supervision requires aligned 8um predictions and native 16um observations")
    prediction16 = prediction8.reshape(batch, rows, 2, columns, 2, prediction8.shape[-1]).sum((2, 4))
    target16 = _select_genes(target_counts_16um, output_gene_indices)
    available = _availability(gene_available, prediction8, output_gene_indices)
    if not bool((active[..., None] & available[:, None, None]).any()):
        raise ValueError("target adaptation needs available output genes and a 32um group with at least two observed 16um bins")
    return _poisson_per_umi(prediction16, target16, active, available)


def source_allocation_kl(prediction8, source_prediction8, target_counts_16um, *, field_valid,
                         gene_available=None, observed_16um=None, output_gene_indices=None,
                         spatial_mask_2um=None):
    """Preserve source 16-to-8 allocations; the only observed counts are 16um.

    With per-parent count conservation, the count Poisson divergence equals
    parent-UMI-weighted conditional KL. Complete observed 16um parents keep
    normalization identical to their measured totals, including zero UMIs.
    """
    if source_prediction8.requires_grad:
        raise ValueError("source predictions must come from a frozen teacher")
    if source_prediction8.shape != prediction8.shape or source_prediction8.device != prediction8.device:
        raise ValueError("source and student predictions must share output genes, grid and device")
    batch, rows, columns, _ = target_counts_16um.shape
    if prediction8.shape[:3] != (batch, rows * 2, columns * 2):
        raise ValueError("source preservation requires the observed16/predicted8 aligned grids")
    active16 = target_observation_mask(target_counts_16um, field_valid, observed_16um)
    if spatial_mask_2um is not None:
        if (spatial_mask_2um.dtype != torch.bool or spatial_mask_2um.shape != field_valid.shape
                or spatial_mask_2um.device != field_valid.device):
            raise ValueError("source preservation mask must follow the native boolean spatial grid")
        active16 = active16 & spatial_mask_2um.reshape(batch, rows, 8, columns, 8).all((2, 4))
    available = _availability(gene_available, prediction8, output_gene_indices)
    active8 = active16.repeat_interleave(2, 1).repeat_interleave(2, 2)
    keep = active8[..., None] & available[:, None, None]
    teacher = torch.where(keep, source_prediction8, 0).double()
    student = torch.where(keep, prediction8, 0).double()
    if (not bool(torch.isfinite(teacher).all() & torch.isfinite(student).all())
            or bool((teacher < 0).any() | (student < 0).any())):
        raise ValueError("scored source and student predictions must be finite and nonnegative")
    observed = _select_genes(target_counts_16um, output_gene_indices)
    observed = torch.where(active16[..., None] & available[:, None, None], observed, 0).double()
    if not bool(torch.isfinite(observed).all()) or bool((observed < 0).any()):
        raise ValueError("scored 16um observations must be finite and nonnegative")
    umis = observed.sum((1, 2, 3))
    terms = teacher * (teacher.clamp_min(1e-12).log() - student.clamp_min(1e-12).log()) + student - teacher
    return (terms.sum((1, 2, 3)) / umis.clamp_min(1)).mean()
