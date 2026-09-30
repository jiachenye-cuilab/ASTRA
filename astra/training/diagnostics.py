"""Fixed spatial-monitor identities and HD16 within-parent allocation diagnostics.

The residual PCC follows the flattened parent/child/measured-gene definition
used in the earlier diagnostic reports. Only complete, scored 16um parents
are included. Both prediction and truth subtract the observed parent C/4;
gap density and undefined correlations are never interpreted as zero PCC.
"""

import math

import torch

from astra.training.allocation_metrics import _flat_pcc, decompose


def monitor_items(fields, config, *, smoke=False):
    """Select original reserved cache rows using geometry-only split metadata."""
    settings = config.get("diagnostics", {})
    if not isinstance(settings, dict) or type(settings.get("enabled", False)) is not bool:
        raise ValueError("diagnostics.enabled must be boolean")
    if not settings.get("enabled", False):
        return []
    grouped = {}
    for item in fields.spatial_monitor_items():
        grouped.setdefault(item["sample"], []).append(item)
    if smoke:
        sections = settings.get("smoke_monitor_sections", [])
        limit = settings.get("smoke_monitor_fields_per_section", 1)
        if not isinstance(sections, list) or not sections or len(set(sections)) != len(sections):
            raise ValueError("smoke_monitor_sections must explicitly name unique loaded training sections")
    else:
        sections = list(grouped)
        limit = settings.get("monitor_fields_per_section", 8)
    if type(limit) is not int or limit <= 0:
        raise ValueError("monitor fields per section must be positive integers")
    if not sections or any(rid not in grouped for rid in sections):
        raise PermissionError("requested monitor section lies outside loaded authorized training resources")
    if any(len(grouped[rid]) < limit for rid in sections):
        raise ValueError("requested monitor exposure exceeds its frozen reserved FOVs")
    return [dict(item) for rid in sections for item in grouped[rid][:limit]]


def _children(value):
    batch, rows, columns, genes = value.shape
    return value.reshape(batch, rows // 2, 2, columns // 2, 2, genes).permute(
        0, 1, 3, 2, 4, 5).reshape(batch, (rows // 2) * (columns // 2), 4, genes)


def _count_stratum_metrics(total=None, prediction=None, truth=None):
    """Condition on observed parent/gene totals; compare allocation with C/4."""
    result = dict(zero_parent_gene_pairs=0, zero_parent_gene_fraction=None)
    if total is not None and total.numel():
        result.update(zero_parent_gene_pairs=int((total == 0).sum()),
                      zero_parent_gene_fraction=float((total == 0).double().mean()))
    for name in ("C1", "C2_4", "Cge5"):
        values = dict(parent_gene_pairs=0, umis=0., conditional_nll_per_umi=None,
                      nll_gain_over_uniform=None, parent_residual_pcc=None, residual_rms_ratio=None)
        if total is not None:
            mask = (total == 1 if name == "C1" else (total >= 2) & (total <= 4) if name == "C2_4" else total >= 5)
            values["parent_gene_pairs"] = int(mask.sum())
            if bool(mask.any()):
                c = total[mask]
                x, y = prediction.transpose(1, 2)[mask], truth.transpose(1, 2)[mask]
                umi = c.sum()
                # Match the original count-NLL log floor before conditioning on C.
                nll = -(y * (x.clamp_min(1e-12).log() - c[:, None].log())).sum() / umi
                x, y = x - c[:, None] / 4, y - c[:, None] / 4
                energy = y.square().sum()
                values.update(umis=float(umi), conditional_nll_per_umi=float(nll),
                    nll_gain_over_uniform=math.log(4) - float(nll), parent_residual_pcc=_flat_pcc(x, y),
                    residual_rms_ratio=float((x.square().sum() / energy).sqrt()) if float(energy) > 1e-20 else None)
        result.update({f"{name}_{key}": value for key, value in values.items()})
    return result


@torch.no_grad()
def diagnostic_metrics(prediction, batch, available, supervision, *, output_gene_indices=None, gene_groups=None,
                       parent_count_strata=False, allocation_decomposition=False,
                       gene_group_allocation_decomposition=False):
    """Return one diagnostic record per FOV, using the caller's fixed core mask.

    ``available`` follows the input panel. For a smaller/reordered output panel,
    supply its explicit indices into parent_counts/input availability. Spatial
    scoring requires all 64 native cells of a canonical 16um parent to be valid,
    supervised and owned by that parent; its four 8um children are then scored.
    Valid gap pixels are excluded. Noncanonical covered ownership is rejected.

    PCC flattens all retained parents x four children x measured output genes.
    ``residual_rms_ratio`` is prediction RMS / truth RMS on exactly that support;
    it measures allocation amplitude separately from correlation. Empty support
    or zero truth RMS yields None, while a uniform prediction has ratio 0 and
    undefined PCC. Optional group masks follow the output panel and must be
    predefined from training statistics. Count strata use the observed parent
    total for each gene, never child placement or prediction. No prediction-dependent selection occurs.
    """
    if isinstance(prediction, dict):
        prediction = prediction["pred_count_8um"]
    truth, counts, owner, valid = (batch.target_count_8um, batch.parent_counts,
                                    batch.owner_map, batch.field_valid)
    if (prediction.ndim != 4 or prediction.shape != truth.shape or owner.ndim != 3
            or tuple(owner.shape) != tuple(valid.shape) or valid.dtype != torch.bool
            or owner.dtype != torch.long or supervision.shape != valid.shape
            or supervision.dtype != torch.bool or any(n % 8 for n in owner.shape[1:])
            or prediction.shape[:3] != (owner.shape[0], owner.shape[1] // 4, owner.shape[2] // 4)
            or any(value.device != prediction.device for value in (truth, counts, owner, valid, supervision))):
        raise ValueError("HD16 diagnostic prediction, labels, masks and grids differ")
    n, rows, columns = owner.shape
    parents, genes = (rows // 8) * (columns // 8), prediction.shape[-1]
    if (counts.ndim != 3 or counts.shape[0] != n or counts.shape[1] < parents
            or batch.parent_valid.shape != counts.shape[:2] or batch.parent_valid.dtype != torch.bool
            or batch.parent_valid.device != prediction.device):
        raise ValueError("HD16 diagnostic parent observations differ")
    if output_gene_indices is None:
        if counts.shape[-1] != genes:
            raise ValueError("explicit output_gene_indices are required for a different output panel")
        index = torch.arange(genes, device=prediction.device)
    else:
        index = torch.as_tensor(output_gene_indices, device=prediction.device)
    if (index.dtype != torch.long or index.shape != (genes,) or index.unique().numel() != genes
            or bool((index < 0).any()) or bool((index >= counts.shape[-1]).any())):
        raise ValueError("output gene indices must preserve the explicit output panel order")
    if (available.dtype != torch.bool or available.device != prediction.device
            or available.shape not in ((counts.shape[-1],), (n, counts.shape[-1]))):
        raise ValueError("diagnostic availability must follow the input gene panel")
    measured = torch.broadcast_to(available, (n, counts.shape[-1])).index_select(-1, index)
    gene_groups = {} if gene_groups is None else gene_groups
    for mask in gene_groups.values():
        if mask.shape != (genes,) or mask.dtype != torch.bool or mask.device != prediction.device:
            raise ValueError("diagnostic group masks must follow the output panel")
    coarse = counts[:, :parents].index_select(-1, index)
    expected = torch.arange(parents, device=owner.device).reshape(rows // 8, columns // 8)
    expected = expected.repeat_interleave(8, 0).repeat_interleave(8, 1)[None]
    if bool(((owner >= 0) & (owner != expected)).any()) or bool((owner < -1).any()):
        raise ValueError("parent-residual PCC requires canonical HD16 ownership, not Spot55 or random parents")
    eligible = valid & supervision & (owner == expected)
    complete = eligible.reshape(n, rows // 8, 8, columns // 8, 8).all((2, 4)).reshape(n, parents)
    complete &= batch.parent_valid[:, :parents]
    predicted_children, true_children = _children(prediction), _children(truth)
    result = []
    decomposition_keys = ("full_gain", "total_gain", "composition_gain", "decomposition_error",
        "total_residual_pcc", "total_residual_rms_ratio", "composition_residual_pcc",
        "composition_residual_rms_ratio")
    for fov in range(n):
        keep, measured_genes = complete[fov], measured[fov]
        record = dict(parent_residual_pcc=None, residual_rms_ratio=None,
                      complete_hd16_parents=int(keep.sum()), scored_8um_cells=4 * int(keep.sum()),
                      available_genes=int(measured_genes.sum()))
        if parent_count_strata:
            record.update(_count_stratum_metrics())
        if allocation_decomposition:
            record.update({key: None for key in ("full_gain", "total_gain", "composition_gain",
                "decomposition_error", "total_residual_pcc", "total_residual_rms_ratio",
                "composition_residual_pcc", "composition_residual_rms_ratio")})
        for name, mask in gene_groups.items():
            record.update({f"{name}_parent_residual_pcc": None, f"{name}_residual_rms_ratio": None,
                           f"{name}_available_genes": int((mask & measured_genes).sum())})
            if gene_group_allocation_decomposition:
                record.update({f"{name}_{key}": None for key in decomposition_keys})
        if bool(keep.any()) and bool(measured_genes.any()):
            total = coarse[fov, keep][:, measured_genes].double()
            x = predicted_children[fov, keep][:, :, measured_genes].double()
            y = true_children[fov, keep][:, :, measured_genes].double()
            if any(not bool(torch.isfinite(value).all()) or bool((value < 0).any()) for value in (total, x, y)):
                raise ValueError("scored HD16 observations, labels and predictions must be finite and nonnegative")
            for value in (x, y):
                if not torch.allclose(value.sum(1), total, atol=1e-8, rtol=1e-9):
                    raise ValueError("HD16 diagnostic children do not reaggregate to the observed parent counts")
            if parent_count_strata:
                record.update(_count_stratum_metrics(total, x, y))
            if allocation_decomposition and float(total.sum()) > 0:
                split = decompose(x, y, total)
                record.update({key: split[key] for key in ("full_gain", "total_gain", "composition_gain",
                    "decomposition_error", "total_residual_pcc", "total_residual_rms_ratio",
                    "composition_residual_pcc", "composition_residual_rms_ratio")})
            if gene_group_allocation_decomposition:
                for name, mask in gene_groups.items():
                    selected = mask[measured_genes]
                    group_total = total[:, selected]
                    if float(group_total.sum()) > 0:
                        # Recompute child totals and the UMI denominator inside this
                        # fixed panel so different output heads remain comparable.
                        split = decompose(x[..., selected], y[..., selected], group_total)
                        record.update({f"{name}_{key}": split[key] for key in decomposition_keys})
            x, y = x - total[:, None] / 4, y - total[:, None] / 4
            record["parent_residual_pcc"] = _flat_pcc(x, y)
            truth_energy = y.square().sum()
            if float(truth_energy) > 1e-20:
                record["residual_rms_ratio"] = float((x.square().sum() / truth_energy).sqrt())
            for name, mask in gene_groups.items():
                selected = mask[measured_genes]
                gx, gy = x[..., selected], y[..., selected]
                record[f"{name}_parent_residual_pcc"] = _flat_pcc(gx, gy)
                energy = gy.square().sum()
                if float(energy) > 1e-20:
                    record[f"{name}_residual_rms_ratio"] = float((gx.square().sum() / energy).sqrt())
        result.append(record)
    return result
