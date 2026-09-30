"""Exact native-count aggregation without a dense 2um gene grid."""

from dataclasses import dataclass

import torch

from astra.training.ownership import ParentGeometry, derive_parent_geometry, validate_owner_batch


@dataclass(frozen=True)
class CompactFieldBatch:
    parent_counts: torch.Tensor
    owner_map: torch.Tensor
    parent_valid: torch.Tensor
    parent_geometry: ParentGeometry
    image_features_2um: torch.Tensor
    field_valid: torch.Tensor
    target_count_8um: torch.Tensor
    output_gene_ids: tuple[str, ...]
    presentation_seeds: tuple[int, ...]

    def model_inputs(self):
        return {name: getattr(self, name) for name in (
            "parent_counts", "owner_map", "parent_valid", "parent_geometry",
            "image_features_2um", "field_valid")}


def collate_sparse_counts(indices, counts, image, field_valid, samples, panels):
    """Indices flatten [B,H,W,G_input]; duplicates sum in INT64, including gap targets.

    Only input parent totals and output 8um targets become dense. Target aggregation
    deliberately ignores owner coverage; zero-count cells retain their loss terms.
    """
    if not samples:
        raise ValueError("cannot collate empty sparse batch")
    device = samples[0].owner_map.device
    owner = torch.stack([sample.owner_map for sample in samples])
    batch, rows, cols = owner.shape
    if rows % 4 or cols % 4 or image.shape != (batch, 6, rows, cols) or field_valid.shape != owner.shape:
        raise ValueError("sparse count/image/8um geometry differs")
    parents = max(sample.parent_valid.numel() for sample in samples)
    parent_valid = torch.zeros((batch, parents), dtype=torch.bool, device=device)
    for i, sample in enumerate(samples):
        parent_valid[i, :sample.parent_valid.numel()] = sample.parent_valid
    validate_owner_batch(owner, parent_valid, field_valid, check_connectivity=False)
    indices = torch.as_tensor(indices, device=device)
    counts = torch.as_tensor(counts, device=device)
    genes, output_genes = len(panels.input_gene_ids), len(panels.output_gene_ids)
    if indices.dtype != torch.int64 or counts.dtype not in (torch.int32, torch.int64) or indices.ndim != 1 or counts.shape != indices.shape:
        raise ValueError("sparse indices/counts must be aligned integer vectors")
    if bool((indices < 0).any() | (indices >= batch * rows * cols * genes).any() | (counts < 0).any()):
        raise ValueError("sparse index or count outside allowed range")
    counts = counts.to(torch.int64)
    cells, gene = indices // genes, indices % genes
    fov = cells // (rows * cols)
    parent = owner.reshape(-1)[cells]
    covered = parent >= 0
    parent_counts = torch.zeros(batch * parents * genes, dtype=torch.int64, device=device)
    parent_index = (fov[covered] * parents + parent[covered]) * genes + gene[covered]
    parent_counts.index_add_(0, parent_index, counts[covered])
    mapping = torch.full((genes,), -1, dtype=torch.long, device=device)
    mapping[list(panels.output_indices)] = torch.arange(output_genes, device=device)
    output_gene = mapping[gene]
    selected = output_gene >= 0
    cell8 = (fov * (rows // 4) + (cells % (rows * cols)) // cols // 4) * (cols // 4) + cells % cols // 4
    target = torch.zeros(batch * (rows // 4) * (cols // 4) * output_genes, dtype=torch.int64, device=device)
    target.index_add_(0, cell8[selected] * output_genes + output_gene[selected], counts[selected])
    return CompactFieldBatch(parent_counts.reshape(batch, parents, genes), owner, parent_valid,
        derive_parent_geometry(owner, parent_valid, cell_um=2.0), image, field_valid,
        target.reshape(batch, rows // 4, cols // 4, output_genes), panels.output_gene_ids,
        tuple(int(sample.parameters.get("seed", -1)) for sample in samples))
