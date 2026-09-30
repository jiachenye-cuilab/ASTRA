import torch
from astra.fine_tuning.geometry import canonical_hd16_owner
from astra.model.ownership import validate_owner_batch

def target_observation_mask(counts_16um, field_valid, observed_16um=None):
    """Keep observed 16um bins with complete spatial support.

    A partial 16um bin is excluded: its measured total cannot be attributed to
    only the remaining valid 2um cells without additional expression data.
    Missing bins may contain arbitrary placeholders, including NaNs.
    """
    if counts_16um.ndim != 4 or min(counts_16um.shape) < 1:
        raise ValueError("counts_16um must be nonempty [B,Hy,Hx,G]")
    batch, rows, columns, _ = counts_16um.shape
    if (field_valid.dtype != torch.bool or field_valid.device != counts_16um.device
            or field_valid.shape != (batch, rows * 8, columns * 8)):
        raise ValueError("field_valid must be boolean [B,Hy*8,Hx*8] on the counts device")
    complete = field_valid.reshape(batch, rows, 8, columns, 8).all(dim=(2, 4))
    if observed_16um is None:
        return complete
    if (observed_16um.dtype != torch.bool or observed_16um.device != counts_16um.device
            or observed_16um.shape != (batch, rows, columns)):
        raise ValueError("observed_16um must be boolean [B,Hy,Hx] on the counts device")
    return complete & observed_16um


def build_target_observations(counts_16um, field_valid, *, parent_bin_um=16,
                              gene_available=None, observed_16um=None):
    """Return model observation tensors; never accept or construct target fine data.

    The input origin must coincide with a native 16um grid edge. For 32um
    parents, sum each aligned 2x2 set of *observed* 16um bins. Missing members
    and FOV edges remain absent support, not observed zero counts. Keep the
    original counts_16um separately for the target self-supervision loss.
    The caller supplies H&E/UNI inputs and protocol_id to the model unchanged.
    """
    if parent_bin_um not in (16, 32):
        raise ValueError("parent_bin_um must be 16 or 32")
    parent_bin_um = int(parent_bin_um)
    if counts_16um.dtype == torch.bool or counts_16um.is_complex():
        raise TypeError("counts_16um must contain numeric counts")
    observed = target_observation_mask(counts_16um, field_valid, observed_16um)
    batch, rows, columns, genes = counts_16um.shape
    if gene_available is None:
        available = torch.ones((batch, genes), dtype=torch.bool, device=counts_16um.device)
    else:
        if (gene_available.dtype != torch.bool or gene_available.device != counts_16um.device
                or gene_available.shape not in ((genes,), (batch, genes))):
            raise ValueError("gene_available must be boolean G or BxG on the counts device")
        available = torch.broadcast_to(gene_available, (batch, genes))
    counts = torch.where(observed[..., None] & available[:, None, None], counts_16um, 0).double()
    if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
        raise ValueError("observed available counts must be finite and nonnegative")

    factor = parent_bin_um // 16
    groups_y, groups_x = (rows + factor - 1) // factor, (columns + factor - 1) // factor
    yy = torch.arange(rows, device=counts.device)[:, None]
    xx = torch.arange(columns, device=counts.device)[None]
    group = ((yy // factor) * groups_x + xx // factor).flatten()
    parents = groups_y * groups_x
    parent_counts = counts.new_zeros((batch, parents, genes))
    parent_counts.scatter_add_(1, group[None, :, None].expand(batch, -1, genes), counts.flatten(1, 2))
    parent_members = torch.zeros((batch, parents), dtype=torch.long, device=counts.device)
    parent_members.scatter_add_(1, group[None].expand(batch, -1), observed.flatten(1).long())
    parent_valid = parent_members > 0

    # Reuse the native owner raster; merge IDs rather than rasterizing expression.
    native = canonical_hd16_owner(field_shape=tuple(field_valid.shape[-2:]), device=counts.device).owner_map
    owner_map = group[native][None].expand(batch, -1, -1)
    support = observed.repeat_interleave(8, 1).repeat_interleave(8, 2)
    owner_map = torch.where(support, owner_map, -1)
    validate_owner_batch(owner_map, parent_valid, field_valid)
    return dict(parent_counts=parent_counts, owner_map=owner_map,
                parent_valid=parent_valid, field_valid=field_valid,
                gene_available=available)


def target_adaptation_mask(counts_16um, field_valid, observed_16um=None):
    """Score only groups containing at least two known 16um members.

    A singleton 32um parent equals its sole 16um label: hard conservation makes
    that supervision an identity, even when the label contains positive counts.
    Singleton observations may still provide model context during adaptation.
    """
    active = target_observation_mask(counts_16um, field_valid, observed_16um)
    batch, rows, columns = active.shape
    padded = active.new_zeros((batch, rows + rows % 2, columns + columns % 2))
    padded[:, :rows, :columns] = active
    members = padded.reshape(batch, padded.shape[1] // 2, 2, padded.shape[2] // 2, 2).sum((2, 4))
    informative = (members >= 2).repeat_interleave(2, 1).repeat_interleave(2, 2)
    return active & informative[:, :rows, :columns]
