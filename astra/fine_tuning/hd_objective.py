import torch
from astra.fine_tuning.hd_losses import source_allocation_kl, target_self_supervision
from astra.fine_tuning.hd_data import TargetBatch
from astra.fine_tuning.geometry import canonical_hd16_owner

def target_from_packet(batch, available, protocol, semantic):
    rows, columns = batch.field_valid.shape[-2:]
    expected = canonical_hd16_owner(field_shape=(rows, columns), device=batch.owner_map.device).owner_map
    if not torch.equal(batch.owner_map, expected[None].expand_as(batch.owner_map)):
        raise ValueError("target16 observations must follow the canonical measured grid")
    counts = batch.parent_counts.detach().reshape(len(protocol), rows // 8, columns // 8, -1)
    features, valid = (None, None) if semantic is None else semantic
    # Native cache aggregation simulates the measured input. Fine-label fields
    # are deliberately never accessed by this boundary or the objective.
    return TargetBatch(counts, batch.image_features_2um, batch.field_valid, protocol,
        gene_available=available, observed_16um=getattr(batch, 'observed_16um', None),
        pathology_features=features, pathology_valid=valid)


def objective(model, target, *, supervision=None):
    if not isinstance(target, TargetBatch):
        raise TypeError("target16 fitting requires a TargetBatch with no fine labels")
    prediction = model(**target.model_inputs(parent_bin_um=32))
    loss = target_self_supervision(prediction, target.counts_16um,
        gene_available=target.gene_available, observed_16um=target.observed_16um,
        output_gene_indices=model.output_gene_indices, spatial_mask_2um=supervision)
    return prediction, loss


def preservation_objective(model, teacher, target, *, supervision=None):
    if not isinstance(target, TargetBatch):
        raise TypeError("source preservation requires a TargetBatch with no fine labels")
    inputs = target.model_inputs(parent_bin_um=16)
    with torch.no_grad():
        reference = teacher(**inputs)["pred_count_8um"]
    prediction = model(**inputs)["pred_count_8um"]
    return source_allocation_kl(prediction, reference, target.counts_16um,
        field_valid=target.field_valid, gene_available=target.gene_available,
        observed_16um=target.observed_16um, output_gene_indices=model.output_gene_indices,
        spatial_mask_2um=supervision)
