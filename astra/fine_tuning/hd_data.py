from dataclasses import dataclass
import torch
from astra.fine_tuning.observations import build_target_observations, target_observation_mask

@dataclass(frozen=True)
class TargetBatch:
    """T has measured 16um counts and images; no high-resolution label field.

    observed_16um identifies available observations, not nonzero expression.
    A caller must split adaptation/validation by nonoverlapping spatial regions
    before constructing batches. Adaptation inputs must not be filled from
    native fine expression. A held-out benchmark may separately simulate 16um
    observations by sparse aggregation after fitting and selection are frozen.
    """

    counts_16um: torch.Tensor
    image_features_2um: torch.Tensor
    field_valid: torch.Tensor
    protocol_id: object
    gene_available: torch.Tensor | None = None
    observed_16um: torch.Tensor | None = None
    pathology_features: torch.Tensor | None = None
    pathology_valid: torch.Tensor | None = None

    def model_inputs(self, *, parent_bin_um=16):
        inputs = build_target_observations(
            self.counts_16um, self.field_valid, parent_bin_um=parent_bin_um,
            gene_available=self.gene_available, observed_16um=self.observed_16um)
        inputs.update(image_features_2um=self.image_features_2um, protocol_id=self.protocol_id,
                      pathology_features=self.pathology_features, pathology_valid=self.pathology_valid)
        return inputs

    @property
    def observation_mask(self):
        return target_observation_mask(self.counts_16um, self.field_valid, self.observed_16um)
