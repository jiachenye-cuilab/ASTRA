# Single-field inference

Run commands from the repository root. Each input describes one registered 256 × 256 µm field of view.

## Custom inputs

```bash
python -m astra prepare-input --input observations.npz --output inputs/my_fov.npz
python -m astra predict --input inputs/my_fov.npz --uni-checkpoint /path/to/pytorch_model.bin --output outputs/my_fov.npz
```

`python -m astra prepare-input` accepts one fully registered FOV. The section entry points
handle tiling, feature preparation and stitching from registered section inputs;
image registration and acquisition-specific file conversion remain caller inputs.
Single-FOV fields and coordinate conventions are documented
in [input preparation API](../astra/data/preparation.py). H&E preprocessing uses the original
colors without Reinhard normalization. Integrals of optical density and
hematoxylin/eosin density must be computed from the original pixels, not from
downsampled RGB. Use finite, nonnegative raw integer UMI counts, without CPM/TPM
normalization or a log transform. Integer-valued floating-point storage is
accepted on input; fractional values are rejected, not rounded. Preparation and
loading validate the count range and convert counts to `int64`, matching the
benchmark coarse-count caches. Inference and fine-tuning convert counts to FP32
for model computation and keep predictions in FP32. Counts are reordered using
[model/input_gene_ids.json](../model/input_gene_ids.json), and missing genes are
marked unavailable.

RGB inputs require locally supplied UNI1 weights obtained under their applicable
terms. ASTRA neither distributes nor automatically downloads these weights.
Their identity is checked against [model/metadata.json](../model/metadata.json).
If your NPZ already contains compatible frozen UNI features, omit
`--uni-checkpoint` and provide the following fields:

| Field | Shape and type |
| --- | --- |
| `parent_counts` | `[P,2000]`, int64 raw nonnegative UMI counts; integer-valued floating-point input is converted on load |
| `gene_ids`, `gene_available` | Ordered `[2000]` Unicode strings, `[2000]` bool |
| `owner_map` | `[128,128]` int64; `-1` denotes an unobserved gap |
| `parent_valid`, `field_valid` | `[P]` bool, `[128,128]` bool |
| `protocol_id` | int64 scalar: `0` for 3′, `1` for WT; independent of geometry |
| `image_features_2um` | `[6,128,128]` float32, raw H&E features |
| `pathology_features`, `pathology_valid` | `[1024,14,14]` float32, `[14,14]` bool |
| `uni_checkpoint_sha256` | String scalar matching the model metadata |

Cached features must use the same FOV, registration, and UNI preprocessing.
An identity field alone cannot establish that externally computed features are
correct.

To reuse frozen features across runs, prepare a feature file once:

```bash
python -m astra prepare-features --input inputs/my_fov.npz --uni-checkpoint /path/to/pytorch_model.bin --output inputs/my_fov_features.npz
python -m astra predict --input inputs/my_fov_features.npz --output outputs/my_fov_cached.npz
```

## Outputs and reproducibility

Single-FOV predictions are saved as NPZ files containing `pred_count_8um[32,32,2000]`
(float32), `gene_ids[2000]`, `gene_available[2000]`, and
`field_valid_8um[32,32]`. Gene order is fixed by
[model/output_gene_ids.json](../model/output_gene_ids.json), and row/column
coordinates are local to the FOV. Unavailable genes and invalid regions must not
be interpreted as biological zero expression. Predicted counts are inferred
quantities rather than native measurements.

Count conservation is checked on observation-by-8 µm intersections, with a
maximum scaled residual of `5e-6` for the FP32 model. Complete aligned HD16
exports additionally re-read and check output at `atol=1e-5`, `rtol=1e-5`.
Spot55 boundaries can cross output cells: even correct intersection-area weights
cannot generally recover the discarded within-cell masses from the final grid.
Reports include separate serialized-grid residuals and support counts;
see [the defined operators and limits](conservation.md).
Reference replay uses `atol=1e-6` and `rtol=1e-5`.

The bundled checkpoint has 4,676,971 parameters and was selected at epoch 96
from a run that stopped at epoch 126.
Its parameters, buffers, and ordered gene identities are retained; optimizer and
RNG states are omitted from the inference checkpoint. Model provenance is
recorded in [model/metadata.json](../model/metadata.json).

Run the [portable checks and optional pretrained replay](reproducibility.md)
to verify your installation. The bundled references are model predictions;
their agreement does not measure biological accuracy. Bitwise equivalence
across hardware or recomputed feature caches is not guaranteed.
