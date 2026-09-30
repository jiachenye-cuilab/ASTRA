# Frozen benchmarks, Visium, ST100 and user training

The released ASTRA state dictionary, architecture and ordered 2,000-gene panel
are retained. These adapters use raw counts, original-color H&E, frozen UNI1
and FP32 ASTRA inference. Benchmarks do not optimize weights. The numerical
examples in the quickstart remain separate from measured benchmark references.

Clone the repository and install the section dependencies as in
[README](../README.md). Run from the repository root, edit your own copies of
the templates in `configs/`, and use new output directories. Paths in JSON are
relative to that JSON file.
No dataset or gated UNI weights are downloaded by these commands.

## HD16 and pseudo-Visium benchmarks

Use the native Space Ranger **2 µm `feature_slice.h5`** and the corresponding
registered original-resolution H&E. HDF5 must contain `metadata_json` with
`spot_pitch=2`, `nrows`, `ncols` and
`transform_matrices.spot_colrow_to_microscope_colrow`; `features/id`, assay
metadata/target sets, native `feature_slices/<index>/{row,col,data}`, and
`masks/square_016um/{row,col,data}`. These are raw integer UMI counts. The
registration maps native 2 µm pixel centers to H&E pixel centers. Mixing
Space Ranger processing versions changes that geometry.

| Template | Model observations | Inference | Default measured scoring support |
| --- | --- | --- | --- |
| [benchmark_hd16.json](../configs/benchmark_hd16.json) | Sum 64 native 2 µm cells per measured tissue 16 µm parent | 256 µm FOV; core/stride 160/160 | Four complete 8 µm children of every native tissue/image-valid 16 µm parent |
| [benchmark_pseudo_visium.json](../configs/benchmark_pseudo_visium.json) | Sum native 2 µm centers inside 55 µm circles, 100 µm hexagonal pitch | 256 µm FOV; core/stride 160/144; average covering cores | 8 µm squares completely inside retained continuous 55 µm circles |

Pseudo-Visium uses `lattice_seed=20260907` and a sample-ID-dependent phase in
one primitive lattice cell. Keep the exact resource ID below when reproducing
the paper lattice. Select circles by native tissue-16 µm mask membership of
their centers, complete capture support and a complete registered 56 µm H&E
square. Selection and geometry never depend on gene expression. An 8 µm measured
reference is the sum of its sixteen native 2 µm cells, with no interpolation.

The fixed six tests are:

| Resource ID | `protocol_id` |
| --- | --- |
| `WT_Ovarian_FF_6p5mm` | 1 |
| `HD3_Ovarian_Cancer_FF` | 0 |
| `WT_Breast_FF_Ultima` | 1 |
| `WT_Colorectal_FFPE_6p5mm` | 1 |
| `WT_Colon_FFPE_6p5mm` | 1 |
| `WT_Pancreas_FFPE_6p5mm_v4p0p1` | 1 |

Their official sources and processing versions are in [datasets.md](datasets.md)
and [resource.json](../resource.json). WT Lung is excluded. Repeat the following
for each section with its own edited config and fresh output directory:

```bash
python -m astra prepare-benchmark --config configs/benchmark_hd16.json --output outputs/hd16-data
python -m astra prepare-section --config outputs/hd16-data/observations/section.json --output outputs/hd16-inputs --device cpu
python -m astra benchmark --inputs outputs/hd16-inputs --reference outputs/hd16-data/reference --output outputs/hd16-benchmark --device cpu

python -m astra prepare-benchmark --config configs/benchmark_pseudo_visium.json --output outputs/pseudo-data
python -m astra prepare-section --config outputs/pseudo-data/observations/section.json --output outputs/pseudo-inputs --device cpu
python -m astra benchmark --inputs outputs/pseudo-inputs --reference outputs/pseudo-data/reference --output outputs/pseudo-benchmark --device cpu
```

Use `--device cuda:0` for image preparation/inference on one assigned GPU.
Preparation writes coarse observations in `observations/` and **measured,
evaluation-only** `counts_8um.npy` in `reference/`. The model reads only the
prepared coarse input. Scoring reopens `prediction/prediction.npy` and reference
arrays, aligns coordinates, and requires identical measured gene availability
and order. Missing coordinates/genes are errors; absent assay genes are not
ground-truth zeros. Outputs include per-gene and summary `metrics.json`.

Metrics are gene PCC and independent per-gene min-max nRMSE, position PCC across
measured genes, and composition Bray–Curtis. Constant-gene PCC remains undefined
(`null`); a zero prediction at a nonzero-reference position has Bray–Curtis 1.
Min-max scaling is part of the nRMSE metric, not prediction postprocessing.

For an explicitly frozen evaluation domain, add `evaluation_query_yx_8um` to
the benchmark config: a unique nonnegative integer `[N,2]` NPY in y,x 8 µm
indices. Optional `--position-mask MASK.npy` on `benchmark`/`evaluate` supplies
a boolean mask in **reference coordinate order**. Paper position comparisons
use support shared by all compared methods; the default ASTRA-only score is
labeled accordingly. External methods, their masks, fine-tuned results and
historical runtime tables are separate assets. These commands enable frozen
ASTRA benchmarking; they do not by themselves reproduce every paper panel or
the other methods. Full six-section raw-data/GPU replay is outside the portable
CPU verification scope.

```bash
python -m astra evaluate --prediction outputs/hd16-benchmark/prediction --reference outputs/hd16-data/reference --output outputs/hd16-rescored.json
```

## Real 10x Visium, including ILC-like sections

Edit [section_visium.json](../configs/section_visium.json). Supply filtered 10x HDF5
counts, exact Ensembl feature IDs and a headered Space Ranger positions CSV
with `barcode,in_tissue,pxl_row_in_fullres,pxl_col_in_fullres`. Optional `qc_pass`
is 0/1. The adapter aligns positions to HDF5 barcode order; it retains tissue
and QC-passing observations without normalizing counts. Use protocol 0 for
3′ Visium. Geometry is 55 µm / 100 µm, field256/core160/stride144.

`spot_to_image` must map the explicit physical 2 µm capture frame to the
**same full-resolution image frame used by the CSV**. Its inverse converts
full-resolution pixel centers to capture coordinates, then to physical centers
`(2*row+1,2*col+1)` µm. A matrix for native NDPI cannot be substituted for one
for a rotated/scaled Space Ranger image. Resolved physical registration and any
necessary coordinate conversion must be supplied consistently before this
adapter. Dimensions and DPI do not establish physical calibration. The identity
template is only a placeholder for aligned 2 µm/pixel imagery.

```bash
python -m astra prepare-visium --config configs/section_visium.json --output outputs/visium-observations
python -m astra prepare-section --config outputs/visium-observations/section.json --output outputs/visium-inputs --device cpu
python -m astra predict-section --inputs outputs/visium-inputs --output outputs/visium-prediction --device cpu
```

Default query centers lie within 100 µm of a retained observed spot. Optional
`tissue_mask_2um` requires at least 8/16 tissue subcells in an 8 µm bin. For
paper-specific native all-spot Voronoi/capture support and tissue/QC criteria,
supply the frozen `query_yx_8um` NPY instead of deriving the default support.
The ILC paper protocol additionally uses >200 detected genes and Hole/Artefact/Out
fraction <0.3. Those annotations and the 43-patient support/registration assets
are not supplied or recomputed here. Native NDPI decoding is not claimed;
use a calibrated, pixel-preserving RGB8 tiled TIFF satisfying the
[decoder contract](image-inputs.md), with the corresponding registration.

## Old ST / ST-TNBC: 100 µm circles and nine FOVs

Edit [section_st100.json](../configs/section_st100.json). Convert source matrices
to rows=spots/columns=genes CSR NPZ (or integer NPY), preserve raw counts and
gene order, and save physical `spot_yx_um[N,2]`. RData/RDS, gene-symbol mapping
and author-specific image refinement are acquisition-specific preparation
steps; this package does not silently infer them. Use the same refined original
image registration for counts, centers, tissue mask and H&E.

The adapter implements the existing ST-TNBC protocol: 100 µm capture diameter,
150 µm nominal pitch, 3′ protocol 0, frozen raw ASTRA. Each spot center is rounded
to the nearest multiple of 8 µm for **FOV placement only**; the observation
operator still uses the supplied unrounded center. Each anchor has one central
256 µm FOV and eight supplement FOVs at x/y offsets −24/0/+24 µm. Every field
conditions on that one complete 100 µm anchor; partial neighboring circles
remain unsupervised. All nine fields need full capture and registered H&E
support. Pad the physical frame and update centers/registration together when
appropriate; the adapter does not invent padding or remove edge anchors.

```bash
python -m astra prepare-section --config configs/section_st100.json --output outputs/st100-inputs --device cpu
python -m astra predict-section --inputs outputs/st100-inputs --output outputs/st100-prediction --device cpu
```

The direct output is `[N,2000]` raw inferred counts on the union of all central
160 µm cores, with equal means of **all** covering fields. `query_yx_8um.npy`,
`xy_um.npy`, `query_coverage.npy`, gene availability, `query_parent.npy` and
`query_spot_area_2um.npy` accompany it. Parent −1 is outside capture circles;
−2 denotes ambiguous circle ownership. Area is the number of in-circle native
2 µm centers (0–16), not continuous circle/square intersection area. Primary
complete-circle analysis also requires that the farthest square corner lie
inside the continuous circle; area=16 alone does not establish that condition.

Optional `tissue_mask_8um` is a boolean NPY in the complete capture frame. It
enables the paper's four-neighbor inverse-distance-squared gap interpolation
up to 32 µm, retaining only tissue paths sampled at 1/4,1/2,3/4. Interpolated
counts and coordinates are separate files; `prediction_source.npy` labels
0=unsupported, 1=direct, 2=interpolated. Without a tissue mask no gap interpolation
occurs. Exclude interpolated counts from primary direct-bin analyses.

Single-field allocation preserves each anchor count over its discretized
circle. The final averaged grid generally does **not** preserve that spot sum.
`report.json` separately reports the single-FOV scaled residual and a diagnostic
reaggregation of the **saved** direct grid using raster-area fractions. It is
an approximate grid representation, not a rescaled prediction or exact
continuous-circle integral. [Conservation scopes](conservation.md) apply with
100 µm replacing 55 µm. The per-FOV scaled FP32 threshold is `5e-6`, retained
from the current cohort FP32 implementation. The independent saved-mean test
uses the cohort's `atol=1e-5` UMI and `rtol=3e-6`: at most nine FP32 additions
give a relative summation bound of about `9.54e-7`, with the remaining margin
covering the existing FP32 forward/batching comparison. These tolerances do
not make a final-grid conservation assertion. There is no ST100 fine-tuning
protocol in this release.
Synthetic geometry/cache/FP32/export tests verify this adapter; raw UNI,
original cohort files and the 92-patient downstream analysis have not been rerun.

## Compatible raw-image cache

Section preparation can use `image_cache` instead of local UNI extraction.
Its `images.json` must declare `preprocessing_mode=raw`, the published
`uni_checkpoint_sha256`, exact `spot_to_image`, and
`density_method=native_RGB_OD_HE_bilinear_midpoint_8x8`. `starts_yx.npy` must
match the exact ordered fields. Required NPY arrays are:

| Name | Shape / dtype |
| --- | --- |
| `density_float32` | `[F,128,128,3]` float32 |
| `area_uint16` | `[F,128,128]` uint16 valid-area fractions / 65535 |
| `field_valid_bool` | `[F,128,128]` boolean |
| `features_float32` | `[F,1024,14,14]` float32, compatible frozen raw UNI1 |
| `pathology_valid_bool` | `[F,14,14]` boolean |

Caching does not prove feature provenance or replace a legitimate UNI access
grant. Exact registration, preprocessing and field identities remain required.
Portable tests explicitly use synthetic density and zero features, **not UNI**.

## Train on your own Visium HD and fine-tune

[User training](training.md#train-on-your-own-visium-hd) uses `configure-training`
and an explicit writable workspace while running from the source checkout.
It preserves the published gene panel's original fitting provenance, spatial
holdout, differentiable FP32 forward and existing loss/optimizer/selection.

[Fine-tuning](fine-tuning.md) remains available for HD16 and Spot55 prepared
coarse caches with disjoint support/selection FOVs. Fine labels in a benchmark
`reference/` are used only by `evaluate`. Frozen benchmarks above do not call
fine-tuning. To score an adapted result, predict with the explicit checkpoint
and call `evaluate`; its report records `fine_tuned=true`. ST100 adaptation
and arbitrary legacy-platform adapters are outside the validated protocol.
