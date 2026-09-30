# Whole-section FP32 inference

The section entry points combine observed-only coarse counts, raw H&E
integration, one frozen UNI cache, bounded prefetch,
batch-four FP32 inference, compact export and mean aggregation of overlapping
cores. Provide measured observations and registered images through a section
configuration file.

For native HD16/pseudo-Visium scoring, real Visium H5/CSV conversion, and the
separate ST100 nine-FOV protocol, use the [workflow recipes](workflows.md).
ST100 does not use the Spot55 diameter, tiling or fine-tuning contract below.

## Inputs and preparation

Install `requirements-section.txt` in your own environment and edit
[configs/section_hd16.json](../configs/section_hd16.json) or
[configs/section_spot55.json](../configs/section_spot55.json). Paths are relative to
the configuration file; absolute paths to your explicit inputs are also accepted.
Keep configuration files separate from generated caches and predictions.

| Configuration field | Format |
| --- | --- |
| `sample_id`, `task` | Stable section identity; `HD16` or `Spot55` |
| `protocol_id` | `0` for 3′ or `1` for WT, independently of geometry |
| `counts` | Dense `.npy` or SciPy sparse `.npz`, `[N,G]`, nonnegative integer raw UMI counts |
| `gene_ids` | JSON array of `G` unique measured gene IDs in count-column order |
| `capture_shape_yx_2um` | Native 2 µm raster dimensions `[rows,columns]` |
| `tissue_image` | Registered uint8 RGB TIFF/BTF, or bounded small PNG/JPEG; see [format rules](image-inputs.md) |
| `spot_to_image` | 3×3 transform from native 2 µm column/row indices to image column/row pixel coordinates |
| `uni_checkpoint` | Local UNI1 `pytorch_model.bin` matching `model/metadata.json` |
| `parent_yx_16um` (HD16) | Integer `[N,2]` `.npy`, global row/column indices on the 16 µm grid |
| `spot_yx_um` (Spot55) | Float `[N,2]` `.npy`, global measured spot centers in µm, y then x |
| `query_yx_8um` (Spot55) | Integer `[Q,2]` `.npy`, unique complete global 8 µm cells to predict |

The template registration matrix is a placeholder. Supply the real matrix, e.g.
`spot_colrow_to_microscope_colrow` in compatible feature-slice metadata. Native
2 µm index `(row,col)` denotes the bin centered at `(2*row+1,2*col+1)` µm.
Spot centers are physical coordinates. Every input must use the same section
coordinate frame. Registration and acquisition-specific file conversion are
user inputs; the package handles subsequent tiling, feature extraction and export.

Gene availability follows the measured gene list, including zero-count columns.
Counts are reordered into the fixed 2,000-gene panel and absent genes are marked
unavailable. No fine-expression labels are needed. UNI weights and raw data are
not distributed or downloaded automatically.

```bash
python -m pip install -r requirements-section.txt
python -m astra prepare-section --config configs/section_hd16.json --output outputs/section-inputs --device cuda:0
python -m astra predict-section --inputs outputs/section-inputs --output outputs/section-prediction --device cuda:0
```

Use `--device cpu` for CPU execution. Commands use two CPU threads. Restrict GPU
visibility to an assigned device before execution; each process uses `cuda:0`.

Preparation integrates OD/H/E from original registered pixels, with eight samples
per axis, the optimized eight-row integrator and a bounded 512 MiB density cache.
It builds raw FP32 UNI features once, in batches of 16 (`--uni-batch-size` can
reduce this). Counts use CSR arrays and image/features use memory maps. Raw H&E
is retained without Reinhard normalization. TIFF uses bounded segments; small
PNG/JPEG uses capped full decoding. Oversized rasters/strips are rejected.
See [memory and calibration](image-inputs.md).

## Geometry, arithmetic and outputs

| Task | Input FOV | Core | Stride | Context per side | Overlap |
| --- | --- | --- | --- | --- | --- |
| HD16 | 256 µm | 160 µm | 160 µm | 48 µm | One core per observed parent |
| Spot55, default | 256 µm | 160 µm | 144 µm | 48 µm | Equal mean of contributing cores |
| Spot55, non-overlapping option | 256 µm | 144 µm | 144 µm | 56 µm | One core per query cell |

Spot55 defaults to 160/144 for both fitting and inference. Set `core_um` and
`stride_um` to 144 for non-overlapping cores. Record the chosen geometry and
match it when reproducing an analysis: changing geometry can change predictions
and adapted weights.

Boundary contexts are padded outside capture. HD16 exports complete observed
parents. Spot55 uses only complete measured circles with valid image support.
All sixteen 2 µm cells must be valid for an exported 8 µm cell. Predictions in
Spot55 gaps remain inferred quantities rather than measured observations.

Inference defaults to batch size 4 and bounded device prefetch; CPU devices fall
back to CPU prefetch. `--batch-size` and `--pipeline cpu` are configurable. Counts,
priors, pooling/allocation and fine-tuning forward/loss/backward use FP32 without
mixed precision or TF32.

FP32 allocation checks `abs(residual)/(1+abs(count)) <= 5e-6`. Complete aligned
HD16 exports are re-read and summed at `atol=1e-5`, `rtol=1e-5`. Spot55's final
grid is an approximate representation; serialized raster reaggregation residuals
are reported separately without forcing conservation. Partial/empty supports
are excluded from full-count comparisons. See [operators and support](conservation.md).

- `prediction.npy`: FP32 `[N,4,2000]` for HD16, children in TL/TR/BL/BR order;
  or `[Q,2000]` for Spot55, written through a memory map.
- `parent_yx_16um.npy` or `query_yx_8um.npy`: corresponding global coordinates.
- `gene_ids.json`, `gene_available.npy`: ordered panel and availability.
- `report.json`: checkpoint identity, geometry, batches, conservation, elapsed
  time and process/GPU memory measurements.

Unavailable genes must not be interpreted as biological zeros. Output directories
must not already exist. Inference timing includes array writes and flushes.
Preparation is recorded separately in `preparation.json`: charge its cost once
for a cold whole-section run and add adaptation time when fine-tuning is used.
Prepared-cache inference alone excludes H&E and UNI extraction.

Reuse the same cache for fitting and inference, with an explicit physical
`boundary_x_2um` for the support/selection split. Entire FOVs crossing that
boundary are excluded. See [Fine-tuning guide](fine-tuning.md).

## Memory controls and verification

Data, batch size, hardware and allocator behavior affect actual requirements.
CPU RAM does not decrease solely with model precision.
CLI allocator caps are `min(8 GiB,75% of device memory)` for preparation/inference
and `min(12 GiB,75% of device memory)` for fine-tuning; caps do not guarantee fit.

[Portable tests](reproducibility.md) exercise synthetic section preparation,
overlap handling and serialized export readback. They do not establish GPU
throughput or replace complete raw-data inference and adaptation evaluations.
