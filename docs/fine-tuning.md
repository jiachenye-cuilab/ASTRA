# Optional ASTRA fine-tuning

`python -m astra fine-tune` adapts ASTRA to **one target section and one observation task**
using coarse measurements only. Fine-tuning uses FP32
forward, backward, allocation and stable count-KL objectives. UNI features and
the source teacher remain frozen. No case-specific weights are distributed.

| Setting | Training recipe |
| --- | --- |
| Initialization | Pretrained ASTRA checkpoint |
| Updated parameters | Full ASTRA reconstruction model |
| Budget | 50 epochs × 64 FOVs; batch 4, accumulation 4; 200 updates, 3,200 exposures |
| Optimizer | AdamW, lr `3e-5`, weight decay `1e-4`, 8-update warmup, gradient norm limit 1 |
| Preservation weight | `0.25` |
| Selection | Coarse selection NLL at epoch 0 and every 10 epochs; retain the best |
| HD16 objective | Aggregate measured 16 µm bins into 32 µm inputs; supervise original 16 µm counts |
| Spot55 objective | Merge neighboring measured spots; supervise original spot counts; two fixed selection pair views |

No target 8 µm or 2 µm expression is read for fitting or selection. Report this
as adaptation using target coarse observations when evaluating generalization.
The complete recipe is in `model/fine_tuning.json`.

## Use one section cache for training and inference

Prepare the section as described in [Whole-section inference](section-inference.md).
Use 160/160 µm core/stride for HD16 and **160/144 for Spot55**. Fitting and
inference share this cache; Spot55 predictions are averaged across overlapping
cores. Images, UNI features and coarse observations are prepared once.

Create a manifest in your task directory:

```json
{
  "sample_id": "my-section",
  "task": "Spot55",
  "prepared_section": "../outputs/section-inputs",
  "boundary_x_2um": 2400
}
```

The cache path is relative to the manifest. Set the real sample identity and
physical split boundary; 2400 is only an example. Support FOVs end at or before
the boundary and selection FOVs start at or after it. Entire 256 µm FOVs crossing
the boundary are excluded. Preserve the original physical boundary when changing
grids for reproduction; the software does not silently recompute it.

At least 64 eligible support FOVs and a nonempty selection region are required.
HD16 needs usable observed labels in the central 160 µm region. Spot55 requires
at least two neighboring pairs per FOV, at most 110 µm apart and with complete
measured circular supports. The sampler selects 64 support FOVs per
epoch from the eligible pool.

```bash
python -m astra fine-tune --manifest configs/my-section.json --output outputs/fine-tuned-smoke --smoke --device cuda:0
python -m astra fine-tune --manifest configs/my-section.json --output outputs/fine-tuned --device cuda:0
python -m astra predict-section --inputs outputs/section-inputs --fine-tuned outputs/fine-tuned/checkpoint.pt --output outputs/fine-tuned-prediction --device cuda:0
```

Omitting `--fine-tuned` uses the pretrained model. Loading checks the base,
architecture, ordered genes, sample, task and recorded geometry. Epoch 0 can
remain the best selected checkpoint despite completed optimization; reports
distinguish selected epoch from total updates.

## Single-FOV manifests

The NPZ adapter uses FP32 objectives and bounded prefetch. Prepare inputs with
`python -m astra prepare-input` and frozen UNI features
with `python -m astra prepare-features`. Spot55 NPZs additionally need
`parent_centers_yx_um[P,2]`, local measured centers in µm. HD16 uses 256 row-major
parent slots; missing observations are masked rather than treated as labels.

```json
{
  "sample_id": "my-section",
  "task": "HD16",
  "support": [{"input": "features/fov0.npz", "origin_yx_um": [0, 0]}],
  "selection": [{"input": "features/selection0.npz", "origin_yx_um": [1024, 0]}]
}
```

This illustrates the schema only; provide at least 64 distinct support files.
All FOVs must share the sample/assay/panel and have real registered origins.
Support and selection FOVs cannot overlap. This adapter does not build a section
grid; use the prepared section cache for whole-section tiling and export.

```bash
python -m astra predict --input features/query.npz --fine-tuned outputs/fine-tuned/checkpoint.pt --sample-id my-section --task HD16 --output outputs/query.npz
```

## Runtime and validation

Execution defaults to CPU and two threads. Use one assigned GPU per process.
The CUDA allocator cap defaults to `min(12 GiB,75% of device memory)`, adjustable
with `--gpu-memory-gib`; it is not a measured requirement. See the
[section guide](section-inference.md) for memory controls.

`--smoke` performs one update on 16 support FOVs and is not complete adaptation.
Output includes `checkpoint.pt`, `epochs.jsonl` and `report.json`; the directory
must not exist. Smoke checkpoints are marked and are not bundled.

The [portable tests](reproducibility.md) and pretrained inference replay do not
replace a complete 50-epoch adaptation run or an evaluation of biological
generalization. Full adaptation requires the coarse target data, prepared
image/features and separate support/selection regions described above.
