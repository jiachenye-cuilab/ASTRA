# Training ASTRA from scratch

The package includes initialization from scratch, native data preparation,
random observation-region generation, 8 µm supervision, spatial validation,
model selection across both tasks, checkpoint saving and resuming.
The entry point is `python -m astra train`; `astra/training/` shares the
`astra/model/` model with inference. The included configuration provides the
training recipe used for the pretrained ASTRA model.

## Train on your own Visium HD

Use [configs/user_training.json](../configs/user_training.json) with native
2 µm `feature_slice.h5`, its registered H&E, explicit assay identity (0=3′,
1=WT), and local gated UNI1 weights. The template coordinates are examples,
not an automatic tissue-selection rule. Select complete 512 µm physical blocks
using image/capture geometry; record their upper-left y,x in 2 µm indices.
Blocks must be aligned to multiples of 256 indices. Each supplies four complete
256 µm FOVs. Designate exactly two of these blocks per section as spatial
monitors (eight FOVs); all their FOVs are excluded from fitting. At least one
additional block is required. Registered paper fixed-test IDs are rejected as
training sections.

Run from the cloned repository root after installing the training/section
dependencies. All task configuration, cache and checkpoint writes stay under
the new `--workspace`; the source model assets are read-only inputs:

```bash
python -m astra configure-training --manifest configs/user_training.json --output tasks/my-training
python -m astra train --workspace tasks/my-training --check-config
python -m astra prepare-training --workspace tasks/my-training --stage check
python -m astra prepare-training --workspace tasks/my-training --smoke --stage all --device cpu
python -m astra train --workspace tasks/my-training --config tasks/my-training/training/smoke/config.json --smoke --device cpu
```

The smoke cache uses three actual training FOVs and eight monitor FOVs per
section; optimization runs the existing three-update engineering budget.
Configuration checks and synthetic tests do not validate native UNI extraction
or full scientific training. Supply real measured fine-resolution labels;
the bundled example reference predictions cannot be substituted for them.

For full training, set `formal_training=true` in your original manifest before
creating the workspace, choose actual tissue blocks and resource budgets, then:

```bash
python -m astra prepare-training --workspace tasks/my-training --stage all --device cuda:0
python -m astra train --workspace tasks/my-training --device cuda:0
```

The configurable manifest budget fields are `seed`, `max_epochs`, `batch_size`,
`learning_rate`, `gradient_accumulation_steps`, and `fields_per_section`. Without
an explicit field quota, each section uses up to 100 distinct retained training
FOVs per epoch; smaller pools are recorded as section overrides. An explicit
quota larger than the retained pool is rejected. The loss, differentiable FP32
model, observation generation, optimizer family and joint spatial HD16/Spot55
selection remain the existing implementation. Inference's no-grad/in-place
forward is not used for backpropagation.

This route initializes from scratch with the **published fixed 2,000-gene panel**.
It does not fit a new panel on your data; the original nine-section fitting
provenance is retained. Only genes measured by each assay supply supervision.
User-specific caches/checkpoints are new experiments, not a recreation of the
pretrained model. To start from frozen weights and adapt using coarse counts,
use [fine-tuning](fine-tuning.md) instead. ST100 fine-tuning is not provided.

## Included assets and required inputs

| Asset | Location |
| --- | --- |
| Training configuration | [training/config.json](../training/config.json) |
| Split of 12 development and 6 test sections | [training/dataset_split.json](../training/dataset_split.json), [Datasets and splits](datasets.md) |
| Frozen 2,000-gene panel and its nine-section fitting reference | [training/panel.json](../training/panel.json), [training/panel_reference.json](../training/panel_reference.json) |
| Training candidate FOVs and their order | `training/geometry/`: 633,230 FOVs across 12 sections; coordinates and indices only |
| Excluded regions and fixed validation FOVs | [training/spatial_split.json](../training/spatial_split.json) |
| Raw data and UNI file locations | [resource.json](../resource.json) |

Install the Python dependencies and obtain the original 10x `feature_slice.h5`
files, registered H&E TIFF/BTF images, and UNI1 weights under their applicable
terms. These data and external weights are not distributed with ASTRA. The
included inference examples do not contain the measured fine-scale labels
required for training.

`resource.json` defaults to `data/<resource_id>/` for datasets and
`weights/UNI/pytorch_model.bin` for UNI. Use the specified filenames and processing
versions, or edit their paths relative to the **ASTRA repository root**.
Coordinates or images from a different Space Ranger processing version of the
same section must not be combined directly with the frozen FOVs. UNI architecture
and weight identity are checked using `model/uni_config.json`, model metadata
and the checkpoint SHA256.

## Installation and input checks

Run these commands from `ASTRA/` in your own Python environment:

```bash
python -m pip install -r requirements-training.txt
python -m astra train --check-config
python -m astra prepare-training --stage check
```

`python -m astra train --check-config` requires no raw data and does not start training.
`python -m astra prepare-training --stage check` lists missing raw data and UNI files; it
checks path existence, not file contents. Training preparation accepts only the
12 development sections and excludes test sections. The training gene panel is
frozen: the final refit does not recompute highly variable genes (HVGs) or fit a
new gene-selection model.

## Start with a small smoke run on real data

```bash
python -m astra prepare-training --smoke --stage all --device cuda:0
python -m astra train --config training/smoke/config.json --smoke --device cpu
```

Before CUDA execution, use `CUDA_VISIBLE_DEVICES` to expose one assigned GPU.
UNI features can also be prepared on CPU, but this is slower. Preparation uses
one process, two CPU threads and UNI batch size 4 by default. The smoke run keeps
3 real training FOVs and 8 spatial validation FOVs per section. Its configuration
is written to `training/smoke/` and its caches to `outputs/training-smoke-cache/`.
It uses the existing FOV selection and measured expression labels.

The training smoke run uses 1 epoch and 3 single-FOV updates, covering random,
HD16 and Spot55 observation regions. It validates on the configured WT and 3′
sections, selects and saves a checkpoint, then reloads it and verifies its score.
Smoke configurations cannot be used for full training, and smoke checkpoints
are software test outputs rather than reproduced ASTRA models.

## Prepare the complete dataset and train

```bash
python -m astra prepare-training --stage all --device cuda:0
python -m astra train --config training/config.json --device cuda:0
```

Preparation can also run separately with `--stage geometry`, `--stage native`
and `--stage uni`. The latter two accept a single section, for example
`--section WT_Colon_FF_11mm`. The geometry stage restores coordinates and indices;
the native stage constructs sparse counts, raw OD/H/E integrals and image
validity masks from native 2 µm data; the UNI stage extracts FP32 features for
complete FOVs. The UNI summary is marked complete only after all 12 sections
are processed. Complete caches with the same configuration can be reused;
partially written caches raise an error rather than being silently overwritten.

The complete training candidate pool contains **633,230 FOVs**. Its FP32 UNI
features alone require approximately **473.5 GiB**, in addition to raw data,
count caches and image caches. Check storage, GPU availability and runtime
budgets before preparing the full dataset or starting training. The package
contains the executable workflow and frozen coordinates, but not these caches.

| Setting | Training configuration |
| --- | --- |
| Initialization and seed | From scratch; 20260907 |
| Data and panel | All 12 development sections; frozen 2,000 input/output genes; raw H&E |
| Input and supervision | 256 µm FOV; central 160 µm supervision core; 8 µm targets |
| Sampling per epoch | 100 FOVs per section, 1,200 total; cycle through the predetermined order when the candidate pool is exhausted |
| Training observation regions | 100% random geometry; HD16 and Spot55 for spatial validation |
| Batch size and gradient accumulation | 16 and 1 |
| Optimizer | AdamW, learning rate 3e-4, weight decay 1e-4, gradient norm limit 1 |
| Learning-rate schedule | 12 warmup updates, followed by a 240-epoch cosine schedule computed from actual optimizer steps |
| Validation and stopping | Every 3 epochs; at least 30 epochs; patience of 10 checks; at most 240 epochs |
| Checkpoint selection | Joint HD16/Spot55 selection with fixed initial anchors and bounded degradation rules |

The configuration specifies the complete rules. Eight fixed validation FOVs per
section are excluded from the training pool; both tasks evaluate the same
96 regions. The training entry point does not read fixed-test expression data.
The pretrained model was selected at epoch 96 from a run that stopped at
epoch 126. The package provides this recipe and implementation; bitwise-identical
training trajectories are not guaranteed across hardware, recomputed UNI caches
or software environments.

## Outputs, resuming and inference

Outputs default to `outputs/training/formal/`. A new run is rejected if the
directory already exists. `best.pt` contains the selected model; `last.pt`
includes the model, AdamW and schedule-related state, random-number-generator
state, gene order and configuration needed to resume. `epochs.jsonl` and
`report.json` record optimizer updates, model selection and stopping reasons.

```bash
python -m astra train --config training/config.json --resume outputs/training/formal/last.pt --device cuda:0
python -m astra predict --input inputs/my_fov.npz --trained-checkpoint outputs/training/formal/best.pt --output outputs/my_prediction.npz
```

An existing run can resume only while training budget remains and the patience
stopping criterion has not been met. Keep the data, panel, model and optimization
recipe unchanged when resuming. The learning-rate schedule depends on the full
training budget, which must be retained for reproduction. The distributed
`model/checkpoint.pt` contains inference state only and cannot replace a
training `last.pt` checkpoint for resuming.

`--trained-checkpoint` loads a newly trained full model and verifies gene order
and UNI identity. Input formats follow the [inference guide](inference.md).
It is mutually exclusive with `--fine-tuned`; omitting both uses the distributed
pretrained ASTRA model. [Target-section fine-tuning](fine-tuning.md) is a separate
entry point that also starts from the pretrained ASTRA model by default.

Keep configurations under `training/` and run outputs under `outputs/`.
See [verification scopes](reproducibility.md) for the available CPU checks.
Full training, GPU execution and training resume are separate checks requiring
the registered datasets, prepared caches and an allocated GPU.
