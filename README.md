# ASTRA

**Assay-aware Spatial Transcriptomic Reconstruction and Allocation**

- **Input:** raw coarse UMI counts, gene IDs, physical coordinates and registered H&E, or compatible cached UNI1 features.
- **Output:** inferred counts on an 8 µm grid for a fixed 2,000-gene panel, with gene availability, spatial support and numerical diagnostics.
- **Workflows:** frozen HD16/pseudo-Visium benchmarks, Visium inference at core/stride 160/144 µm, ST100 inference with nine FOVs, Visium HD training and HD16/Spot55 fine-tuning. Portable CPU tests and pretrained example replay are available; full raw-data cohorts and training require separate validation.

[Install](#installation) · [Quickstart](#quickstart) · [Inputs](#inputs) ·
[Pretrained use](#pretrained-inference-and-benchmarks) · [Training](#training-and-fine-tuning) ·
[Reproducibility](#reproducibility) · [Limitations](#limitations) · [License](#license)

## Installation

Clone the repository and run ASTRA from its root directory. Use Conda to
create a dedicated **Python 3.12** environment, then install the pinned
dependencies with pip:

```bash
git clone https://github.com/jiachenye-cuilab/ASTRA.git
cd ASTRA
conda create -n astra python=3.12 pip
conda activate astra
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m astra --help
```

Keep the `astra` environment active and use the cloned repository as your
working directory for the commands below. If you already have a dedicated
Python 3.12 environment, you can use it instead; a separate venv is optional
and should not be layered inside the Conda environment. Dependencies are
pinned in [requirements.txt](requirements.txt). Section preparation,
benchmarking and training also need the corresponding dependency file linked
below. The installation above uses CPU PyTorch for the bundled quickstart.
For GPU workflows, use the matching PyTorch 2.5.1 / torchvision 0.20.1 CUDA
installation from the [official version-specific instructions](https://pytorch.org/get-started/previous-versions/)
and one allocated GPU per process. Other Python minor versions are not declared
supported.

## Quickstart

The bundled checkpoint and cached features allow offline CPU inference:

```bash
python -m astra prepare-input --self-test
python -m astra predict --input examples/hd16_input.npz --output outputs/hd16.npz --reference examples/hd16_reference.npz
python -m astra predict --input examples/spot55_input.npz --output outputs/spot55.npz --reference examples/spot55_reference.npz
```

Each output NPZ contains `pred_count_8um[32,32,2000]`, ordered `gene_ids`,
`gene_available` and `field_valid_8um`; a JSON report records diagnostics.
These two inputs use one fixed-test colon FOV with **640 measured genes out
of the 2,000-gene panel**. The reference files are pretrained predictions for
numerical replay, not measured 8 µm ground truth or a paper benchmark.
Unavailable genes are not ground-truth zeros.

## Inputs

| Input | Required information | Guide |
| --- | --- | --- |
| One registered FOV | Coarse counts, measured gene IDs, masks and raw image features or compatible UNI cache | [Single-field inference](docs/inference.md) |
| Whole section | Count matrix, capture coordinates, registered RGB image and explicit physical calibration | [Section inference](docs/section-inference.md), [image formats](docs/image-inputs.md) |
| Native Visium HD benchmark | Space Ranger 2 µm `feature_slice.h5`, tissue mask and registered H&E | [Benchmark recipe](docs/workflows.md#hd16-and-pseudo-visium-benchmarks) |
| Real 10x Visium | Filtered feature-barcode H5 and full-resolution tissue positions CSV | [Visium recipe](docs/workflows.md#real-10x-visium-including-ilc-like-sections) |
| Legacy ST100 | Raw spot counts, 100 µm capture centers, image registration and optional tissue mask | [ST100 recipe](docs/workflows.md#old-st--st-tnbc-100-µm-circles-and-nine-fovs) |

Use raw, nonnegative integer UMI counts with unique gene IDs. Preparation
reorders counts into the fixed panel and marks absent genes unavailable.
Image dimensions or generic DPI do not determine physical calibration.
Registration and acquisition-specific conversions must preserve the same
coordinate frame for counts, images and masks.

## Pretrained inference and benchmarks

The supplied ASTRA checkpoint uses **FP32**, the fixed 2,000-gene panel and
original-color H&E preprocessing. Frozen inference does not optimize weights.
Raw image feature extraction requires separately obtained **UNI1** weights:
request access at [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI), follow
the owner's terms, and supply the downloaded `pytorch_model.bin` locally.
Its identity must match [model/metadata.json](model/metadata.json).
Bundled cached examples do not need a separate UNI download.

Choose a template, copy it to your task's `configs/` and edit it before running:

| Workflow | Template | Guide |
| --- | --- | --- |
| HD16 section, core/stride 160/160 µm | [section_hd16.json](configs/section_hd16.json) | [Section inference](docs/section-inference.md) |
| Visium/Spot55 section, core/stride 160/144 µm | [section_spot55.json](configs/section_spot55.json) | [Section inference](docs/section-inference.md) |
| Frozen HD16 benchmark | [benchmark_hd16.json](configs/benchmark_hd16.json) | [Benchmarks](docs/workflows.md#hd16-and-pseudo-visium-benchmarks) |
| Frozen pseudo-Visium benchmark | [benchmark_pseudo_visium.json](configs/benchmark_pseudo_visium.json) | [Benchmarks](docs/workflows.md#hd16-and-pseudo-visium-benchmarks) |
| Real 10x Visium, including ILC-like data | [section_visium.json](configs/section_visium.json) | [Visium](docs/workflows.md#real-10x-visium-including-ilc-like-sections) |
| ST100, center + eight supplemental FOVs | [section_st100.json](configs/section_st100.json) | [ST100](docs/workflows.md) |

For an HD16 section:

```bash
python -m pip install -r requirements-section.txt
cp configs/section_hd16.json configs/my_hd16.json
# Edit input paths, calibration and registration in the copied configuration.
python -m astra prepare-section --config configs/my_hd16.json --output outputs/section-inputs --device cpu
python -m astra predict-section --inputs outputs/section-inputs --output outputs/section-prediction --device cpu
```

Configuration paths are relative to the configuration file. Use new output
directories for each run; use `--device cuda:0` on an allocated GPU. Section
outputs contain `prediction.npy`, spatial coordinates, gene availability and
`report.json`. The [workflow guide](docs/workflows.md) provides complete
preparation, inference and scoring commands for each supported input.
Benchmark preparation keeps measured 8 µm references separate from model input;
evaluation reopens the saved prediction files.

## Training and fine-tuning

Install [requirements-training.txt](requirements-training.txt) and follow
[training on your own Visium HD](docs/training.md#train-on-your-own-visium-hd).
The [user_training.json](configs/user_training.json) manifest declares your
sections, native count files, images, UNI weights and spatial monitor blocks:

```bash
python -m pip install -r requirements-training.txt
cp configs/user_training.json configs/my_training.json
# Edit data paths, registration and monitor blocks in the copied manifest.
python -m astra configure-training --manifest configs/my_training.json --output tasks/my-training
python -m astra train --workspace tasks/my-training --check-config
```

Edit a copied manifest first. The training guide covers cache preparation,
scratch training, checkpoints and resume. The supplied fixed-panel recipe,
dataset partition and geometry in `training/` support the documented base
training protocol; `python -m astra train --check-config` checks it offline.

[Target-section fine-tuning](docs/fine-tuning.md) supports HD16 and Spot55 coarse
observations with disjoint support and selection regions. It starts from the
pretrained model and uses a differentiable FP32 forward. Target 8 µm/2 µm labels
are not used for adaptation or checkpoint selection. ST100 fine-tuning is not
provided.

## Reproducibility

[Assets, sources and verification commands](docs/reproducibility.md) distinguish
portable synthetic CPU checks, opt-in pretrained replay, measured benchmarking
and full training. [Datasets and splits](docs/datasets.md) describe public sources
and fitting boundaries. The source repository includes the CLI, required model
assets, editable configurations, portable tests and CPU CI.

```text
astra/       Python package and CLI
model/       Frozen checkpoint, architecture, ordered genes and model metadata
configs/     Editable workflow templates
examples/    Offline inputs and reference predictions
docs/        User guides, scientific contracts and data sources
training/    Published base-training recipe, panel, partition and geometry
tests/       Portable fixtures and regression checks
.github/     Source-checkout CPU CI
```

## Limitations

Exact allocation conservation applies to observation/cell intersection masses.
It transfers to complete aligned HD16 parents. Spot55/ST100 rasterization and
overlap averaging generally do not preserve that invariant on the final grid;
saved-grid residuals are reported separately. See [operators and tolerances](docs/conservation.md).

Image format support, memory limits and rejected conversions are specified in
the [image guide](docs/image-inputs.md). Synthetic adapter tests do not establish
biological validity for every platform. Full raw-data/GPU benchmarks, cohort
analyses, complete training and adaptation runs are not covered by CPU smoke
tests. Downstream statistical analyses and paper plotting assets are not part
of this method package. Adjacent sections are not identical-location ground
truth.

## License

Code and documentation use the [MIT License](LICENSE). Datasets, dependencies,
UNI and model-derived assets have their own terms; MIT does not replace them.
Raw datasets and gated UNI weights are not distributed or downloaded by ASTRA.
See the [asset licensing inventory](docs/reproducibility.md#acquisition-and-licensing)
for asset provenance, acquisition requirements and applicable upstream terms.
