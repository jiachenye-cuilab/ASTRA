# Reproducibility and asset sources

The source version is `2026.9.30.dev0`. `VERSION=2026.09.23` preserves the
historical release identifier. Model identity, physical scale,
arithmetic and ordered-gene checksums are in [model metadata](../model/metadata.json).
Training partitions and preprocessing provenance are retained because they
define the scientific protocol.

## Assets and verification scope

| Asset | Input → output / command guide | Availability and verification |
| --- | --- | --- |
| [Pretrained checkpoint](../model/checkpoint.pt), [model configuration](../model/config.json), [ordered panel](../model/input_gene_ids.json) | Prepared FOV → inferred 8 µm counts; [quickstart](../README.md#quickstart) | Included; CPU strict loading and opt-in numerical replay; no optimizer/RNG state |
| [Example inputs and references](../examples/) | One fixed-test colon FOV in HD16 and Spot55 geometry → saved NPZ prediction | Included; 640/2,000 measured genes; references are predictions, not measured ground truth |
| [Section preparation/inference](section-inference.md), [image decoders](image-inputs.md) | Registered raw observations or compatible cache → section arrays/report | Included; synthetic geometry, image, cache and serialized export tests; raw UNI extraction not covered |
| [Native benchmarks and scoring](workflows.md#hd16-and-pseudo-visium-benchmarks) | Native 2 µm data → coarse observations + separate measured 8 µm reference → frozen prediction and metrics | Included; deterministic preparation/prediction/readback/scoring tests; full six-section/GPU replay not run |
| [Visium conversion](workflows.md#real-10x-visium-including-ilc-like-sections), [template](../configs/section_visium.json) | Filtered H5 + registered tissue positions → section input | Included; barcode alignment, counts and calibration tests; complete ILC cohort not run |
| [ST100 adapter](workflows.md), [template](../configs/section_st100.json) | 100 µm counts + coordinates + image/cache → nine-FOV mean export and optional separate gap interpolation | Included; synthetic geometry, saved mean, empty-support and tissue-barrier tests; complete raw ST-TNBC cohort not run |
| [Training code](../astra/training/), [user manifest](../configs/user_training.json), [base recipe](../training/config.json), [partition](../training/dataset_split.json), [panel](../training/panel.json), [geometry](../training/geometry/) | User HD workspace or registered base data → caches/checkpoints; [training guide](training.md) | Included; offline configuration, panel provenance and spatial holdout tests; full training/resume not run |
| [Fine-tuning code](../astra/fine_tuning/), [recipe](../model/fine_tuning.json) | Coarse target support/selection regions → adapted checkpoint; [fine-tuning guide](fine-tuning.md) | Included; local synthetic CPU smoke checked one update, frozen teacher and checkpoint replay for HD16 and Spot55; full 50-epoch adaptation not run |
| [Portable tests](../tests/), [CPU CI](../.github/workflows/cpu.yml) | Synthetic fixtures and separately enabled pretrained examples → regression checks | Included; local Linux/Python 3.12 CPU execution; hosted CI requires its own run |

This method package provides ASTRA workflows and benchmark adapters. External
comparison implementations, their shared evaluation masks, downstream analyses
and paper plotting assets are outside its scope; ASTRA benchmark scores alone
do not reproduce all paper comparisons.

STv1 uses **100 µm diameter / 150 µm pitch**, separate from Spot55 **55 µm
diameter / 100 µm pitch**. The ST100 guide defines the observation raster,
registration, complete-anchor support, nine-FOV placement and averaging.
File conversion or successful checkpoint loading alone does not establish
biological validation; adjacent sections are not identical-location ground
truth. Final-grid conservation limits are in [conservation.md](conservation.md).

## Run the checks

Clone the repository and activate your environment as in
[README](../README.md#installation). Run these commands from its root:

```bash
python -m pip install -r requirements-section.txt
python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  python -B -m unittest discover -s tests -v
```

Fixtures use deterministic synthetic counts/images, two synthetic genes,
random weights seeded at 930 and synthetic features **not produced by UNI**.
Checks include CLI/imports, missing inputs, gene order, synthetic checkpoint
identity, final serialized exports, overlap/boundaries and bounded decoding.
They do not measure biological reconstruction accuracy.

Bundled pretrained replay is opt-in and uses the supplied checkpoint and cached
features without downloading UNI:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 ASTRA_RUN_BUNDLED=1 \
  python -B -m unittest discover -s tests -k Bundled -v
```

CPU CI uses Python 3.12 on Ubuntu 22.04/24.04 with pinned direct dependencies.
Other Python minor versions are not declared supported. GPU inference, UNI
regeneration, full base/user training, resume, adaptation and complete paper
benchmarks require licensed data, weights, prepared caches and an allocated
GPU; they are separate from these CPU checks.

## Acquisition and licensing

| Asset | Source and conditions |
| --- | --- |
| Code/documentation | [MIT](../LICENSE); this grant does not license external data or models |
| Visium HD / CRC | Section-specific [dataset sources](datasets.md), including [10x CRC v4](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc-v4); retain processing versions, citation, notices and applicable [site/dataset terms](https://www.10xgenomics.com/legal/terms-of-use) |
| ILC | [Zenodo 14924871](https://zenodo.org/records/14924871); check the record's access, license, citation and linked source before use |
| ST-TNBC | [Zenodo 14204217](https://zenodo.org/records/14204217); obtain matrices, images and annotations under their record/file-specific terms |
| UNI1 | Request individual gated access at [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI), accept the owner's terms and obtain `pytorch_model.bin` locally; gated weights are not redistributed |
| Bundled ASTRA checkpoint | ASTRA reconstruction weights trained using frozen UNI1 features; the checkpoint contains no UNI encoder weights. [Model metadata](../model/metadata.json) records identity; applicable upstream UNI terms remain relevant |
| Cached example features | Precomputed UNI1 outputs for offline example replay; these are features, not UNI encoder weights. Applicable UNI and source-dataset terms remain in effect |
| Dependencies and optional codecs | Each project's own license and notices apply |

[UNI's upstream terms](https://huggingface.co/MahmoodLab/UNI#license-and-terms-of-use)
specify academic/noncommercial use and address models trained on UNI outputs.
The repository's MIT code license does not replace those terms or grant
additional rights to third-party or UNI-derived assets.
