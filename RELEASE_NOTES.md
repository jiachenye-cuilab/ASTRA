# ASTRA source package 2026.9.30.dev0

Clone the repository, install its dependencies and use `python -m astra` from
the repository root. The pretrained weights, architecture, ordered 2,000-gene
panel and FP32 inference/fine-tuning workflow are retained.

- Offline HD16 and Spot55 inference examples with reference predictions.
- Whole-section preparation and prediction with registered images or compatible
  UNI caches; HD16 core/stride 160/160 µm and Spot55 160/144 µm.
- Frozen native HD16 and pseudo-Visium benchmarking with separately prepared
  measured 8 µm references and saved-output scoring.
- Real 10x Visium H5/CSV conversion and a separate ST100 nine-FOV adapter.
- Visium HD scratch training in a user workspace and HD16/Spot55 fine-tuning.
- Portable CPU regression tests and source-checkout CI.

Read [README](README.md) for installation and
[verification scopes](docs/reproducibility.md) for prerequisites and limits.
Example replay and synthetic tests are software checks; complete raw-data
benchmarks, training and cohort analyses need separate validation.
