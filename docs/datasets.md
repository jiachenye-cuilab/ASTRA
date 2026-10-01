# ASTRA datasets

This page describes the data, training/validation/test splits and evaluation
boundaries for **ASTRA** (raw H&E, 2,000 genes) and its optional fine-tuning
component. Resource IDs identify datasets rather than local filesystem paths.
Full training data, downstream cohorts, clinical tables and case-specific
fine-tuned weights are not distributed with the package. Base-model training
code, the frozen gene panel, FOV coordinates and spatial splits are included;
see [Training guide](training.md).

Cohort-specific preprocessing, sample selection and statistical analyses are
described in the accompanying paper.

## Overview

| Dataset | Size and unit | Role |
| --- | --- | --- |
| Visium HD development set | 12 sections: 8 WT and 4 HD 3′ | Final base-model training, with spatial validation regions held out in every section |
| Base-model validation | 8 non-overlapping FOVs per development section; 96 FOVs per task | Joint HD16/Spot55 model selection; selected checkpoint at epoch 96 |
| Visium HD fixed test set | 6 sections: 5 WT and 1 HD 3′; 5 known source groups | Prespecified evaluation after the model is frozen |
| ILC | 43 primary sections from 43 patients | Downstream analysis of breast invasive lobular carcinoma using measured Visium data |
| CRC | Matched HD, measured Visium, Xenium and Flex data; see below | Colorectal cancer reconstruction and microenvironment analysis |
| ST-TNBC | 94 specimens in 92 patient groups; 106 array/subarray resources | Spatial and clinical downstream analysis of triple-negative breast cancer |

WT denotes probe-based Visium HD Whole Transcriptome; HD 3′ denotes the Visium
HD 3′ assay. Their protocol identifiers are `protocol_id=1` and `0`, respectively,
independent of the HD16/Spot55 **observation geometry**.
FF = fresh frozen; FFPE = formalin-fixed paraffin-embedded; NAT = normal adjacent
tissue. FOVs, adjacent sections, processing versions and multiple arrays must
not be counted as independent patients.

## Base-model training, validation and testing

### Final training: 12 development sections

Data come from the 10x Genomics pages linked below. Use the listed Space Ranger
versions; another processing version of the same section is not an additional
sample. All sections have human tissue H&E images and native high-resolution
expression data.

| Resource ID | Tissue and preservation | Cross-validation fold | Space Ranger | Official source |
| --- | --- | --- | --- | --- |
| `WT_Colon_FF_11mm` | Colon cancer, FF, 11 mm | fold2 | 4.1.0 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-11mm-human-colon-cancer-HE) |
| `WT_Brain_FFPE_6p5mm` | Brain cancer, FFPE, 6.5 mm | fold2 | 4.1.0 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-6p5mm-human-brain-cancer) |
| `WT_Kidney_FFPE_6p5mm` | Kidney, FFPE, 6.5 mm | fold3 | 4.0.1 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-kidney-ffpe-v4) |
| `WT_Lymph_Node_FFPE_6p5mm` | Lymph node, FFPE, 6.5 mm | fold3 | 4.0.1 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-lymph-node-v4) |
| `WT_Prostate_FFPE_6p5mm` | Prostate cancer, FFPE, 6.5 mm | fold4 | 4.0.1 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-prostate-cancer-ffpe) |
| `WT_Breast_Fixed_Frozen_11mm` | Breast cancer, fixed frozen, 11 mm | fold4 | 4.1.0 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-11mm-human-breast-cancer) |
| `WT_Heart_FFPE_6p5mm` | Heart, FFPE, 6.5 mm | fold1 | 4.1.0 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-6p5mm-human-heart) |
| `WT_Pancreas_FFPE_11mm` | Pancreas, FFPE, 11 mm | fold1 | 4.1.0 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-11mm-human-pancreas) |
| `HD3_Ovarian_Cancer` | Ovarian cancer, HD 3′, FF | fold2 | 4.0.1 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-three-prime-ovarian-cancer-fresh-frozen) |
| `HD3_Lymph_Node` | Lymph node, HD 3′, FF | fold3 | 4.1.0 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-three-prime-human-lymph-node-fresh-frozen) |
| `HD3_Tonsil` | Tonsil, HD 3′, FF | fold4 | 4.1.0 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-three-prime-human-tonsil-fresh-frozen) |
| `HD3_Pancreatic_Cancer` | Pancreatic cancer, HD 3′, FF | fold1 | 4.0.1 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-three-prime-human-pancreatic-cancer-fresh-frozen) |

Download **Feature Slice H5** from **Output and supplemental files** and the
original full-resolution H&E `tissue_image` from **Input files**. Keep the provider
filenames and use the repository-relative paths in [resource.json](../resource.json).

The [dataset split](../training/dataset_split.json) defines four folds. Each fold
holds out the corresponding 3 sections (2 WT and 1 HD 3′) and trains on the other
9. The table records roles during cross-validation. **The released final-refit
model uses all 12 development sections; the three fold1 sections are not an
independent whole-section validation set for this model.**

Pretrained model selection uses 8 non-overlapping spatial FOVs per development
section, excluded from the training coordinates. HD16 and Spot55 observations
are constructed for the same regions: 96 FOVs per task, not 192 independent
regions. Training ran for 126 epochs and joint spatial validation selected
epoch 96. The training configuration has no whole-section validation entries;
spatial validation regions are specified separately. Training observation
regions use random geometry; HD16 and Spot55 are used for validation.

### Fixed testing: 6 sections

| Resource ID | Tissue and technology | Source group | Space Ranger | Official source |
| --- | --- | --- | --- | --- |
| `WT_Ovarian_FF_6p5mm` | Ovarian cancer, WT, FF, 6.5 mm | Adjacent ovarian sections | 4.0.1 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-ovarian-cancer-discovery-fresh-frozen) |
| `HD3_Ovarian_Cancer_FF` | Ovarian cancer, HD 3′, FF | Same group as the preceding row | 4.0.1 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-three-prime-ovarian-cancer-discovery-fresh-frozen) |
| `WT_Breast_FF_Ultima` | Breast cancer, WT, FF, Ultima sequencing | Breast Ultima | 4.0.1 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-breast-cancer-ff-ultima-4) |
| `WT_Colorectal_FFPE_6p5mm` | CRC P2, WT, FFPE | Human_CRC_P2 | 4.0.1 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc-v4) |
| `WT_Colon_FFPE_6p5mm` | Colon cancer, WT, FFPE, 6.5 mm | Colon FFPE | 4.1.0 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-6p5mm-human-colon-cancer) |
| `WT_Pancreas_FFPE_6p5mm_v4p0p1` | Pancreas, WT, FFPE, 6.5 mm | Pancreas FFPE | 4.0.1 | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-human-pancreas-4) |

For each section, use its official source and the Space Ranger version listed
above. Download the native **Feature Slice H5** from **Output and supplemental
files** and the original full-resolution H&E `tissue_image` from **Input files**.
For both ovarian sections, select the deeply sequenced dataset rather than the
minimum-depth alternative. Keep the provider filenames and place the files at
the corresponding `feature_slice` and `tissue_image` paths in
[resource.json](../resource.json). These paths are relative to the repository
root; the raw files are not bundled.

The two ovarian test sections are [adjacent sections from one source](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-ovarian-cancer-discovery-fresh-frozen),
from a different donor than the [training ovarian section](https://www.10xgenomics.com/datasets/visium-hd-three-prime-ovarian-cancer-fresh-frozen).
Count the adjacent test pair as one source group in aggregate results. The five
groups do not imply that every donor identity has been verified. Sequencing-depth
controls, such as minimum-depth versions of the same section, do not add
independent samples.

Fine-scale test expression is used only for evaluation after the model is frozen,
not for base-model fitting or checkpoint selection. This table specifies test
roles; **the package does not provide all test scores or a new performance
evaluation**. The SR 3.0.0 CRC P2 asset `WT_CRC_P2_FFPE_v3p0p0` is a reprocessing
of the same section and is also excluded from fitting and model selection.
Reusing P2 in downstream CRC analysis does not add an independent test case.
The provider notes localized H&E focus issues in the 6.5 mm pancreas test
section; retain per-section results when reporting performance.

### The 2,000-gene panel

Input and output gene order is fixed; see
[input_gene_ids.json](../model/input_gene_ids.json) and
[output_gene_ids.json](../model/output_gene_ids.json). The panel retains 120 marker
genes, then fills the remaining positions using equally weighted per-section
HVG percentile ranks. Fitting uses the **9 fold1 training sections** and their
training regions outside spatial validation: all sections above except
`WT_Heart_FFPE_6p5mm`, `WT_Pancreas_FFPE_11mm` and `HD3_Pancreatic_Cancer`.
The final refit on 12 sections retains this frozen panel. Test data and
downstream cohorts are not used to reselect genes. Missing input genes must be
marked with `gene_available`.

## Downstream ILC cohort

Sources: [ILC dataset on Zenodo](https://zenodo.org/records/14924871) and
[the authors' code and data](https://github.com/BCTL-Bordet/ILC-Spatial-Transcriptomics).
The `ILC_Visium` collection contains **43 distinct patients with primary ILC,
one section per patient**, profiled with fresh-frozen Visium: 55 µm spot diameter
and 100 µm center-to-center spacing.

This cohort is not used for ASTRA base-model training or selection. It has no
native 8 µm RNA ground truth; the model's 8 µm outputs are predictions.

## Downstream CRC cohort

Sources: [10x Human CRC multiplatform cohort](https://www.10xgenomics.com/platforms/visium/product-family/dataset-human-crc)
and [the authors' analyses and annotations](https://github.com/10XGenomics/HumanColonCancer_VisiumHD).
The FFPE datasets include Visium HD WT, measured Visium CytAssist v2,
[Xenium](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280314) and
[Chromium Flex](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280311).

These data are used for downstream reconstruction and microenvironment analysis,
not for base-model training or selection. CRC P2
(`WT_Colorectal_FFPE_6p5mm`) is also part of the fixed test set and does not
add an independent test case. Cross-assay serial sections are not bin-level
ground truth at identical locations.

## Downstream ST-TNBC cohort

Source: [ST TNBC, Zenodo 14204217](https://zenodo.org/records/14204217).
The cohort uses original Spatial Transcriptomics (ST v1) on fresh-frozen tissue,
with **100 µm spot diameter and 150 µm spacing**. It comprises 106 arrays or
subarrays from 94 specimens in 92 patient groups.

This cohort is not used for base-model training or selection and has no native
8 µm RNA ground truth. Whole-section inference uses the
[ST100 adapter](workflows.md#old-st--st-tnbc-100-µm-circles-and-nine-fovs).
ST v1 inputs cannot be used directly as Spot55 fine-tuning inputs. Cohort
inference outputs are not bundled.

## Optional fine-tuning and included examples

The [fine-tuning component](fine-tuning.md) adapts the model to an individual
target section using spatially disjoint support and selection regions. It uses
only target coarse counts, H&E and frozen UNI features, without reading target
8 µm or 2 µm labels. Evaluation must disclose that target observations were used
for adaptation; it does not measure generalization without access to target
observations. The package provides the general adaptation component, without
case-specific weights for ILC, CRC or ST-TNBC.

The included `examples/hd16_input.npz` and `examples/spot55_input.npz` come from
the same FOV in `WT_Colon_FFPE_6p5mm`, with native 2 µm grid origin
`[row, col]=[2672, 560]`. This section belongs to the **fixed test set** above.
Both examples use coarse observations and image features, with **640/2000**
genes available after reordering. The `*_reference.npz` files contain pretrained
ASTRA predictions, not measured fine-scale labels. Two observation geometries
from one FOV are not independent samples and cannot establish full-panel
performance.

Dataset access and use are subject to the original providers' access and
citation requirements. The package does not automatically download these
cohorts. See the [README](../README.md) to prepare your own inputs; inference from
raw RGB requires separately obtained UNI weights. UNI is an external pretrained
encoder; its complete pretraining sample list is not provided here.

Machine-readable base-model data splits, training settings, the gene panel and
spatial holdout regions are provided in
[training/dataset_split.json](../training/dataset_split.json),
[training/config.json](../training/config.json), [training/panel.json](../training/panel.json)
and [training/spatial_split.json](../training/spatial_split.json), respectively.
