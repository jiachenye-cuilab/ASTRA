# ASTRA datasets

This page describes the data, training/validation/test splits and evaluation
boundaries for **ASTRA** (raw H&E, 2,000 genes) and its optional fine-tuning
component. Resource IDs identify datasets rather than local filesystem paths.
Full training data, downstream cohorts, clinical tables and case-specific
fine-tuned weights are not distributed with the package. Base-model training
code, the frozen gene panel, FOV coordinates and spatial splits are included;
see [Training guide](training.md).

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

Data come from [10x Genomics public datasets](https://www.10xgenomics.com/datasets).
Use the specified Space Ranger 4 inputs; another processing version of the same
section is not an additional sample. All sections have human tissue H&E images
and native high-resolution expression data.

| Resource ID | Tissue and preservation | Cross-validation fold |
| --- | --- | --- |
| `WT_Colon_FF_11mm` | Colon cancer, FF, 11 mm | fold2 |
| `WT_Brain_FFPE_6p5mm` | Brain cancer, FFPE, 6.5 mm | fold2 |
| `WT_Kidney_FFPE_6p5mm` | Kidney, FFPE, 6.5 mm; default SR 4.0.1 | fold3 |
| `WT_Lymph_Node_FFPE_6p5mm` | Lymph node, FFPE, 6.5 mm; default SR 4.0.1 | fold3 |
| `WT_Prostate_FFPE_6p5mm` | Prostate cancer, FFPE, 6.5 mm | fold4 |
| `WT_Breast_Fixed_Frozen_11mm` | Breast cancer, fixed frozen, 11 mm | fold4 |
| `WT_Heart_FFPE_6p5mm` | Heart, FFPE, 6.5 mm | fold1 |
| `WT_Pancreas_FFPE_11mm` | Pancreas, FFPE, 11 mm | fold1 |
| `HD3_Ovarian_Cancer` | Ovarian cancer, HD 3′, FF | fold2 |
| `HD3_Lymph_Node` | Lymph node, HD 3′, FF | fold3 |
| `HD3_Tonsil` | Tonsil, HD 3′, FF | fold4 |
| `HD3_Pancreatic_Cancer` | Pancreatic cancer, HD 3′, FF | fold1 |

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

| Resource ID | Tissue and technology | Source group |
| --- | --- | --- |
| `WT_Ovarian_FF_6p5mm` | Ovarian cancer, WT, FF, 6.5 mm | Adjacent ovarian sections |
| `HD3_Ovarian_Cancer_FF` | Ovarian cancer, HD 3′, FF | Same group as the preceding row |
| `WT_Breast_FF_Ultima` | Breast cancer, WT, FF, Ultima sequencing | Breast Ultima |
| `WT_Colorectal_FFPE_6p5mm` | CRC P2, WT, FFPE; default SR 4.0.1 | Human_CRC_P2 |
| `WT_Colon_FFPE_6p5mm` | Colon cancer, WT, FFPE, 6.5 mm | Colon FFPE |
| `WT_Pancreas_FFPE_6p5mm_v4p0p1` | Pancreas, WT, FFPE, 6.5 mm; SR 4.0.1 | Pancreas FFPE |

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

Sample identifiers:

```text
ST1 ST3 ST5 ST7 ST33 ST34 ST35 ST36 ST37 ST38 ST39
ST40 ST41 ST42 ST43 ST44 ST45 ST46 ST47 ST48 ST50
ST51 ST52 ST53 ST54 ST55 ST56 ST57 ST58 ST59 ST60
ST61 ST62 ST63 ST64 ST65 ST66 ST67 ST68 ST69 ST70 ST71 ST72
```

Inputs include Space Ranger counts, spot coordinates and scale factors, original
NDPI H&E, the authors' region annotations, `STutility_object.RDS`, and the
clinical supplementary table `pnas.2517567123.sd01.xlsx`. Downstream analysis
uses 71,997 spots retained by the authors' quality control (QC), together with
their CARD cell-composition estimates and other annotations. Missing tumor-region
annotations in ST54/ST56 exclude those samples only from the corresponding
tumor-interface analysis, not from the entire cohort.

This cohort is not used for ASTRA base-model training or selection. ILC analysis
uses the frozen pretrained ASTRA model. Native 8 µm RNA ground truth is
unavailable; 8 µm outputs are predictions. Regional, cell-composition and
clinical associations provide downstream evidence, not direct validation of
reconstruction accuracy at individual bins.

## Downstream CRC cohort

Sources: [10x Human CRC multiplatform cohort](https://www.10xgenomics.com/platforms/visium/product-family/dataset-human-crc)
and [the authors' analyses and annotations](https://github.com/10XGenomics/HumanColonCancer_VisiumHD).
The tissues are FFPE. Serial sections from the same donor across assays are
grouped by patient.

| Platform | Resource ID or sample | Role |
| --- | --- | --- |
| Visium HD WT | `WT_CRC_P1_FFPE`, `WT_Colorectal_FFPE_6p5mm` (P2), `WT_CRC_P5_FFPE` | P1/P2/P5 tumors; P2 also belongs to the base-model fixed test set |
| Visium HD WT | `WT_CRC_P3_NAT_FFPE`, `WT_CRC_P5_NAT_FFPE` | P3/P5 normal adjacent tissue |
| Measured Visium CytAssist v2 | `Visium_CRC_P2_FFPE`, `Visium_CRC_P3_NAT_FFPE`, `Visium_CRC_P5_NAT_FFPE` | 55 µm spots at 100 µm spacing; 4,269, 3,887 and 2,638 tissue spots, respectively |
| Xenium | `Xenium_CRC_P1_FFPE`, `Xenium_CRC_P2_FFPE`, `Xenium_CRC_P5_FFPE` | Targeted expression and cell/nuclear segmentation in P1/P2/P5 serial sections; [GSE280314](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280314) |
| Chromium Flex | `CRC_Flex_SingleCell_Metadata` | Single-cell expression and the authors' annotations; [GSE280311](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280311) |

Flex includes `P1CRC`, `P2CRC`, `P3CRC`, `P4CRC`, `P5CRC`, `P2NAT`, `P3NAT` and
`P5NAT`. The input matrix contains 279,609 cells and 18,082 genes; the authors'
QC retains 260,506 cells. Exclude the target donor when constructing a reference
from other donors; overlapping reference and target data are not independent
validation.

The default P2 HD input uses [Space Ranger 4.0.1](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc-v4).
The authors' SR 3.0.0 spatial annotations and [SpaceHack pathology-region and
nuclear annotations](https://zenodo.org/records/11402686) concern the same P2
data. Check coordinate versions before use; these annotations incorporate H&E
and expression-marker information. The authors' earlier Xenium metadata must
not be joined directly to newer Xenium outputs by cell ID.

CRC analysis distinguishes **simulated Spot55 observations aggregated from the
same HD section** from **measured Visium P2 and other serial sections**. Simulated
observations can be compared with native fine-scale counts from the same section.
For measured Visium, the matched HD/Xenium sections are adjacent sections and
support appropriate regional or cell-type comparisons, not bin-level ground
truth at identical locations. The measured P2 analysis also compares the
pretrained model with adaptation using target coarse observations. Case-specific
weights and the complete downstream analysis scripts are not distributed in
this inference package.

## Downstream ST-TNBC cohort

Source: [ST TNBC, Zenodo 14204217](https://zenodo.org/records/14204217).
The cohort uses original Spatial Transcriptomics (ST v1) on fresh-frozen tissue,
with **100 µm spot diameter and 150 µm spacing**. There are 106
`ST_TNBC_<specimen>_<array>_<subarray>` resources representing 94 specimens in
92 patient groups. Specimen numbers are 1–16 and 19–96; a specimen can contain
multiple arrays or subarrays. TNBC30 is a recurrence specimen from TNBC53, and
TNBC66 is a recurrence specimen from TNBC58; neither represents a new patient.

The main inputs are the authors' `rawCountsMatrices`, `byArray`, original H&E
images and spatial annotations. `Clinical/ids.RDS` maps arrays to specimens;
`Clinical.RDS` and `Clinical.xlsx` provide clinical data. Clinical statistics use
patient groups; serial sections and recurrence specimens do not increase the
patient sample size.

The cohort analysis includes:

- **Spatial analysis of 82 patients**, one primary section per patient: specimen
  numbers 1–16 and 19–96 excluding
  `11, 24, 26, 30, 34, 36, 49, 55, 60, 66, 68, 79`.
  These are the inclusion criteria for this analysis, not a blanket low-quality
  designation for excluded samples.
- **Clinical and raw RNA analysis of 92 patient groups**, a different denominator
  from the spatial analysis.
- Specimens 31, 32 and 45 are included in both illustrative and cohort analyses;
  the cohort analysis is exploratory for these cases, not independent validation.
  Spatial analyses use the same 1,485 measured genes as the normalization
  denominator. Missing observations in the 2,000-gene panel must not be treated
  as measured zeros.
- The four local examples are `ST_TNBC_31_CN16_C2`, `ST_TNBC_32_CN16_E1`,
  `ST_TNBC_32_CN16_E2` and `ST_TNBC_45_CN23_C1`; E1 and E2 from specimen 32 belong
  to the same patient.

The cohort is not used for base-model training or selection and has no native
8 µm RNA ground truth. The 100 µm ST v1 geometry requires appropriate
preprocessing and whole-section inference adapters; **raw ST v1 inputs cannot
be used directly as Spot55 fine-tuning inputs in this package**. Downstream
whole-section stitching averages predictions across positions and fills gaps;
the final stitched maps are not claimed to preserve strict per-spot conservation.
This page describes the data and analysis scope. The package does not include
an ST v1 whole-section adapter or cohort inference outputs.

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
