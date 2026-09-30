"""Prepare one 256um FOV for python -m astra predict using registered RGB and H&E features.

Common observations.npz fields:
  parent_counts[P,N], gene_ids[N] (Unicode), owner_map[128,128] (int64, -1 gap),
  parent_valid[P] (bool), field_valid[128,128] (bool), protocol_id (0=3prime, 1=WT;
  assay identity, independent of HD16/Spot55 observation geometry),
  cell_um=2.0, physical_extent_um=[256.0,256.0]. Optional gene_available[N]
  defaults to True for each supplied gene; absent model genes become unavailable.

Choose exactly one image source:
  A. source_rgb_uint8[H,W,3]: an UNRESAMPLED microscope-pixel crop, including
     at least three source pixels of halo around the mapped FOV;
     source_rgb_is_native=True and spot_to_rgb[3,3]: local 2um (column,row)
     integer cell centers -> crop RGB (x,y) integer pixel centers.
     With original registration T, global FOV origin (row0,col0), and source
     crop origin (crop_y,crop_x), the required matrix is
       translate(-crop_x,-crop_y) @ T @ translate(col0,row0).
     RGB-to-OD/H/E is applied BEFORE bilinear 8x8 midpoint area integration.
  B. density_2um[128,128,3] float32 (OD,H,E), valid_area_2um[128,128] uint16,
     density_method='native_RGB_OD_HE_bilinear_midpoint_8x8', plus
     rgb[3,224,224] float32 in [0,1], rgb_valid[224,224] bool and
     rgb_sampling='registered_full_FOV_bilinear_pixel_centers_no_crop'.
     Both supplied image representations must describe the same complete FOV.

An already-resized RGB128 or RGB224 alone cannot reconstruct the original
area-integrated H&E features. This tool does not infer registration, stain
normalization, segmentation, a gene panel, or parent counts from raw 10x data.
UNI weights and network access are unnecessary here. UNI's
ImageNet normalization is applied later by python -m astra predict's frozen UNI extractor.
"""

import json

import numpy as np
import torch


from astra.data.preprocessing import _mapped_bounds, aggregate_density_region, full_valid_cells, quantize_valid_fraction, image_features_from_raw, sample_registered_rgb
from astra.model.ownership import validate_owner_batch


DENSITY_METHOD = "native_RGB_OD_HE_bilinear_midpoint_8x8"
RGB_SAMPLING = "registered_full_FOV_bilinear_pixel_centers_no_crop"


class ArrayRGBReader:
    """A single in-memory tile implementing the RGB reader interface."""

    def __init__(self, rgb):
        self.rgb = rgb
        self.height, self.width = rgb.shape[:2]
        self.tile_height, self.tile_width, self.tile_columns = self.height, self.width, 1

    def read_tile(self, index):
        if index != 0:
            raise IndexError("array RGB reader contains exactly one tile")
        return self.rgb


def scalar(source, name):
    value = np.asarray(source[name])
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return value.item()


def string_genes(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "US":
        raise ValueError(f"{name} must be a one-dimensional string array, not pickled objects")
    ids = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains empty or duplicate gene IDs")
    return ids


def images(source, field_valid):
    native = "source_rgb_uint8" in source
    if native == ("density_2um" in source):
        raise ValueError("supply exactly one image source: native RGB crop or exact density_2um")
    if native:
        rgb_source = np.asarray(source["source_rgb_uint8"])
        if (rgb_source.dtype != np.uint8 or rgb_source.ndim != 3 or rgb_source.shape[-1] != 3
                or min(rgb_source.shape[:2]) < 4 or scalar(source, "source_rgb_is_native") is not True):
            raise ValueError("source_rgb_uint8 must contain declared unresampled uint8 microscope RGB pixels")
        transform = np.asarray(source["spot_to_rgb"], dtype=np.float64)
        if transform.shape != (3, 3) or not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-12:
            raise ValueError("spot_to_rgb must be a finite invertible local-center registration matrix")
        reader = ArrayRGBReader(rgb_source)
        xmin, xmax, ymin, ymax = _mapped_bounds(transform, row_start=0, row_stop=128,
                                               column_start=0, column_stop=128)
        if xmin < 2.5 or ymin < 2.5 or xmax > reader.width-3.5 or ymax > reader.height-3.5:
            raise ValueError("native RGB crop must include the complete mapped FOV and three source pixels of halo")
        density, coverage = aggregate_density_region(reader, transform, row_start=0, row_stop=128,
            column_start=0, column_stop=128, samples_per_cell_axis=8)
        valid_area = quantize_valid_fraction(coverage)
        full_valid = full_valid_cells(transform, row_start=0, row_stop=128, column_start=0,
            column_stop=128, image_width=reader.width, image_height=reader.height)
        if np.any(field_valid & ~full_valid):
            raise ValueError("valid field cells extend beyond source RGB support")
        rgb, rgb_valid = sample_registered_rgb(reader, transform, np.asarray([0, 0]), output_size=224)
        rgb = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        mode = "native_rgb_crop"
    else:
        if scalar(source, "density_method") != DENSITY_METHOD or scalar(source, "rgb_sampling") != RGB_SAMPLING:
            raise ValueError("precomputed density/RGB must declare the exact feature algorithms and common FOV")
        density, valid_area = np.asarray(source["density_2um"]), np.asarray(source["valid_area_2um"])
        rgb, rgb_valid = np.asarray(source["rgb"]), np.asarray(source["rgb_valid"])
        mode = "precomputed_density_and_registered_rgb"
    if (density.shape != (128, 128, 3) or density.dtype != np.float32
            or valid_area.shape != (128, 128) or valid_area.dtype != np.uint16
            or not np.isfinite(density).all() or np.any(density < 0)
            or np.any(density[valid_area == 0] != 0) or np.any(valid_area[field_valid] != 65535)):
        raise ValueError("raw OD/H/E density and uint16 area must be valid; valid field cells require full image area")
    if (rgb.shape != (3, 224, 224) or rgb.dtype != np.float32 or not np.isfinite(rgb).all()
            or np.any((rgb < 0) | (rgb > 1)) or rgb_valid.shape != (224, 224) or rgb_valid.dtype != np.bool_):
        raise ValueError("UNI RGB must be float32 [3,224,224] in [0,1] with boolean validity")
    if not rgb_valid.reshape(14, 16, 14, 16).all((1, 3)).any():
        raise ValueError("RGB must contain at least one fully valid 16x16 UNI token")
    if np.any(rgb[:, ~rgb_valid] != 1):
        raise ValueError("outside-image RGB samples must be white, matching registered RGB sampling")
    features = image_features_from_raw(torch.from_numpy(np.ascontiguousarray(density))[None],
        torch.from_numpy(np.ascontiguousarray(valid_area))[None], torch.from_numpy(field_valid)[None])
    return np.ascontiguousarray(features[0].permute(2, 0, 1).numpy()), rgb, rgb_valid, mode


def prepare_arrays(source, model_gene_ids):
    """Prepare one FOV from a mapping/NpzFile; never read undeclared extra arrays."""
    genes = string_genes(model_gene_ids, "model_gene_ids")
    if len(genes) != 2000:
        raise ValueError("the released model requires its exact ordered 2000-gene panel")
    input_genes = string_genes(source["gene_ids"], "gene_ids")
    if scalar(source, "cell_um") != 2 or not np.array_equal(source["physical_extent_um"], [256., 256.]):
        raise ValueError("owner, counts and both image branches must share the complete 256um FOV on a 2um grid")
    owner, valid, field = (np.asarray(source[name]) for name in ("owner_map", "parent_valid", "field_valid"))
    counts = np.asarray(source["parent_counts"])
    protocol = scalar(source, "protocol_id")
    if (owner.shape != (128, 128) or owner.dtype != np.int64 or field.shape != owner.shape
            or field.dtype != np.bool_ or not field.any() or valid.ndim != 1 or valid.dtype != np.bool_
            or valid.size < 1 or counts.shape != (valid.size, len(input_genes))
            or counts.dtype.kind not in "iuf" or isinstance(protocol, bool) or protocol not in (0, 1)
            or np.asarray(source["protocol_id"]).dtype.kind not in "iu"):
        raise ValueError("observation count/owner/validity shapes or protocol ID differ from the single-FOV schema")
    validate_owner_batch(torch.from_numpy(owner)[None], torch.from_numpy(valid)[None], torch.from_numpy(field)[None])
    supplied_available = np.asarray(source.get("gene_available", np.ones(len(input_genes), dtype=bool)))
    if supplied_available.shape != (len(input_genes),) or supplied_available.dtype != np.bool_:
        raise ValueError("gene_available must be boolean and follow the supplied gene_ids order")
    observed = valid[:, None] & supplied_available[None]
    if not np.isfinite(counts[observed]).all() or np.any(counts[observed] < 0):
        raise ValueError("observed available parent counts must be finite and nonnegative")
    counts = np.where(observed, counts, 0).astype(np.float64)
    lookup = {gene: index for index, gene in enumerate(input_genes)}
    ordered = np.zeros((valid.size, len(genes)), dtype=np.float64)
    available = np.zeros(len(genes), dtype=bool)
    for output, gene in enumerate(genes):
        index = lookup.get(gene)
        if index is not None and supplied_available[index]:
            ordered[:, output], available[output] = counts[:, index], True
    if not available.any():
        raise ValueError("no observed assay gene overlaps the released panel")
    features, rgb, rgb_valid, mode = images(source, field)
    metadata = dict(image_source=mode, cell_um=2.0, physical_extent_um=[256.0, 256.0],
                    density_method=DENSITY_METHOD, rgb_sampling=RGB_SAMPLING,
                    feature_order=["log1p_OD", "log1p_H", "log1p_E", "relative_log_OD", "relative_log_H", "relative_log_E"],
                    relative_log_epsilon=1e-6, relative_log_clip=5.0, valid_area_scale=65535,
                    input_genes=len(input_genes), available_model_genes=int(available.sum()))
    result = dict(parent_counts=ordered, owner_map=owner, parent_valid=valid, field_valid=field,
                image_features_2um=features, gene_ids=np.asarray(genes), gene_available=available,
                protocol_id=np.asarray(protocol, dtype=np.int64), rgb=rgb, rgb_valid=rgb_valid,
                preparation_metadata_json=np.asarray(json.dumps(metadata)))
    if "parent_centers_yx_um" in source:
        centers = np.asarray(source["parent_centers_yx_um"], dtype=np.float64)
        if centers.shape != (valid.size, 2) or not np.isfinite(centers[valid]).all():
            raise ValueError("parent_centers_yx_um must provide each valid parent local center")
        result["parent_centers_yx_um"] = centers
    return result


def self_test():
    genes = np.asarray([f"gene_{index}" for index in range(2000)])
    owner = np.full((128, 128), -1, dtype=np.int64)
    owner[10:18, 20:28] = 0
    source = dict(parent_counts=np.asarray([[7., 3.]]), gene_ids=genes[[1, 0]],
                  owner_map=owner, parent_valid=np.asarray([True]), field_valid=np.ones_like(owner, dtype=bool),
                  protocol_id=np.asarray(0), cell_um=np.asarray(2.), physical_extent_um=np.asarray([256., 256.]),
                  source_rgb_uint8=np.full((148, 148, 3), 127, dtype=np.uint8), source_rgb_is_native=np.asarray(True),
                  spot_to_rgb=np.asarray([[1., 0., 10.], [0., 1., 10.], [0., 0., 1.]]))
    output = prepare_arrays(source, genes)
    assert output["parent_counts"][0, :3].tolist() == [3., 7., 0.]
    assert output["gene_available"].sum() == 2 and output["image_features_2um"].shape == (6, 128, 128)
    assert np.allclose(output["image_features_2um"][3:], 0, atol=1e-6)
    assert np.allclose(output["rgb"], 127/255, atol=1e-7) and output["rgb_valid"].all()
    reader = ArrayRGBReader(source["source_rgb_uint8"])
    density, fraction = aggregate_density_region(reader, source["spot_to_rgb"], row_start=0, row_stop=128,
        column_start=0, column_stop=128, samples_per_cell_axis=8)
    precomputed = {key: value for key, value in source.items()
                   if key not in ("source_rgb_uint8", "source_rgb_is_native", "spot_to_rgb")}
    precomputed.update(density_2um=density, valid_area_2um=quantize_valid_fraction(fraction),
                       density_method=np.asarray(DENSITY_METHOD), rgb=output["rgb"], rgb_valid=output["rgb_valid"],
                       rgb_sampling=np.asarray(RGB_SAMPLING))
    reference = prepare_arrays(precomputed, genes)
    for name in ("image_features_2um", "rgb", "rgb_valid", "parent_counts", "gene_available"):
        assert np.array_equal(output[name], reference[name]), name
    source["source_rgb_is_native"] = np.asarray(False)
    try:
        prepare_arrays(source, genes)
    except ValueError:
        pass
    else:
        raise AssertionError("a resized RGB patch was accepted as native source pixels")
    print(json.dumps(dict(status="passed", native_and_density_paths_exact=True, gene_mapping=True,
                          nonlinear_HE_resize_shortcut_rejected=True)))
