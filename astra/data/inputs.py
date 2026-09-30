"""Validate coarse observations and load frozen UNI representations."""
import numpy as np
import torch
import torch.nn.functional as F

from astra.runtime import ROOT, sha256


def load_input(path, gene_ids, *, keys=None):
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in (source.files if keys is None else keys) if key in source.files}
    if not np.array_equal(data["gene_ids"], np.asarray(gene_ids)):
        raise ValueError("gene_ids must match model/input_gene_ids.json; use python -m astra prepare-input")
    shapes = dict(owner_map=(128, 128), field_valid=(128, 128),
                  image_features_2um=(6, 128, 128), gene_available=(len(gene_ids),), protocol_id=())
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
    valid = data["parent_valid"]
    if valid.ndim != 1 or valid.size < 1 or data["parent_counts"].shape != (valid.size, len(gene_ids)):
        raise ValueError("parent_valid must be [P] and parent_counts must be [P,2000]")
    for key in ("field_valid", "parent_valid", "gene_available"):
        if data[key].dtype != np.bool_:
            raise ValueError(f"{key} must be boolean")
    if data["owner_map"].dtype != np.int64 or data["protocol_id"].dtype != np.int64:
        raise ValueError("owner_map and protocol_id must be int64")
    if int(data["protocol_id"]) not in (0, 1):
        raise ValueError("protocol_id must be 0 (3prime) or 1 (WT), independently of geometry")
    for key in ("parent_counts", "image_features_2um"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"{key} contains nonfinite values")
    if (data["parent_counts"] < 0).any() or not data["gene_available"].any():
        raise ValueError("counts must be nonnegative and at least one input gene must be measured")
    return data



def semantic_input(data, device, metadata, uni_checkpoint):
    if "pathology_features" in data:
        if uni_checkpoint is not None:
            raise ValueError("cached UNI features are already present; omit --uni-checkpoint")
        features, valid = data["pathology_features"], data["pathology_valid"]
        identity = data["uni_checkpoint_sha256"]
        if (features.shape != (1024, 14, 14) or features.dtype != np.float32
                or valid.shape != (14, 14) or valid.dtype != np.bool_
                or not np.isfinite(features).all() or not valid.any()
                or identity.shape != () or str(identity.item()) != metadata["uni_checkpoint_sha256"]):
            raise ValueError("cached UNI feature shape, validity, dtype or weight identity differs")
        return (torch.from_numpy(features)[None].to(device),
                torch.from_numpy(valid)[None].to(device), "precomputed_UNI1")
    if uni_checkpoint is None:
        raise ValueError("RGB inputs require --uni-checkpoint pointing to local UNI1 weights")
    rgb, valid = data["rgb"], data["rgb_valid"]
    if (rgb.shape != (3, 224, 224) or valid.shape != (224, 224)
            or valid.dtype != np.bool_ or not np.isfinite(rgb).all()
            or (rgb < 0).any() or (rgb > 1).any()):
        raise ValueError("RGB must be [3,224,224] in [0,1], with boolean rgb_valid[224,224]")
    if sha256(uni_checkpoint) != metadata["uni_checkpoint_sha256"]:
        raise ValueError("UNI checkpoint differs from the training UNI1 weights")
    from astra.model.pathology import FrozenUNI
    uni = FrozenUNI(uni_checkpoint, config_path=ROOT / "model/uni_config.json").to(device).eval()
    mask = torch.from_numpy(valid)[None, None].to(device).float()
    mask = F.avg_pool2d(mask, 16, 16)[:, 0] == 1
    if not bool(mask.any()):
        raise ValueError("RGB contains no fully valid UNI token")
    # A caller's NumPy strides must not select a different UNI convolution layout.
    rgb_tensor = torch.from_numpy(rgb)[None].to(device=device, dtype=torch.float32).contiguous()
    features = uni(rgb_tensor, extent_um=256.0)
    return features, mask, "local_UNI1_RGB"
