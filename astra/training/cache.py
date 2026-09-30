"""Read immutable sparse count, H&E and UNI artifacts with explicit section roles.

The native count records remain sparse until they are reduced to observed parent
totals and supervised 8um targets. No dense native-resolution gene grid is built.
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from astra.training.generator import OwnerMapSample, canonical_hd16_owner
from astra.data.image_features import image_features_from_raw
from astra.training.panels import GenePanels
from astra.training.sparse import collate_sparse_counts
from astra.training.utils import ROOT, artifact_path


@dataclass(frozen=True)
class PreparedFields:
    """One CPU batch, before any device-dependent floating-point operations."""

    keys: tuple
    indices: torch.Tensor
    counts: torch.Tensor
    density: torch.Tensor
    area: torch.Tensor
    valid: torch.Tensor
    available: torch.Tensor
    protocol: torch.Tensor


@dataclass(frozen=True)
class PreparedFeatures:
    keys: tuple
    features: torch.Tensor | None
    valid: torch.Tensor | None


def _cpu_tensor(values, *, pin_memory):
    tensor = torch.as_tensor(values)
    if tensor.device.type != "cpu":
        raise ValueError("cache preparation must stay on CPU")
    return tensor.pin_memory() if pin_memory and not tensor.is_pinned() else tensor


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def authorized_ids(config, role):
    if role == "train":
        return config["wt"]["train"] + config["three_prime"]["auxiliary_train"]
    if role == "validation":
        return config["wt"]["validation"] + config["three_prime"].get("validation", [])
    raise PermissionError("the fitting loader only supports train and validation")


def validate_resources(config):
    """Resolve section identities exclusively through the repository registry."""
    if config["resource_registry"] != "resource.json":
        raise ValueError("resource.json is the only resource registry")
    registry = read_json(ROOT / "resource.json")
    training, validation = authorized_ids(config, "train"), authorized_ids(config, "validation")
    if len(training + validation) != len(set(training + validation)):
        raise ValueError("duplicate or overlapping train/validation sections")
    excluded = set(config["wt"].get("test", []) + config["wt"].get("downstream_reserved", [])
                   + config["wt"].get("excluded_tissue_arrays", []) + config["three_prime"].get("test", []))
    if excluded.intersection(training + validation):
        raise PermissionError("held-out or reserved sections cannot enter fitting")
    for rid in training + validation:
        record = registry["datasets"][rid]
        expected = "visium_hd_3prime" if rid in (config["three_prime"]["auxiliary_train"]
            + config["three_prime"].get("validation", [])) else "visium_hd_wt"
        if record.get("collection") != expected or not record.get("feature_slice") or not record.get("tissue_image"):
            raise ValueError(f"resource protocol or required assets differ: {rid}")
        if any(Path(record[key]).is_absolute() for key in ("feature_slice", "tissue_image")):
            raise ValueError("registry data paths must be relative to the repository")
    if "dataset_split" in config:
        split = read_json(artifact_path(config["dataset_split"]))
        folds = split["cross_validation"]["validation_folds"]
        dev = {rid for ids in folds.values() for rid in ids}
        tests = {rid for ids in split["fixed_test_groups"].values() for rid in ids}
        final = config.get("final_refit")
        if final:
            if (final.get("scope") != "all_12_development_from_scratch" or set(training) != dev or validation
                    or final.get("model_strategy") != split["fit_boundaries"]["final_model_strategy"]
                    or final.get("epoch") != config["training"]["max_epochs"]):
                raise PermissionError("final refit must use all development sections and the predefined single-model budget")
        else:
            val = set(folds[config["fold"]])
            if set(training) != dev - val or set(validation) != val:
                raise PermissionError("job roles differ from the experiment's shared dataset split")
        if not tests <= excluded or config.get("split_id") != split["split_id"]:
            raise PermissionError("job roles differ from the experiment's shared dataset split")
    if config["model"].get("use_pathology", True):
        if config["cache"]["uni_resource_id"] not in registry["model_weights"]:
            raise ValueError("UNI model resource is absent from resource.json")
    return registry


def fixed_outer_preprocessing(config):
    """Explicit conditional audit of an already frozen outer-training task.

    This is not independent inner-fold feature selection: gene identities and
    image normalization were fixed using the outer training cohort. Never
    relabel their original fit provenance as inner-training-only.
    """
    final_reference = config.get("final_refit_preprocessing_reference")
    if final_reference is not None:
        if not config.get("final_refit"):
            raise PermissionError("a final-refit preprocessing reference requires the final-refit data scope")
        validate_resources(config)
        reference = read_json(artifact_path(final_reference))
        validate_resources(reference)
        colors = [{k: v for k, v in c["image_preprocessing"].items() if k != "cache_root"}
                  for c in (config, reference)]
        if (reference.get("final_refit") or reference.get("fixed_outer_preprocessing")
                or not set(authorized_ids(reference, "train")) <= set(authorized_ids(config, "train"))
                or config["panel_artifact"] != reference["panel_artifact"] or colors[0] != colors[1]
                or config["split_id"] != reference["split_id"]):
            raise PermissionError("final refit must preserve the selected development-only panel and color provenance")
        return reference
    policy = config.get("fixed_outer_preprocessing")
    if policy is None:
        return config
    if (policy.get("purpose") != "conditional_internal_transfer_diagnostic"
            or policy.get("independent_zero_shot_test") is not False):
        raise PermissionError("frozen outer preprocessing is restricted to conditional diagnostics")
    reference = read_json(artifact_path(policy["reference_config"]))
    if "fixed_outer_preprocessing" in reference:
        raise PermissionError("nested preprocessing references are not supported")
    validate_resources(reference)
    colors = [{k: v for k, v in c["image_preprocessing"].items() if k != "cache_root"}
              for c in (config, reference)]
    if (set(authorized_ids(config, "train") + authorized_ids(config, "validation"))
            != set(authorized_ids(reference, "train"))
            or config["panel_artifact"] != reference["panel_artifact"]
            or colors[0] != colors[1]):
        raise PermissionError("conditional audit must retain the exact outer-training cohort and fitted artifacts")
    return reference


def load_panels(config):
    record = read_json(artifact_path(config["panel_artifact"]))
    reference = fixed_outer_preprocessing(config)
    if config.get("panel_policy") == "reuse_published_panel":
        from astra.assets import ROOT as package_root
        published = read_json(package_root / "model/input_gene_ids.json")
        if (record.get("frozen") is not True or record["input_gene_ids"] != published
                or record["output_gene_ids"] != published):
            raise ValueError("user training must preserve the published ordered panel and its original fitting provenance")
    elif record.get("frozen") is not True or record.get("fit_sections") != authorized_ids(reference, "train"):
        raise ValueError("panel must be frozen and fitted on exactly the configured training sections")
    if "dataset_split" in config and record.get("spatial_split_artifact") != reference["cache"]["spatial_split_artifact"]:
        raise ValueError("panel fitting must exclude this fold's spatial monitor")
    panels = GenePanels(record["input_gene_ids"], record["output_gene_ids"])
    input_size = config.get("input_panel_size", config["panel_size"])
    output_size = config.get("output_panel_size", config["panel_size"])
    if (type(input_size) is not int or type(output_size) is not int
            or len(panels.input_gene_ids) != input_size or len(panels.output_gene_ids) != output_size
            or config["panel_size"] != output_size):
        raise ValueError("configured panel size differs from its frozen artifact")
    return panels


class BlockStore:
    """Memory-map one exact, read-only native sparse block cache."""

    def __init__(self, record, root, panels):
        self.record, self.root, self.panels = record, Path(root), panels
        files = dict(blocks="block_starts_yx.npy", indptr="block_indptr.npy", local="local_linear.npy",
                     genes="gene_index.npy", counts="count_uint16.npy", density="density_float32.npy",
                     area="valid_area_uint16.npy", valid="full_valid_bool.npy", starts="starts_yx.npy")
        for name, filename in files.items():
            setattr(self, name, np.load(self.root / filename, mmap_mode="r", allow_pickle=False))
        role = record["role"]
        self.slots = np.load(self.root / f"{role}_block_slot.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(self.root / f"{role}_offset_yx.npy", mmap_mode="r", allow_pickle=False)
        blocks, fields = record["blocks"], record["fields"]
        if (record["block_cells_2um"] != 256 or record["patch_cells_2um"] != 128
                or self.blocks.shape != (blocks, 2) or self.indptr.shape != (blocks + 1,)
                or self.starts.shape != (fields, 2) or self.slots.shape != (fields,)
                or self.offsets.shape != (fields, 2) or self.indptr[0] != 0
                or np.any(np.diff(self.indptr) < 0)
                or self.local.shape != self.genes.shape or self.genes.shape != self.counts.shape
                or self.local.shape != (int(self.indptr[-1]),)
                or self.local.dtype != np.uint16 or self.genes.dtype != np.uint16
                or self.counts.dtype != np.uint16 or self.density.dtype != np.float32
                or self.density.shape != (blocks, 256, 256, 3)
                or self.area.shape != (blocks, 256, 256) or self.area.dtype != np.uint16
                or self.valid.shape != (blocks, 256, 256) or self.valid.dtype != np.bool_):
            raise ValueError("native sparse cache shape or precision differs")
        if np.any(self.slots >= blocks) or np.any(self.offsets > 128):
            raise ValueError("field lies outside its cached block")
        np.testing.assert_array_equal(self.blocks[self.slots] + self.offsets, self.starts)
        self.availability = np.asarray(record["gene_available"], dtype=bool)
        cached_genes = record["input_gene_ids"]
        positions = {gene: index for index, gene in enumerate(cached_genes)}
        self.gene_lookup = np.full(len(cached_genes), -1, dtype=np.int64)
        selected = [positions[gene] for gene in panels.input_gene_ids]
        self.gene_lookup[selected] = np.arange(len(selected))
        self.availability = self.availability[selected]
        if self.availability.shape != (len(panels.input_gene_ids),):
            raise ValueError("cached gene availability differs from the input panel")

    def locations(self, role, field_indices):
        if role != self.record["role"]:
            raise PermissionError("cannot reinterpret a cached section's role")
        indices = np.asarray(field_indices)
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= self.record["fields"])):
            raise IndexError("field index lies outside its authorized cache")
        return [(int(self.slots[i]), *map(int, self.offsets[i])) for i in indices]

    def sparse_entries(self, locations, output_indices):
        if len(locations) != len(output_indices) or not locations:
            raise ValueError("sparse field indices and output slots differ")
        grouped, index_parts, count_parts = defaultdict(list), [], []
        for output, (slot, y, x) in zip(output_indices, locations, strict=True):
            grouped[slot].append((output, y, x))
        gene_count = len(self.panels.input_gene_ids)
        for slot, requests in grouped.items():
            start, stop = map(int, self.indptr[slot:slot + 2])
            local = np.asarray(self.local[start:stop], dtype=np.int64)
            rows, columns = local // 256, local % 256
            genes = np.asarray(self.genes[start:stop], dtype=np.int64)
            if np.any(genes >= len(self.gene_lookup)):
                raise ValueError("sparse gene index lies outside the cached panel")
            genes = self.gene_lookup[genes]
            counts = np.asarray(self.counts[start:stop], dtype=np.int32)
            for output, y, x in requests:
                keep = ((rows >= y) & (rows < y + 128) & (columns >= x)
                        & (columns < x + 128) & (genes >= 0) & self.availability[genes.clip(0)]
                        & self.valid[slot, rows, columns])
                indices = (((int(output) * 128 + rows[keep] - y) * 128 + columns[keep] - x)
                           * gene_count + genes[keep])
                index_parts.append(indices)
                count_parts.append(counts[keep])
        return np.concatenate(index_parts), np.concatenate(count_parts)

    def image_patch(self, location):
        slot, y, x = location
        return tuple(np.asarray(array[slot, y:y + 128, x:x + 128])
                     for array in (self.density, self.area, self.valid))


def open_training_store(config, root, panels, *, role="train"):
    root = Path(root).resolve()
    if not root.is_relative_to(ROOT):
        raise PermissionError("count/image cache must be a repository artifact")
    record = read_json(root / "section.json")
    if (record["role"] != role or record["resource_id"] not in authorized_ids(config, role)
            or record["resource_id"] != root.name):
        raise PermissionError("cache lies outside the configured fitting role")
    cached = GenePanels(record["input_gene_ids"], record["output_gene_ids"])
    # Every output is re-aggregated from the native input-panel sparse records.
    panel_matches = (set(panels.input_gene_ids) <= set(cached.input_gene_ids)
        if config["cache"].get("allow_gene_superset", False) else cached.input_gene_ids == panels.input_gene_ids)
    if record.get("status") != "complete" or not panel_matches:
        raise ValueError("incomplete cache or mismatched input gene identities/order")
    expected = 0 if record["resource_id"] in (config["three_prime"]["auxiliary_train"]
        + config["three_prime"].get("validation", [])) else 1
    if record["protocol_id"] != expected:
        raise ValueError("cached protocol disagrees with the resource role")
    return BlockStore(record, root, panels)


class CachedFields:
    """Persistent mappings, opened only for the explicitly requested fitting IDs."""

    def __init__(self, config, panels, device="cpu", *, roles=("train", "validation"), resource_ids=None,
                 layout_tasks=None):
        validate_resources(config)
        self.config, self.panels, self.device = config, panels, torch.device(device)
        self.entries, self.stores, self.starts = [], {}, {}
        color = config.get('image_preprocessing', {'mode': 'raw'})
        color_root = None
        if color['mode'] == 'tissue_masked_cielab_reinhard':
            color_root = artifact_path(color['cache_root'])
            summary = read_json(color_root / 'summary.json')
            if summary['status'] != 'complete' or summary['settings'] != color:
                raise ValueError('normalized H&E cache is incomplete or uses different settings')
            statistics_path = artifact_path(color['statistics_artifact'])
            statistics = read_json(statistics_path)
            if statistics['reference_sections'] != authorized_ids(fixed_outer_preprocessing(config), 'train'):
                raise ValueError('H&E reference must contain exactly the training sections')
            color_sha = hashlib.sha256(statistics_path.read_bytes()).hexdigest()
        elif color['mode'] != 'raw':
            raise ValueError('unsupported H&E preprocessing')
        permitted = [rid for role in roles for rid in authorized_ids(config, role)]
        wanted = set(permitted if resource_ids is None else resource_ids)
        if not wanted or not wanted <= set(permitted):
            raise PermissionError("requested cache IDs lie outside the authorized roles")
        for role in roles:
            for rid in authorized_ids(config, role):
                if rid not in wanted:
                    continue
                root = artifact_path(config["cache"][f"{role}_root"]) / rid
                store = open_training_store(config, root, panels, role=role)
                if color_root is not None:
                    record = summary['sections'][rid]
                    if record['statistics_sha256'] != color_sha or record['source_cache'] != str(root.relative_to(ROOT)):
                        raise ValueError('normalized H&E parameter or source-cache identity differs')
                    normalized = np.load(color_root / rid / 'density_float32.npy', mmap_mode='r', allow_pickle=False)
                    np.testing.assert_array_equal(np.load(color_root / rid / 'block_starts_yx.npy', allow_pickle=False), store.blocks)
                    if normalized.shape != store.density.shape or normalized.dtype != np.float32:
                        raise ValueError('normalized H&E and native cache geometry differ')
                    store.density = normalized
                self.stores[rid], self.starts[rid] = store, store.starts
                self.entries.append(dict(section=rid, role=role, protocol_id=store.record["protocol_id"],
                                         fields=store.record["fields"]))
        self.limits = {(e["section"], e["role"], e["protocol_id"]): e["fields"] for e in self.entries}
        self.layout_records, self.layout_owners, self.random_owners = {}, {}, {}
        tasks = (set(layout_tasks) if layout_tasks is not None else
                 {"HD16", "Spot55"} | ({"random"} if "random" in
                     (*config.get("validation_tasks", []), *config.get("final_evaluation_tasks", [])) else set()))
        if not tasks or not tasks <= {"HD16", "Spot55", "random"}:
            raise ValueError("unsupported requested validation layouts")
        if tasks == {"HD16"}:
            return
        layout_root = artifact_path(config["cache"]["validation_layout_root"])
        if any(e["role"] == "validation" for e in self.entries):
            if read_json(layout_root / "summary.json")["status"] != "complete":
                raise ValueError("fixed validation layout preparation is incomplete")
        for entry in self.entries:
            if entry["role"] != "validation":
                continue
            rid = entry["section"]
            record = read_json(layout_root / rid / "section.json")
            if (record["resource_id"] != rid or record["role"] != "validation"
                    or record["fields"] != entry["fields"]):
                raise ValueError("fixed validation layout identity differs")
            np.testing.assert_array_equal(np.load(layout_root / rid / "starts_yx.npy", allow_pickle=False), self.starts[rid])
            self.layout_records[rid] = record
            if "Spot55" in tasks:
                owners = np.load(layout_root / rid / "spot55_owner.npy", mmap_mode="r", allow_pickle=False)
                if owners.shape != (entry["fields"], 128, 128):
                    raise ValueError("fixed Spot55 layout dimensions differ")
                self.layout_owners[rid] = owners
            if "random" in tasks:
                random_owners = np.load(layout_root / rid / "random_owner.npy", mmap_mode="r", allow_pickle=False)
                count = config["training"]["random_validation_fields_per_section"]
                if random_owners.shape != (count, 128, 128) or len(record["random_parents_per_fov"]) != count:
                    raise ValueError("fixed random validation layouts differ")
                self.random_owners[rid] = random_owners

    def _keys(self, items, role):
        if not items:
            raise ValueError("cannot load an empty batch")
        keys = []
        for item in items:
            rid, protocol, index = item["sample"], item["protocol_id"], item["field_index"]
            limit = self.limits.get((rid, role, protocol))
            if type(index) is not int or type(protocol) is not int or limit is None or not 0 <= index < limit:
                raise PermissionError("field lies outside its configured section, role or protocol")
            keys.append((rid, role, protocol, index))
        return tuple(keys)

    def prepare(self, items, role, *, pin_memory=None):
        """Read/pack only one batch on CPU; suitable for the single prefetch worker."""
        keys = self._keys(items, role)
        pin_memory = self.device.type == "cuda" if pin_memory is None else pin_memory
        grouped, sparse, images = defaultdict(list), [], [None] * len(keys)
        for out, (rid, _, _, index) in enumerate(keys):
            grouped[rid].append((out, index))
        for rid, requests in grouped.items():
            store = self.stores[rid]
            locations = store.locations(role, [index for _, index in requests])
            sparse.append(store.sparse_entries(locations, [out for out, _ in requests]))
            for (out, _), location in zip(requests, locations, strict=True):
                images[out] = store.image_patch(location)
        density, area, valid = (np.stack(values) for values in zip(*images))
        values = (np.concatenate([s[0] for s in sparse]), np.concatenate([s[1] for s in sparse]),
                  density, area, valid, np.stack([self.stores[k[0]].availability for k in keys]),
                  np.asarray([k[2] for k in keys], dtype=np.int64))
        return PreparedFields(keys, *(_cpu_tensor(value, pin_memory=pin_memory) for value in values))

    def materialize(self, prepared, items, masks, role):
        """Transfer on the caller's current stream, retaining the original GPU math."""
        keys = self._keys(items, role)
        if not isinstance(prepared, PreparedFields) or prepared.keys != keys:
            raise ValueError("prepared count batch identity or order differs")
        if len(masks) != len(keys):
            raise ValueError("owner map count differs from its fields")
        samples = []
        for mask in masks:
            if mask.owner_map.device.type != "cpu" and mask.owner_map.device != self.device:
                raise ValueError("owner map lies on another device")
            samples.append(OwnerMapSample(mask.owner_map.to(self.device, non_blocking=True),
                mask.parent_valid.to(self.device, non_blocking=True), mask.parameters))
        indices, counts, density, area, valid, available, protocol = (
            getattr(prepared, name).to(self.device, non_blocking=True)
            for name in ("indices", "counts", "density", "area", "valid", "available", "protocol"))
        image = image_features_from_raw(density, area, valid).permute(0, 3, 1, 2).contiguous()
        batch = collate_sparse_counts(indices, counts, image, valid, samples, self.panels)
        return batch, available, protocol

    def batch(self, items, masks, role):
        return self.materialize(self.prepare(items, role), items, masks, role)

    def validation_masks(self, items, task, *, device=None):
        self._keys(items, "validation")
        device = self.device if device is None else torch.device(device)
        if task == "HD16":
            return [canonical_hd16_owner(device=device)] * len(items)
        if task not in ("Spot55", "random"):
            raise ValueError("unknown validation task")
        result = []
        for item in items:
            rid, index = item["sample"], item["field_index"]
            if rid not in (self.layout_owners if task == "Spot55" else self.random_owners):
                raise ValueError(f"validation layout was not loaded: {task}")
            record = self.layout_records[rid]
            owners = self.layout_owners[rid] if task == "Spot55" else self.random_owners[rid]
            prefix = "" if task == "Spot55" else "random_"
            result.append(OwnerMapSample(torch.tensor(owners[index], dtype=torch.long,
                device=device), torch.ones(record[prefix + "parents_per_fov"][index], dtype=torch.bool,
                device=device), record[prefix + "fovs"][index]))
        return result

    def training_entries(self):
        spec = read_json(artifact_path(self.config["cache"]["spatial_split_artifact"]))
        if set(spec["sections"]) != set(authorized_ids(self.config, "train")):
            raise ValueError("spatial split must cover exactly the configured training sections")
        result = []
        for entry in self.entries:
            if entry["role"] != "train":
                continue
            rid = entry["section"]
            record, starts = spec["sections"][rid], self.starts[rid]
            selected = np.load(artifact_path(record["training_indices"]), allow_pickle=False)
            excluded = np.zeros(len(starts), dtype=bool)
            for origin in record["block_origins_yx_2um"]:
                origin = np.asarray(origin)
                excluded |= np.all((starts < origin + 256) & (starts + 128 > origin), axis=1)
            if (record["source_fields"] != entry["fields"]
                    or not np.array_equal(selected, np.flatnonzero(~excluded))):
                raise ValueError("training pool does not preserve the frozen spatial exclusion")
            for item in record["monitor_items"]:
                if item["sample"] != rid or not np.array_equal(starts[item["field_index"]], item["start_yx_2um"]):
                    raise ValueError("spatial monitor field identity differs")
            result.append(dict(entry, fields=len(selected), field_indices=selected))
        return result

    def spatial_monitor_items(self):
        """Read reserved FOV identities directly, without training-pool filtering.

        These are original cache row IDs inside the frozen exclusion blocks,
        not indices into the retained training pool. Only metadata and cached
        coordinates are inspected here; counts and image patches are untouched.
        """
        spec = read_json(artifact_path(self.config["cache"]["spatial_split_artifact"]))
        if set(spec["sections"]) != set(authorized_ids(self.config, "train")):
            raise ValueError("spatial monitor split must cover exactly the authorized training sections")
        items, seen = [], set()
        for entry in self.entries:
            if entry["role"] != "train":
                continue
            rid = entry["section"]
            record, starts = spec["sections"][rid], self.starts[rid]
            if (record["source_fields"] != entry["fields"]
                    or record["protocol_id"] != entry["protocol_id"]):
                raise ValueError("spatial monitor source pool or protocol differs")
            blocks = np.asarray(record["block_origins_yx_2um"])
            if (blocks.ndim != 2 or blocks.shape[1] != 2 or not len(blocks)
                    or not np.issubdtype(blocks.dtype, np.integer) or np.any(blocks < 0)):
                raise ValueError("spatial monitor blocks must be nonnegative integer coordinates")
            for original in record["monitor_items"]:
                item = dict(original)
                if item["sample"] != rid or item["protocol_id"] != entry["protocol_id"]:
                    raise PermissionError("spatial monitor field belongs to a different section or protocol")
                self._keys([item], "train")
                origin = np.asarray(item["start_yx_2um"])
                if (origin.shape != (2,) or not np.issubdtype(origin.dtype, np.integer)
                        or np.any(origin < 0) or np.any(origin % 8)
                        or not np.array_equal(starts[item["field_index"]], origin)):
                    raise ValueError("spatial monitor coordinate identity or HD16 alignment differs")
                if not np.any(np.all((origin >= blocks) & (origin + 128 <= blocks + 256), axis=1)):
                    raise ValueError("spatial monitor FOV is outside the frozen training-exclusion blocks")
                key = (rid, item["protocol_id"], item["field_index"])
                if key in seen:
                    raise ValueError("duplicate spatial monitor field")
                seen.add(key)
                items.append(item)
        return items


class CachedFeatures:
    """Read only requested FP32 UNI rows, retaining their exact coordinate match."""

    def __init__(self, config, fields, *, enabled=True):
        self.fields, self.device, self.enabled, self.parts = fields, fields.device, enabled, {}
        self.sha = None
        if not enabled:
            return
        summary = read_json(artifact_path(config["cache"]["uni_summary"]))
        if summary["status"] != "complete" or summary["dtype"] != "float32":
            raise ValueError("UNI cache is incomplete or has changed precision")
        self.sha = summary["checkpoint_sha256"]
        for entry in fields.entries:
            rid = entry["section"]
            root = artifact_path(config["cache"]["uni_root"]) / rid
            record = read_json(root / "section.json")
            offset = record.get("field_offset", 0)
            if record["fields"] != entry["fields"] or not 0 <= offset <= record["fields"]:
                raise ValueError("UNI and count field pools differ")
            paths = ([(root / record["prefix_cache"], 0, offset), (root, offset, record["fields"])]
                     if offset else [(root, 0, record["fields"])])
            parts = []
            for path, start, end in paths:
                if not path.resolve().is_relative_to(ROOT):
                    raise PermissionError("UNI artifact prefix lies outside this repository")
                meta = read_json(path / "section.json")
                if (meta["status"] != "complete" or meta["resource_id"] != rid or meta["role"] != entry["role"]
                        or meta["uni"]["checkpoint_sha256"] != self.sha):
                    raise ValueError("UNI section, role or model identity differs")
                np.testing.assert_array_equal(np.load(path / "starts_yx.npy", mmap_mode="r", allow_pickle=False),
                                              fields.starts[rid][start:end])
                features = np.load(path / "features_float32.npy", mmap_mode="r", allow_pickle=False)
                valid = np.load(path / "valid_bool.npy", mmap_mode="r", allow_pickle=False)
                if (features.dtype != np.float32 or features.shape != (end - start, 1024, 14, 14)
                        or valid.dtype != np.bool_ or valid.shape != (end - start, 14, 14)):
                    raise ValueError("UNI shape or dtype differs")
                parts.append((start, end, features, valid))
            self.parts[rid] = parts

    def prepare(self, items, role, *, pin_memory=None):
        keys = self.fields._keys(items, role)
        if not self.enabled:
            return PreparedFeatures(keys, None, None)
        pin_memory = self.device.type == "cuda" if pin_memory is None else pin_memory
        features = torch.empty((len(keys), 1024, 14, 14), dtype=torch.float32, pin_memory=pin_memory)
        valid = torch.empty((len(keys), 14, 14), dtype=torch.bool, pin_memory=pin_memory)
        feature_rows, valid_rows = features.numpy(), valid.numpy()
        for out, (rid, _, _, index) in enumerate(keys):
            for start, end, values, mask in self.parts[rid]:
                if start <= index < end:
                    np.copyto(feature_rows[out], values[index - start])
                    np.copyto(valid_rows[out], mask[index - start])
                    break
            else:
                raise IndexError("UNI row lies outside the mapped field pool")
        return PreparedFeatures(keys, features, valid)

    def encode(self, prepared, items, role):
        keys = self.fields._keys(items, role)
        if not isinstance(prepared, PreparedFeatures) or prepared.keys != keys:
            raise ValueError("prepared UNI batch identity or order differs")
        if not self.enabled:
            if prepared.features is not None or prepared.valid is not None:
                raise ValueError("disabled UNI cannot consume prepared features")
            return None, None
        if prepared.features is None or prepared.valid is None:
            raise ValueError("enabled UNI requires prepared features")
        return (prepared.features.to(self.device, non_blocking=True),
                prepared.valid.to(self.device, non_blocking=True))

    def batch(self, items, role):
        return self.encode(self.prepare(items, role), items, role)
