"""Self-contained sparse-cache training with whole-section validation."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from astra.training.cache import CachedFeatures, CachedFields, authorized_ids, load_panels, read_json, validate_resources
from astra.data.batching import prefetched_batches
from astra.training.checkpoint import load_checkpoint, save_checkpoint
from astra.training.diagnostics import diagnostic_metrics, monitor_items
from astra.training.layouts import core_supervision
from astra.training.optimization import bounded_selection, cosine_multiplier, thresholds
from astra.training.model_factory import model_class
from astra.training.parent_sampling import assign_parent_tasks, training_parent_masks
from astra.training.parent_views import training_epoch_items, full_batch_view_items, validate_view_policy
from astra.training.training import evaluate_batch, train_step, validate_loss_config
from astra.training.utils import ROOT, artifact_path

TASKS = ("HD16", "Spot55")
DEFAULT_CONFIG = ROOT / "training" / "config.json"


def validate_config(config):
    validate_resources(config)
    panels = load_panels(config)
    settings = config["training"]
    validate_view_policy(settings)
    if settings.get("learning_rate_schedule", "validation_plateau") not in (
            "validation_plateau", "constant_after_warmup", "warmup_cosine_over_actual_optimizer_steps"):
        raise ValueError("unknown learning-rate schedule")
    validate_loss_config(settings.get("training_loss"))
    if config.get("model_family", "v033") not in ("ASTRA", "v030", "v033"):
        raise ValueError("model_family must identify the published ASTRA architecture")
    warmup_clock(settings)
    milestones = settings.get("milestone_epochs", [])
    if (not isinstance(milestones, list) or len(set(milestones)) != len(milestones)
            or any(type(epoch) is not int or epoch <= 0 or epoch > settings["max_epochs"] for epoch in milestones)):
        raise ValueError("milestone_epochs must be distinct positive epochs within the budget")
    fraction = settings.get("cuda_memory_fraction", 1.0)
    if type(fraction) not in (float, int) or not 0 < fraction <= 1:
        raise ValueError("cuda_memory_fraction must lie in (0, 1]")
    threshold = settings.get("empty_cache_reserved_threshold_gib", 48)
    if type(threshold) not in (float, int) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("empty_cache_reserved_threshold_gib must be finite and positive")
    if settings.get("batch_pipeline", "cpu") not in ("cpu", "device"):
        raise ValueError("batch_pipeline must be cpu or device")
    mask_batch_size = settings.get("cuda_mask_batch_size", 96)
    if type(mask_batch_size) is not int or mask_batch_size <= 0:
        raise ValueError("cuda_mask_batch_size must be a positive integer")
    if type(settings["formal_training"]) is not bool:
        raise ValueError("formal_training must be explicitly boolean")
    if config["model"].get("use_fine_head", False):
        raise ValueError("the base CLI supervises 8um only; train the optional fine head via adaptation with reference fine labels")
    if config["model"].get("adaptation_mode", False):
        raise ValueError("adaptation_mode freezes the base; use the adaptation API instead of the base training CLI")
    for key in ("batch_size", "gradient_accumulation_steps", "max_epochs", "validation_interval",
                "validation_fields_per_section", "patience_checks", "cpu_threads", "plateau_patience_checks"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("seed", "minimum_epochs", "warmup_optimizer_steps"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"training.{key} must be a nonnegative integer")
    for key in ("learning_rate", "max_gradient_norm"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if not math.isfinite(settings["weight_decay"]) or settings["weight_decay"] < 0:
        raise ValueError("weight decay must be finite and nonnegative")
    if (set(settings["region_weights"]) != {"covered", "gap", "mixed"}
            or any(not math.isfinite(v) or v < 0 for v in settings["region_weights"].values())
            or not any(settings["region_weights"].values())):
        raise ValueError("region weights must be finite and nonnegative, with at least one positive region")
    if (settings["cpu_threads"] > 4 or not 0 < settings["plateau_factor"] < 1
            or not 0 < settings["minimum_learning_rate_ratio"] <= 1
            or settings["minimum_epochs"] > settings["max_epochs"]
            or settings["validation_fields_per_section"] != (4 if config.get("engineering_smoke_dataset", False) else 100)
            or config["supervision_core_um"] != 160):
        raise ValueError("thread limit, learning-rate schedule or frozen validation/core160 contract differs")
    if settings["sampling_mode"] not in ("fresh", "shuffled_cycles", "fresh_then_shuffled_cycles"):
        raise ValueError("unsupported deterministic field sampling mode")
    bounded = config["selection"]["rule"] == "bounded_regression_fixed_initial_anchor"
    holdout = config.get("final_refit", {}).get("selection") == "development_spatial_holdout"
    if bounded:
        thresholds(config["selection"]["minimum_improvement"])
        thresholds(config["selection"]["maximum_regression"])
        if not holdout or not config.get("diagnostics", {}).get("enabled"):
            raise ValueError("bounded final-refit selection requires explicit development spatial holdout")
    elif (config["selection"]["maximum_regression"] != 0
            or not math.isfinite(config["selection"]["minimum_improvement"])
            or config["selection"]["minimum_improvement"] < 0):
        raise ValueError("joint selection requires zero task regression and nonnegative improvement")
    selection_tasks = tuple(config["selection"].get("tasks", TASKS))
    if selection_tasks not in (TASKS, ("HD16",)):
        raise ValueError("selection must use legacy joint tasks or HD16 alone")
    expected_rule = ("bounded_regression_fixed_initial_anchor" if holdout else
                     "fixed_epoch" if config.get("final_refit") else "HD16_nll" if selection_tasks == ("HD16",) else
                     "both_tasks_nonincreasing_and_one_improves_against_same_accepted_checkpoint")
    if config["selection"]["rule"] != expected_rule:
        raise ValueError("selection rule and monitored tasks disagree")
    allowed_validation = (TASKS, (*TASKS, "random"), ("HD16",)) if config.get("final_refit") else (TASKS, (*TASKS, "random"))
    if tuple(config.get("validation_tasks", TASKS)) not in allowed_validation:
        raise ValueError("validation tasks must retain HD16/Spot55 and optionally fixed random layouts")
    final_tasks = tuple(config.get("final_evaluation_tasks", []))
    if final_tasks not in ((), ("random",)) or set(final_tasks).intersection(config.get("validation_tasks", TASKS)):
        raise ValueError("final-only evaluation may contain random, separately from routine validation")
    if "random" in (*config.get("validation_tasks", []), *final_tasks):
        count = settings.get("random_validation_fields_per_section")
        if type(count) is not int or not 3 <= count <= settings["validation_fields_per_section"]:
            raise ValueError("invalid fixed random validation budget")
    budgets = {rid: settings["section_field_overrides"].get(rid, settings["fields_per_section"])
               for rid in authorized_ids(config, "train")}
    if (set(settings["section_field_overrides"]) - set(budgets)
            or any(type(v) is not int or v <= 0 for v in budgets.values())):
        raise ValueError("each training section needs an explicit positive field exposure")
    if sum(budgets.values()) != settings["fields_per_epoch"]:
        raise ValueError("section exposure quotas do not match fields_per_epoch")
    smoke = config["smoke"]
    if (smoke["training_fields"] < 3 or smoke["training_fields"] > 16
            or smoke["epochs"] != 1 or smoke["validation_fields_per_section"] != 1
            or smoke["batch_size"] != 1):
        raise ValueError("smoke must retain all three observation paths within its fixed short budget")
    diagnostics = config.get("diagnostics", {})
    if type(diagnostics.get("enabled", False)) is not bool:
        raise ValueError("diagnostics.enabled must be boolean")
    for key in ("gene_groups_from_panel", "parent_count_strata", "allocation_decomposition",
                "gene_group_allocation_decomposition"):
        if type(diagnostics.get(key, False)) is not bool:
            raise ValueError(f"diagnostics.{key} must be boolean")
    if diagnostics.get("enabled", False):
        interval = diagnostics.get("interval_epochs", 10)
        if type(interval) is not int or interval <= 0:
            raise ValueError("diagnostics.interval_epochs must be positive")
    # Validate observation quotas without reading any expression values.
    assign_parent_tasks([], config, 1)
    return panels, budgets


def observe_smoke_inputs(digest, batch, available, protocol, semantic, *, include_image=True,
                         include_observation=True):
    """One aggregate identity check for the actual paired smoke observations."""
    tensors = ([batch.parent_counts, batch.owner_map, batch.parent_valid] if include_observation else []) + [batch.field_valid,
               batch.image_features_2um if include_image else None, batch.target_count_8um, available, protocol]
    if semantic is not None:
        tensors.extend(semantic)
    for tensor in tensors:
        if tensor is None:
            digest.update(b"none")
        else:
            value = tensor.detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())


def update_selection(previous, scores, *, epoch, minimum_improvement, patience_checks, minimum_epochs,
                     tasks=TASKS, fixed_epoch=False, maximum_regression=None):
    """Compare only the configured selection tasks with one accepted checkpoint."""
    if tuple(tasks) not in (TASKS, ("HD16",)) or set(scores) != set(tasks) or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("selection requires finite scores for exactly its configured tasks")
    if previous is not None and epoch <= previous["last_validation_epoch"]:
        raise ValueError("validation epochs must strictly advance")
    if maximum_regression is not None:
        if fixed_epoch or tuple(tasks) != TASKS:
            raise ValueError("bounded selection requires both tasks and a selected checkpoint")
        return bounded_selection(previous, scores, epoch=epoch, minimum_improvement=minimum_improvement,
            maximum_regression=maximum_regression, patience_checks=patience_checks, minimum_epochs=minimum_epochs)
    if fixed_epoch:
        return dict(best_scores=dict(scores), best_epoch=epoch, last_validation_epoch=epoch,
                    stale_checks=0, accepted=True, should_stop=False, rule="fixed_epoch")
    best = None if previous is None else previous["best_scores"]
    if best is not None and set(best) != set(tasks):
        raise ValueError("resume cannot change the checkpoint selection tasks")
    accepted = best is None or (all(scores[t] <= best[t] for t in tasks)
        and any(best[t] - scores[t] > minimum_improvement for t in tasks))
    stale = 0 if accepted else previous["stale_checks"] + 1
    return dict(previous or {}, best_scores=dict(scores if accepted else best),
        best_epoch=epoch if accepted else previous["best_epoch"], last_validation_epoch=epoch,
        stale_checks=stale, accepted=accepted,
        should_stop=epoch >= minimum_epochs and stale >= patience_checks)


def summarize_diagnostics(records):
    """Average each defined FOV metric; undefined correlations remain null."""
    if not records:
        return {}
    if any(set(record) != set(records[0]) for record in records):
        raise ValueError("diagnostic records have different metric definitions")
    result = {}
    for name in records[0]:
        values = [record[name] for record in records if record[name] is not None]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("diagnostic metrics must use null for undefined values")
        result[name] = dict(mean=float(np.mean(values)) if values else None,
                            defined_fields=len(values), total_fields=len(records))
    return result


@contextmanager
def batch_packets(fields, features, config, items, *, batch_size, role, tasks=None, task=None,
                  timings=None, masks=None):
    """Prepare one next batch; device mode overlaps materialization with model work."""
    settings = config["training"]
    pipeline = settings.get("batch_pipeline", "cpu")
    if masks is None and pipeline == "device" and fields.device.type == "cuda":
        layouts = tasks if tasks is not None else [task] * len(items)
        masks = (fields.validation_masks(items, task) if role == "validation" else
                 training_parent_masks(items, layouts, fields, config,
                     cuda_batch_size=settings.get("cuda_mask_batch_size", 96)))
    if masks is not None and len(masks) != len(items):
        raise ValueError("precomputed masks must retain the complete FOV order")

    def prepare(begin):
        selected = items[begin:begin + batch_size]
        layouts = tasks[begin:begin + batch_size] if tasks is not None else [task] * len(selected)
        selected_masks = (masks[begin:begin + batch_size] if masks is not None else
            fields.validation_masks(selected, task, device="cpu") if role == "validation" else
            training_parent_masks(selected, layouts, fields, config, device="cpu"))
        return selected, selected_masks, fields.prepare(selected, role), features.prepare(selected, role)

    def encode(prepared):
        selected, masks, counts, semantic = prepared
        batch, available, protocol = fields.materialize(counts, selected, masks, role)
        return selected, batch, available, protocol, features.encode(semantic, selected, role)

    with prefetched_batches(range(0, len(items), batch_size), prepare, encode,
                            pipeline=pipeline, device=fields.device) as packets:
        def consume():
            for packet in packets:
                if timings is not None:
                    for key in ("cpu_prepare_seconds", "main_wait_seconds", "materialize_host_seconds"):
                        timings[key] += getattr(packet, key)
                yield packet.value
                del packet
        iterator = consume()
        try:
            yield iterator
        finally:
            iterator.close()


def score_items(model, fields, features, config, items, *, task, role, batch_size):
    scores, area_scores, target_umis, diagnostics = [], [], [], []
    contributions = {region: [] for region in ("covered", "gap", "mixed")}
    area_contributions = {region: [] for region in contributions}
    diagnose = config.get("diagnostics", {}).get("enabled", False) and task == "HD16"
    gene_groups = None
    if diagnose and config["diagnostics"].get("gene_groups_from_panel", False):
        groups = read_json(artifact_path(config["panel_artifact"]))["evaluation_gene_groups"]
        output_ids = model.panels.output_gene_ids
        gene_groups = {}
        for name, ids in groups.items():
            ids = set(ids)
            if ids - set(output_ids):
                raise ValueError("diagnostic group includes genes outside the frozen output panel")
            gene_groups[name] = torch.tensor([g in ids for g in output_ids], dtype=torch.bool, device=fields.device)
    with batch_packets(fields, features, config, items, batch_size=batch_size, role=role, task=task) as packets:
        for selected, batch, available, protocol, semantic in packets:
            supervision = core_supervision(batch_size=len(selected), device=fields.device,
                                            core_um=config["supervision_core_um"])
            prediction, losses, area = evaluate_batch(model, batch, available, protocol,
                                                       semantic, supervision)
            scores.extend(losses["loss_per_patch"].detach().cpu().tolist())
            area_scores.extend(area["loss_per_patch"].detach().cpu().tolist())
            target_umis.extend(losses["target_umis"].detach().cpu().tolist())
            for region, values in losses["region_contributions"].items():
                contributions[region].extend(values.detach().cpu().tolist())
            if config.get("diagnostics", {}).get("enabled", False):
                for region, values in area["region_contributions"].items():
                    area_contributions[region].extend(values.detach().cpu().tolist())
            if diagnose:
                diagnostics.extend(diagnostic_metrics(prediction, batch, available, supervision,
                    output_gene_indices=model.output_gene_indices, gene_groups=gene_groups,
                    parent_count_strata=config["diagnostics"].get("parent_count_strata", False),
                    allocation_decomposition=config["diagnostics"].get("allocation_decomposition", False),
                    gene_group_allocation_decomposition=config["diagnostics"].get(
                        "gene_group_allocation_decomposition", False)))
            del prediction, losses, area, batch
    row = dict(fields=len(items), nll_per_umi=float(np.mean(scores)),
        area_nll_per_umi=float(np.mean(area_scores)), mean_target_umis=float(np.mean(target_umis)),
        region_nll_contributions_per_fov_umi={name: float(np.mean(values))
                                             for name, values in contributions.items()})
    if config.get("diagnostics", {}).get("enabled", False):
        row["nll_gain_over_area"] = row["area_nll_per_umi"] - row["nll_per_umi"]
        row["region_nll_gain_over_area_per_fov_umi"] = {
            name: float(np.mean(area_contributions[name]) - np.mean(contributions[name]))
            for name in contributions}
    if diagnose:
        if len(diagnostics) != len(items):
            raise ValueError("HD16 diagnostics must return one record per FOV")
        row["diagnostics"] = summarize_diagnostics(diagnostics)
        if config["diagnostics"].get("allocation_decomposition", False):
            split = row["diagnostics"]["full_gain"]
            if (split["defined_fields"] != len(items)
                    or abs(split["mean"] - row["nll_gain_over_area"]) > 1e-7):
                raise ValueError("allocation decomposition does not reproduce the original full-support NLL gain")
    return row


def aggregate_sections(sections):
    result = dict(nll_per_umi=float(np.mean([r["nll_per_umi"] for r in sections.values()])), sections=sections)
    if all("nll_gain_over_area" in row for row in sections.values()):
        result["nll_gain_over_area"] = float(np.mean([r["nll_gain_over_area"] for r in sections.values()]))
    if all("diagnostics" in row for row in sections.values()):
        result["diagnostics"] = {}
        for name in next(iter(sections.values()))["diagnostics"]:
            rows = [r["diagnostics"][name] for r in sections.values()]
            means = [r["mean"] for r in rows if r["mean"] is not None]
            result["diagnostics"][name] = dict(mean=float(np.mean(means)) if means else None,
                defined_sections=len(means), total_sections=len(sections),
                defined_fields=sum(r["defined_fields"] for r in rows))
    return result


def selection_scores(validation_report, tasks=TASKS):
    """Extract the predeclared validation NLLs for checkpoint selection."""
    return {task: float(validation_report[task]["nll_per_umi"]) for task in tasks}


def validation(model, fields, features, config, *, smoke=False, full_batch_smoke=False, tasks=None):
    if config.get("final_refit"):
        return spatial_monitor(model, fields, features, config, smoke=smoke,
            full_batch_smoke=full_batch_smoke, tasks=tasks or config["validation_tasks"])["tasks"]
    batch_size = (config["smoke"]["batch_size"] if smoke and not full_batch_smoke
                  else config["training"]["batch_size"])
    count = (batch_size if full_batch_smoke else config["smoke"]["validation_fields_per_section"] if smoke
             else config["training"]["validation_fields_per_section"])
    output = {}
    for task in (config.get("validation_tasks", TASKS) if tasks is None else tasks):
        sections = {}
        task_count = min(count, config["training"]["random_validation_fields_per_section"]) if task == "random" else count
        for rid in authorized_ids(config, "validation"):
            if fields.stores[rid].record["fields"] != config["training"]["validation_fields_per_section"]:
                raise ValueError("validation must use the same complete frozen 100-FOV section pools")
            protocol = fields.stores[rid].record["protocol_id"]
            items = [dict(sample=rid, protocol_id=protocol, field_index=i) for i in range(task_count)]
            sections[rid] = score_items(model, fields, features, config, items, task=task,
                                        role="validation", batch_size=batch_size)
        output[task] = aggregate_sections(sections)
        output[task]["protocols"] = {name: aggregate_sections({rid: row for rid, row in sections.items() if rid in ids})
            for name, ids in (("WT", config["wt"]["validation"]), ("3prime", config["three_prime"].get("validation", []))) if ids}
    return output


def spatial_monitor(model, fields, features, config, *, smoke=False, full_batch_smoke=False, tasks=None):
    items = monitor_items(fields, config, smoke=smoke or full_batch_smoke)
    batch_size = (config["smoke"]["batch_size"] if smoke and not full_batch_smoke
                  else config["training"]["batch_size"])
    result = dict(used_for_checkpoint_selection=config.get("final_refit", {}).get("selection") == "development_spatial_holdout",
        optimizer_excluded=True,
        engineering_subset=smoke or full_batch_smoke, fields=len(items), items=items, tasks={})
    for task in (TASKS if tasks is None else tasks):
        sections = {}
        for rid in dict.fromkeys(item["sample"] for item in items):
            selected = [item for item in items if item["sample"] == rid]
            sections[rid] = score_items(model, fields, features, config, selected, task=task,
                                        role="train", batch_size=batch_size)
        result["tasks"][task] = aggregate_sections(sections)
        protocols = {}
        for name, ids in (("WT", config["wt"]["train"]),
                          ("3prime", config["three_prime"]["auxiliary_train"])):
            selected = {rid: row for rid, row in sections.items() if rid in ids}
            if selected:
                protocols[name] = aggregate_sections(selected)
        result["tasks"][task]["protocols"] = protocols
    return result


def capture_rng(device):
    numpy = np.random.get_state()
    state = dict(torch=torch.get_rng_state(), python=random.getstate(),
        numpy=dict(name=numpy[0], keys=numpy[1].tolist(), position=numpy[2],
                   has_gauss=numpy[3], cached_gaussian=numpy[4]))
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy["name"], np.asarray(numpy["keys"], dtype=np.uint32),
                        numpy["position"], numpy["has_gauss"], numpy["cached_gaussian"]))
    if device.type == "cuda":
        if "cuda" not in state:
            raise ValueError("CUDA resume requires the original CUDA RNG state")
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def atomic_checkpoint(path, model, optimizer, state):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_checkpoint(temporary, model, optimizer=optimizer, step=state["optimizer_updates"], training_state=state)
    os.replace(temporary, path)


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_epoch_cache(settings, device):
    """Retain v30's bounded allocator and thresholded epoch-end cache release."""
    if torch.device(device).type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_reserved(device)
    triggered = before > settings.get("empty_cache_reserved_threshold_gib", 48) * 2**30
    if triggered:
        torch.cuda.empty_cache()
    return dict(epoch_end_allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
        empty_cache_triggered=triggered, reserved_before_release_gib=before / 2**30,
        reserved_after_release_gib=torch.cuda.memory_reserved(device) / 2**30,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        peak_cuda_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)


def warmup_clock(settings):
    clock = settings.get("warmup_clock", "optimizer_updates")
    if clock not in ("reference_micro_batches", "optimizer_updates"):
        raise ValueError("warmup_clock must be reference_micro_batches or optimizer_updates")
    return clock


def restore_warmup_counter(state, settings):
    """Legacy update-clock checkpoints may lack a historical micro-batch count."""
    previous = state.get("warmup_clock", warmup_clock(state["config"]["training"]))
    if previous != warmup_clock(settings):
        raise ValueError("resume cannot change the warmup clock")
    count = state.get("reference_micro_batches")
    if count is None:
        if previous == "reference_micro_batches":
            raise ValueError("micro-batch warmup resume requires its recorded counter")
        return None
    if type(count) is not int or count < 0:
        raise ValueError("reference_micro_batches must be a nonnegative integer")
    return count


def advance_reference_counter(count, micro_batches):
    if type(micro_batches) is not int or micro_batches <= 0:
        raise ValueError("each optimizer update must consume a positive number of micro-batches")
    return None if count is None else count + micro_batches


def set_learning_rate(optimizer, settings, updates, plateau_scale, *, reference_micro_batches=None):
    warmup = settings["warmup_optimizer_steps"]
    counter = reference_micro_batches if warmup_clock(settings) == "reference_micro_batches" else updates
    if type(counter) is not int or counter < 0:
        raise ValueError("the selected warmup clock requires a nonnegative counter")
    ratio = min(1., (counter + 1) / warmup) if warmup else 1.
    if settings.get("learning_rate_schedule") == "warmup_cosine_over_actual_optimizer_steps":
        if warmup_clock(settings) != "optimizer_updates" or plateau_scale != 1.:
            raise ValueError("legacy cosine requires the optimizer-update clock without plateau scaling")
        ratio = cosine_multiplier(updates, settings)
    for group in optimizer.param_groups:
        group["lr"] = settings["learning_rate"] * ratio * plateau_scale


def full_batch_smoke_items(items, config):
    """Use one complete optimizer update with both assays and all three layouts.

    This deterministic engineering subset is not a scientific sampling policy.
    It changes neither the formal epoch order nor the configured exposure quotas.
    """
    settings = config["training"]
    if settings.get("fov_parent_views") is not None:
        return full_batch_view_items(items, config)
    batch_size, accumulation = settings["batch_size"], settings["gradient_accumulation_steps"]
    size = batch_size * accumulation
    if batch_size < 2 or size < 6 or size > 64:
        raise ValueError("full-batch smoke needs a mixed-assay batch and 6..64 total FOVs")
    pools = {protocol: [] for protocol in (0, 1)}
    allowed = set(authorized_ids(config, "train"))
    for item in items:
        if item["sample"] not in allowed or item["protocol_id"] not in pools:
            raise PermissionError("full-batch smoke accepts only the authorized training epoch")
        expected = 0 if item["sample"] in config["three_prime"]["auxiliary_train"] else 1
        if item["protocol_id"] != expected:
            raise ValueError("smoke assay ID disagrees with its training section")
        pools[expected].append(item)
    selected = []
    for index in range(size):
        protocol, position = (1 if index % 2 == 0 else 0), index // 2
        if position >= len(pools[protocol]):
            raise ValueError("training epoch lacks sufficient WT/3prime fields for full-batch smoke")
        selected.append(dict(pools[protocol][position]))
    if len({(i["sample"], i["field_index"]) for i in selected}) != size:
        raise ValueError("full-batch smoke fields must be unique within the optimizer update")
    layouts = ("random", "HD16", "Spot55")
    tasks = [layouts[index % len(layouts)] for index in range(size)]
    report = dict(smoke_layout_override=True, engineering_subset=True,
        fields={task: tasks.count(task) for task in layouts},
        assay_fields={"WT": sum(i["protocol_id"] == 1 for i in selected),
                      "3prime": sum(i["protocol_id"] == 0 for i in selected)},
        protocol_layout_fields={str(protocol): {task: sum(i["protocol_id"] == protocol and t == task
            for i, t in zip(selected, tasks, strict=True)) for task in layouts} for protocol in (0, 1)})
    return selected, tasks, report


def resume_config_matches(previous, current):
    """Require the same execution settings apart from an increased epoch budget."""
    previous, current = deepcopy(previous), deepcopy(current)
    for config in (previous, current):
        config.setdefault("model_family", "v033")
        if config["model_family"] == "ASTRA":
            config["model_family"] = "v030"
        # Descriptive labels may change; execution flags and panel settings remain checked.
        for key in ("initialization", "panel", "formal_training"):
            config.get("boundaries", {}).pop(key, None)
        config["training"].setdefault("warmup_clock", "optimizer_updates")
        if "model" in config and config["model_family"] == "v033":
            config["model"].setdefault("allocation_post_center_norm", "layernorm")
    before = previous["training"].pop("max_epochs")
    after = current["training"].pop("max_epochs")
    return after >= before and previous == current


def configuration_version(config_path):
    """Identify the published ASTRA architecture used by the training runner."""
    if not Path(config_path).resolve().is_relative_to(ROOT / "training"):
        raise ValueError("keep task configurations under ASTRA/training")
    return "v030"


def run(config_path, *, output=None, smoke=False, resume=None, device=None, full_batch_smoke=False):
    started = time.monotonic()
    if smoke and full_batch_smoke:
        raise ValueError("--smoke and --full-batch-smoke are distinct budgets; choose one")
    short_run = smoke or full_batch_smoke
    config_path = Path(config_path).resolve()
    version = configuration_version(config_path)
    config = read_json(config_path)
    if version == "v030" and config.get("model_family") not in ("ASTRA", "v030"):
        raise ValueError("training configurations must select the ASTRA model family")
    panels, budgets = validate_config(config)
    settings = config["training"]
    if not short_run and not settings["formal_training"]:
        raise PermissionError("formal_training=false: only configuration checks and bounded smoke runs are enabled")
    if not short_run and config.get("engineering_smoke_dataset", False):
        raise PermissionError("the bounded engineering cache cannot be used for formal training")
    target = torch.device(device or ("cpu" if short_run else "cuda:0"))
    if target.type not in ("cpu", "cuda") or (not short_run and target.type != "cuda"):
        raise ValueError("formal training requires one explicitly authorized GPU; smoke also supports CPU")
    if target.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or target.index not in (None, 0):
            raise ValueError("expose exactly one authorized GPU with CUDA_VISIBLE_DEVICES")
        target = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(settings.get("cuda_memory_fraction", 1.0), target)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(settings["cpu_threads"])
    torch.manual_seed(settings["seed"])
    np.random.seed(settings["seed"] % 2**32)
    random.seed(settings["seed"])
    if target.type == "cuda":
        torch.cuda.manual_seed(settings["seed"])
        torch.cuda.reset_peak_memory_stats(target)
    root = artifact_path(config["output_root"])
    if not root.is_relative_to(ROOT / "outputs"):
        raise ValueError("training outputs must stay under ASTRA/outputs")
    resume = Path(resume).resolve() if resume else None
    output = (Path(output).resolve() if output else resume.parent if resume
              else root / ("full_batch_smoke" if full_batch_smoke else "smoke" if smoke else "formal"))
    if output == root or not output.is_relative_to(root):
        raise ValueError("use a dedicated new run directory under the configured round output")
    if resume is not None:
        if resume != output / "last.pt" or not output.is_dir():
            raise ValueError("resume must use this run directory's last.pt")
    else:
        output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", dict(status="initializing", config=str(config_path.relative_to(ROOT)),
        smoke=short_run, full_batch_smoke=full_batch_smoke, resumed_from=str(resume) if resume else None))
    try:
        return _fit(config_path, config, panels, budgets, output, target, started,
                    smoke=smoke, resume=resume, full_batch_smoke=full_batch_smoke)
    except Exception as error:
        info = read_json(output / "run.json")
        info.update(status="failed", error=dict(type=type(error).__name__, message=str(error)),
                    elapsed_seconds=time.monotonic() - started)
        write_json(output / "run.json", info)
        raise


def _fit(config_path, config, panels, budgets, output, target, started, *, smoke, resume, full_batch_smoke=False):
    settings = config["training"]
    short_run = smoke or full_batch_smoke
    # No held-out/test data are opened by this CLI, including for final replay.
    fields = CachedFields(config, panels, target)
    features = CachedFeatures(config, fields, enabled=config["model"].get("use_pathology", True))
    entries = fields.training_entries()
    if any(budgets[e["section"]] > e["fields"] for e in entries):
        raise ValueError("a training section cannot supply its configured epoch exposure")
    maximum = config["smoke"]["epochs"] if short_run else settings["max_epochs"]
    selection, first_epoch, updates, exposures, plateau_scale = None, 1, 0, 0, 1.
    reference_micro_batches = 0
    if resume:
        model, payload = load_checkpoint(resume, expected_panels=panels, device=target)
        state = payload["training_state"]
        if (not resume_config_matches(state["config"], config) or state["smoke"] != short_run
                or state.get("full_batch_smoke", False) != full_batch_smoke
                or state["uni_sha"] != features.sha or state["device_type"] != target.type):
            raise ValueError("resume changes scientific settings, execution scope, device type or UNI identity")
        selection, first_epoch = state["selection"], state["epoch"] + 1
        updates, exposures, plateau_scale = state["optimizer_updates"], state["field_exposures"], state["plateau_scale"]
        reference_micro_batches = restore_warmup_counter(state, settings)
    else:
        model = model_class(config.get("model_family", "v033"))(**panels.as_dict(), **config["model"]).to(target)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    if resume:
        if payload.get("optimizer") is None:
            raise ValueError("training resume checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(state["rng"], target)
        if selection is not None:
            best_model, best_payload = load_checkpoint(output / "best.pt", expected_panels=panels, device="cpu")
            if (best_payload["training_state"]["epoch"] != selection["best_epoch"]
                    or not resume_config_matches(best_payload["training_state"]["config"], config)):
                raise ValueError("resume run is missing the selected best checkpoint")
            del best_model, best_payload
            # Loading a model consumes initialization RNG; restore after this check.
            restore_rng(state["rng"], target)
        del payload, state
    if first_epoch > maximum or (selection is not None and selection["should_stop"]):
        raise ValueError("run has already met its epoch budget or patience stopping rule")
    batch_size = config["smoke"]["batch_size"] if smoke else settings["batch_size"]
    accumulation = 1 if smoke else settings["gradient_accumulation_steps"]
    info = dict(config=str(config_path.relative_to(ROOT)), smoke=short_run,
        model_family=config.get("model_family", "v033"),
        data_pipeline="device_prefetch_cuda_masks" if settings.get("batch_pipeline", "cpu") == "device"
            and target.type == "cuda" else "cpu_prefetch_pinned",
        cuda_mask_batch_size=settings.get("cuda_mask_batch_size", 96),
        full_batch_smoke=full_batch_smoke, performance_evaluation=not short_run, device=str(target),
        micro_batch_size=batch_size, gradient_accumulation_steps=accumulation,
        configured_effective_batch_size=batch_size * accumulation, warmup_clock=warmup_clock(settings),
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES") if target.type == "cuda" else None,
        torch_version=str(torch.__version__), uni_sha=features.sha, test_expression_read=False,
        parameters=sum(p.numel() for p in model.parameters()), fields_per_epoch=settings["fields_per_epoch"],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        cuda_memory_fraction=settings.get("cuda_memory_fraction", 1.0),
        empty_cache_reserved_threshold_gib=settings.get("empty_cache_reserved_threshold_gib", 48),
        package_version=(ROOT / "VERSION").read_text(encoding="utf-8").strip())
    write_json(output / "run.json", dict(info, status="running", resumed_from=str(resume) if resume else None))
    latest_validation = None
    smoke_inputs = hashlib.sha256() if full_batch_smoke else None
    smoke_nonimage = hashlib.sha256() if full_batch_smoke else None
    smoke_fields = hashlib.sha256() if full_batch_smoke else None
    optimizer_update_fov_counts, optimizer_update_seconds, smoke_items = [], [], None
    for epoch in range(first_epoch, maximum + 1):
        epoch_start, total_loss, optimization_loss = time.monotonic(), 0., 0.
        objective_totals = dict.fromkeys(("weighted_data_loss", "weighted_parent_kl", "mean_fov_weight",
                                         "downweighted_fov_fraction"), 0.)
        items = training_epoch_items(entries=entries, section_budgets=budgets, epoch=epoch, seed=settings["seed"],
            allowed_training_ids=authorized_ids(config, "train"), sampling_mode=settings["sampling_mode"],
            view_policy=settings.get("fov_parent_views"))
        if full_batch_smoke:
            items, tasks, observation_report = full_batch_smoke_items(items, config)
            smoke_items = [dict(item, task=task) for item, task in zip(items, tasks, strict=True)]
        elif smoke:
            items = items[:config["smoke"]["training_fields"]]
            tasks, observation_report = assign_parent_tasks(items, config, epoch, smoke=True)
        else:
            tasks, observation_report = assign_parent_tasks(items, config, epoch)
        pipeline_timings = dict(cpu_prepare_seconds=0., main_wait_seconds=0., materialize_host_seconds=0.)
        log_names = ["loss", "optimization_loss"] + (list(objective_totals)
            if settings.get("training_loss") is not None else [])
        loss_records = []
        train_started = time.monotonic()
        masks = None
        pipeline_timings["mask_generation_seconds"] = 0.
        if settings.get("batch_pipeline", "cpu") == "device" and target.type == "cuda":
            masks = training_parent_masks(items, tasks, fields, config,
                cuda_batch_size=settings.get("cuda_mask_batch_size", 96))
            torch.cuda.synchronize(target)
            pipeline_timings["mask_generation_seconds"] = time.monotonic() - train_started
        optimization_started = time.monotonic()
        # Each accumulation group averages FOVs, including an incomplete final group.
        with batch_packets(fields, features, config, items, batch_size=batch_size, role="train",
                           tasks=tasks, timings=pipeline_timings, masks=masks) as packets:
            for group_begin in range(0, len(items), batch_size * accumulation):
                update_start = time.monotonic()
                group_end = min(group_begin + batch_size * accumulation, len(items))
                set_learning_rate(optimizer, settings, updates, plateau_scale,
                                  reference_micro_batches=reference_micro_batches)
                group_micro_batches = 0
                for begin in range(group_begin, group_end, batch_size):
                    end = min(begin + batch_size, group_end)
                    selected, batch, available, protocol, semantic = next(packets)
                    if smoke_inputs is not None:
                        observe_smoke_inputs(smoke_inputs, batch, available, protocol, semantic)
                        observe_smoke_inputs(smoke_nonimage, batch, available, protocol, semantic, include_image=False)
                        observe_smoke_inputs(smoke_fields, batch, available, protocol, semantic, include_observation=False)
                    prediction, losses = train_step(model, optimizer, batch, available, protocol,
                        semantic, core_supervision(batch_size=len(selected),
                            device=target, core_um=config["supervision_core_um"]),
                        max_gradient_norm=settings["max_gradient_norm"], loss_weight=len(selected) / (group_end - group_begin),
                        zero_grad=begin == group_begin, optimizer_step=end == group_end,
                        region_weights=settings["region_weights"], loss_config=settings.get("training_loss"))
                    # Defer scalar downloads; retain the original Python-float summation order.
                    loss_records.append((len(selected), torch.stack([losses[name].detach() for name in log_names])))
                    exposures += len(selected)
                    group_micro_batches += 1
                    del prediction, losses, batch, semantic
                updates += 1
                reference_micro_batches = advance_reference_counter(reference_micro_batches, group_micro_batches)
                if short_run:
                    if target.type == "cuda":
                        torch.cuda.synchronize(target)
                    optimizer_update_fov_counts.append(group_end - group_begin)
                    optimizer_update_seconds.append(time.monotonic() - update_start)
        values = torch.stack([record[1] for record in loss_records]).cpu().tolist()
        totals = {name: sum(record[0] * row[index] for record, row in zip(loss_records, values, strict=True))
                  for index, name in enumerate(log_names)}
        total_loss, optimization_loss = totals.pop("loss"), totals.pop("optimization_loss")
        objective_totals.update(totals)
        pipeline_timings["optimization_seconds"] = time.monotonic() - optimization_started
        pipeline_timings["training_seconds"] = time.monotonic() - train_started
        del loss_records, values, masks
        row = dict(epoch=epoch, fields=len(items), loss=total_loss / len(items), optimizer_updates=updates,
            optimization_loss=optimization_loss / len(items), field_exposures=exposures,
            parent_sampling=observation_report, learning_rate=optimizer.param_groups[0]["lr"],
            reference_micro_batches=reference_micro_batches, pipeline_timings=pipeline_timings)
        if settings.get("training_loss") is not None:
            row["training_objective"] = {key: value / len(items) for key, value in objective_totals.items()}
        if short_run or epoch % settings["validation_interval"] == 0 or epoch == maximum:
            latest_validation = validation(model, fields, features, config,
                                            smoke=smoke, full_batch_smoke=full_batch_smoke)
            selected_tasks = tuple(config["selection"].get("tasks", TASKS))
            selection = update_selection(selection, selection_scores(latest_validation, selected_tasks),
                epoch=epoch, minimum_improvement=config["selection"]["minimum_improvement"],
                patience_checks=settings["patience_checks"], minimum_epochs=settings["minimum_epochs"], tasks=selected_tasks,
                fixed_epoch=config["selection"]["rule"] == "fixed_epoch",
                maximum_regression=(config["selection"]["maximum_regression"]
                    if config["selection"]["rule"] == "bounded_regression_fixed_initial_anchor" else None))
            if selection["accepted"]:
                selection["checkpoint_validation_scores"] = selection_scores(latest_validation, tuple(latest_validation))
            if (settings.get("learning_rate_schedule", "validation_plateau") == "validation_plateau"
                    and not selection["accepted"] and selection["stale_checks"] % settings["plateau_patience_checks"] == 0):
                plateau_scale = max(settings["minimum_learning_rate_ratio"], plateau_scale * settings["plateau_factor"])
            row.update(validation=latest_validation, selection=selection)
        diagnostics = config.get("diagnostics", {})
        if not config.get("final_refit") and diagnostics.get("enabled", False) and (short_run or epoch % diagnostics.get("interval_epochs", 10) == 0):
            row["spatial_monitor"] = spatial_monitor(model, fields, features, config,
                smoke=smoke, full_batch_smoke=full_batch_smoke)
        state = dict(config=config, epoch=epoch, optimizer_updates=updates, field_exposures=exposures,
            selection=selection, plateau_scale=plateau_scale, rng=capture_rng(target), uni_sha=features.sha,
            smoke=short_run, full_batch_smoke=full_batch_smoke, device_type=target.type,
            warmup_clock=warmup_clock(settings), reference_micro_batches=reference_micro_batches)
        if "validation" in row and selection["accepted"]:
            atomic_checkpoint(output / "best.pt", model, optimizer, state)
        atomic_checkpoint(output / "last.pt", model, optimizer, state)
        if not short_run and epoch in settings.get("milestone_epochs", []):
            save_checkpoint(output / f"epoch{epoch:03d}.pt", model, optimizer=optimizer,
                            step=updates, training_state=state)
        row.update(release_epoch_cache(settings, target))
        row["seconds"] = time.monotonic() - epoch_start
        with (output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        if selection is not None and selection["should_stop"]:
            break
    if selection is None:
        raise RuntimeError("completed run has no evaluated checkpoint")
    del optimizer, model
    if target.type == "cuda":
        torch.cuda.empty_cache()
    best, payload = load_checkpoint(output / "best.pt", expected_panels=panels, device=target)
    if payload["training_state"]["epoch"] != selection["best_epoch"]:
        raise ValueError("selected checkpoint epoch differs after reload")
    replay = validation(best, fields, features, config, smoke=smoke, full_batch_smoke=full_batch_smoke)
    expected_scores = selection.get("checkpoint_validation_scores", selection["best_scores"])
    difference = max(abs(replay[t]["nll_per_umi"] - value) for t, value in expected_scores.items())
    if difference > 1e-6:
        raise ValueError(f"selected checkpoint validation replay changed by {difference}")
    stopped = selection["should_stop"]
    report = dict(info, status="complete", best_epoch=selection["best_epoch"], total_epochs_run=epoch,
        max_epochs=maximum, stale_checks=selection["stale_checks"], patience_checks=settings["patience_checks"],
        stop_reason="full_batch_smoke_complete" if full_batch_smoke else "smoke_complete" if smoke else "patience" if stopped else "budget-censored",
        budget_censored=not short_run and epoch >= maximum and not stopped,
        optimizer_updates=updates, field_exposures=exposures, reference_micro_batches=reference_micro_batches,
        milestone_checkpoints={str(mark): str((output / f"epoch{mark:03d}.pt").relative_to(ROOT))
            for mark in settings.get("milestone_epochs", []) if (output / f"epoch{mark:03d}.pt").exists()},
        selected_checkpoint=str((output / "best.pt").relative_to(ROOT)),
        checkpoint_sha256=checkpoint_digest(output / "best.pt"), checkpoint_reload_max_score_difference=difference,
        validation=replay, elapsed_seconds=time.monotonic() - started,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(target) / 2**30 if target.type == "cuda" else None,
        peak_cuda_reserved_gib=torch.cuda.max_memory_reserved(target) / 2**30 if target.type == "cuda" else None)
    if short_run:
        report.update(optimizer_update_fov_counts=optimizer_update_fov_counts,
            optimizer_update_seconds=optimizer_update_seconds,
            actual_effective_batch_size=max(optimizer_update_fov_counts))
    if full_batch_smoke:
        expected = settings["batch_size"] * settings["gradient_accumulation_steps"]
        if optimizer_update_fov_counts != [expected]:
            raise AssertionError("full-batch smoke must finish exactly one complete optimizer update")
        report["smoke_training_items"] = smoke_items
        report["smoke_observation_sha256"] = smoke_inputs.hexdigest()
        report['smoke_nonimage_observation_sha256'] = smoke_nonimage.hexdigest()
        report['smoke_field_observation_sha256'] = smoke_fields.hexdigest()
    if config.get("final_refit"):
        report.update(final_refit=config["final_refit"], checkpoint_selection_rule=config["selection"]["rule"],
                      validation_is_development_spatial_monitor=True)
    elif config.get("diagnostics", {}).get("enabled", False):
        report["spatial_monitor"] = spatial_monitor(best, fields, features, config,
            smoke=smoke, full_batch_smoke=full_batch_smoke)
    if config.get("final_evaluation_tasks"):
        report["final_evaluation"] = dict(used_for_checkpoint_selection=False,
            checkpoint=report["selected_checkpoint"],
            tasks=validation(best, fields, features, config, smoke=smoke,
                full_batch_smoke=full_batch_smoke, tasks=config["final_evaluation_tasks"]))
    report["elapsed_seconds"] = time.monotonic() - started
    if target.type == "cuda":
        report.update(peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(target) / 2**30,
                      peak_cuda_reserved_gib=torch.cuda.max_memory_reserved(target) / 2**30)
    write_json(output / "report.json", report)
    write_json(output / "run.json", dict(info, status="complete", report="report.json"))
    print(json.dumps(report, allow_nan=False), flush=True)
    return report


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--workspace", type=Path, help="explicit user task created by configure-training")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="resume this run's last.pt with optimizer and RNG state")
    parser.add_argument("--device", help="cpu for smoke, or cuda:0 with one authorized GPU exposed")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true", help="run the configured short real-cache smoke budget")
    modes.add_argument("--full-batch-smoke", action="store_true",
        help="one complete formal-sized accumulated update with both assays and all three parent layouts")
    parser.add_argument("--check-config", action="store_true", help="validate identities and budgets without training")
    args = parser.parse_args(argv)
    if args.workspace:
        from astra.training.utils import use_workspace
        use_workspace(args.workspace)
    args.config = args.config or ROOT / "training/config.json"
    if args.check_config:
        panels, budgets = validate_config(read_json(args.config))
        print(json.dumps(dict(status="valid", input_genes=len(panels.input_gene_ids),
            output_genes=len(panels.output_gene_ids), section_budgets=budgets,
            training_started=False, test_expression_read=False), allow_nan=False))
    else:
        run(args.config, output=args.output, smoke=args.smoke, resume=args.resume, device=args.device,
            full_batch_smoke=args.full_batch_smoke)


if __name__ == "__main__":
    main()
