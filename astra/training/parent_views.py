"""Fixed-budget FOV exposure with one, two or four phases of 16um parents."""
from collections import Counter, defaultdict
import hashlib
import json

import numpy as np

from astra.training.schedule import epoch_items
from astra.training.utils import stable_seed


PHASES = ((0, 0), (0, 8), (8, 0), (8, 8))
MODES = ("canonical_repeat", "fixed_phase", "multi_phase")


def validate_view_policy(settings):
    policy = settings.get("fov_parent_views")
    if policy is None:
        return None
    if (not isinstance(policy, dict) or set(policy) != {"mode", "repeats", "phases_um"}
            or policy["mode"] not in MODES or type(policy["repeats"]) is not int
            or policy["repeats"] not in (1, 2, 4)
            or (policy["mode"] != "multi_phase" and policy["repeats"] != 4)
            or policy["phases_um"] != [list(p) for p in PHASES]
            or settings["parent_probabilities"] != {"random": 0., "HD16": 1., "Spot55": 0.}
            or settings["sampling_mode"] != "shuffled_cycles"):
        raise ValueError("FOV views require 1/2/4 presentations, balanced HD16 phases and HD16-only shuffled cycles")
    quotas = [settings["fields_per_section"], *settings["section_field_overrides"].values()]
    if any(type(q) is not int or q <= 0 or q % (4 * policy["repeats"]) for q in quotas):
        raise ValueError("per-section exposures must be divisible by four times the presentation count")
    return policy


def selected_phase(policy, phase, view, pair_xor=None):
    if policy["mode"] == "canonical_repeat":
        return 0
    if policy["mode"] == "fixed_phase":
        return phase
    if policy["repeats"] == 2:
        if pair_xor not in (1, 2, 3):
            raise ValueError("dual views require a nonzero two-bit phase offset")
        return phase if view == 0 else phase ^ pair_xor
    return (phase + view) % 4


def training_epoch_items(*, entries, section_budgets, epoch, seed, allowed_training_ids,
                         sampling_mode="fresh", view_policy=None):
    if view_policy is None:
        return epoch_items(entries=entries, section_budgets=section_budgets, epoch=epoch,
            seed=seed, allowed_training_ids=allowed_training_ids, sampling_mode=sampling_mode)
    if (type(epoch) is not int or epoch < 1 or sampling_mode != "shuffled_cycles"
            or view_policy["mode"] not in MODES or len(entries) != len(allowed_training_ids)
            or {e["section"] for e in entries} != set(allowed_training_ids)
            or set(section_budgets) != set(allowed_training_ids)):
        raise ValueError("view sampling requires the closed training pool and a positive epoch")
    repeats = view_policy["repeats"]
    if type(repeats) is not int or repeats not in (1, 2, 4):
        raise ValueError("unsupported number of FOV presentations")
    items = []
    for entry in entries:
        rid, quota = entry["section"], section_budgets[entry["section"]]
        if (entry["role"] != "train" or entry["protocol_id"] not in (0, 1)
                or type(quota) is not int or quota <= 0 or quota % (4 * repeats)):
            raise ValueError("invalid training section or balanced view budget")
        indices = np.asarray(entry.get("field_indices", np.arange(entry["fields"])))
        if len(indices) != entry["fields"] or len(np.unique(indices)) != len(indices):
            raise ValueError("eligible FOV identities differ")
        # Assign one immutable phase stratum to each FOV; never use expression to partition.
        order = np.random.default_rng(stable_seed(seed, rid, "v033_phase_partition")).permutation(indices)
        per_phase = quota // (4 * repeats)
        for phase in range(4):
            pool = order[phase::4]
            if len(pool) < per_phase:
                raise ValueError("phase stratum cannot supply this epoch's unique FOV budget")
            cached_cycle, permutation = None, None
            for draw in range((epoch - 1) * per_phase, epoch * per_phase):
                cycle, position = divmod(draw, len(pool))
                if cycle != cached_cycle:
                    permutation = np.random.default_rng(stable_seed(
                        seed, rid, phase, cycle, "v033_phase_cycle")).permutation(pool)
                    cached_cycle = cycle
                index = int(permutation[position])
                for view in range(repeats):
                    phase_index = selected_phase(view_policy, phase, view, 1 + draw % 3)
                    items.append(dict(sample=rid, field_index=index, protocol_id=entry["protocol_id"],
                        fixed_phase_index=phase, view_index=view, hd16_phase_um=list(PHASES[phase_index]),
                        sampling_cycle=cycle, mask_seed=stable_seed(seed, rid, index, cycle, "random_owner"),
                        coverage_stratum=0))
                    if repeats == 2:
                        # XOR 1/2/3 rotates horizontal, vertical and diagonal phase pairs.
                        items[-1]["phase_pair_xor"] = 1 + draw % 3
    order = np.random.default_rng(stable_seed(seed, epoch, "v033_view_order")).permutation(len(items))
    return [items[int(i)] for i in order]


def view_sampling_report(items, policy, *, smoke=False):
    sections = {}
    for rid in dict.fromkeys(i["sample"] for i in items):
        selected = [i for i in items if i["sample"] == rid]
        phases = Counter(tuple(i["hd16_phase_um"]) for i in selected)
        if not smoke:
            grouped = defaultdict(list)
            for item in selected:
                grouped[item["field_index"]].append(item)
                phase = selected_phase(policy, item["fixed_phase_index"], item["view_index"], item.get("phase_pair_xor"))
                if tuple(item["hd16_phase_um"]) != PHASES[phase]:
                    raise ValueError("FOV phase assignment differs from its configured condition")
            if any(sorted(i["view_index"] for i in rows) != list(range(policy["repeats"])) for rows in grouped.values()):
                raise ValueError("each FOV must appear in exactly its configured number of presentations per epoch")
            if policy["mode"] != "canonical_repeat" and any(phases[p] != len(selected) // 4 for p in PHASES):
                raise ValueError("phase exposures are not balanced within the training section")
        sections[rid] = dict(fields=len(selected), unique_fovs=len({i["field_index"] for i in selected}),
            phase_fields={str(list(p)): phases[p] for p in PHASES})
    identities = [[i[k] for k in ("sample", "field_index", "protocol_id", "view_index")] for i in items]
    return dict(fields={"random": 0, "HD16": len(items), "Spot55": 0},
        fov_parent_views=policy, sections=sections, smoke_layout_override=False, engineering_subset=smoke,
        field_order_sha256=hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest())


def full_batch_view_items(items, config):
    settings = config["training"]
    repeats = settings["fov_parent_views"]["repeats"]
    fovs_per_phase = 4 // repeats
    if settings["batch_size"] * settings["gradient_accumulation_steps"] != 16:
        raise ValueError("view smoke requires one 16-presentation optimizer update")
    grouped = defaultdict(list)
    for item in items:
        grouped[(item["sample"], item["field_index"])].append(item)
    chosen = []
    for phase in range(4):
        protocol = 1 if phase % 2 == 0 else 0
        candidates = [rows for rows in grouped.values() if rows[0]["fixed_phase_index"] == phase
                      and rows[0]["protocol_id"] == protocol]
        if len(candidates) < fovs_per_phase:
            raise ValueError("view smoke requires all four fixed-phase strata and both assays")
        group = []
        for candidate in candidates[:fovs_per_phase]:
            rows = sorted(candidate, key=lambda i: i["view_index"])
            if [i["view_index"] for i in rows] != list(range(repeats)):
                raise ValueError("view smoke requires all configured presentations of each selected FOV")
            group.append(rows)
        chosen.append(group)
    interleaved = [chosen[phase][position] for position in range(fovs_per_phase) for phase in range(4)]
    selected = [rows[view] for view in range(repeats) for rows in interleaved]
    return selected, ["HD16"] * 16, view_sampling_report(selected, settings["fov_parent_views"], smoke=True)
