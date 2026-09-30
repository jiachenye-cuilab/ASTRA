"""Fresh per-section exposure, globally shuffled without protocol quotas."""

import math
import numpy as np

from astra.training.utils import stable_seed


def full_coverage_plan(entries, section_budgets, *, validation_interval):
    """First scheduled validation after every training FOV has been presented once."""
    if type(validation_interval) is not int or validation_interval <= 0:
        raise ValueError("validation interval must be a positive integer")
    if len(entries) != len(section_budgets) or {e["section"] for e in entries} != set(section_budgets):
        raise ValueError("coverage requires exactly the configured training sections")
    sections = {}
    for entry in entries:
        fields, quota = entry["fields"], section_budgets[entry["section"]]
        if entry["role"] != "train" or any(type(v) is not int or v <= 0 for v in (fields, quota)) or quota > fields:
            raise ValueError("coverage requires nonempty training pools and valid per-section quotas")
        sections[entry["section"]] = dict(fields=fields, fovs_per_epoch=quota,
            first_complete_epoch=math.ceil(fields / quota))
    first = max(row["first_complete_epoch"] for row in sections.values())
    return dict(minimum_epochs=math.ceil(first / validation_interval) * validation_interval,
        total_unique_fovs=sum(row["fields"] for row in sections.values()), sections=sections)


def epoch_items(*, entries, section_budgets, epoch, seed, allowed_training_ids, sampling_mode="fresh"):
    """Budgets are per-section allocations after any related-group apportionment.

    No total-FOV cap redistributes or dilutes these budgets. The caller must provide
    the closed config training IDs, not infer authorization from entry metadata.
    """
    if epoch < 1 or type(epoch) is not int:
        raise ValueError("epoch must be a positive integer")
    if sampling_mode not in ("fresh", "shuffled_cycles", "fresh_then_shuffled_cycles"):
        raise ValueError("unknown training sampling mode")
    if len(entries) != len({entry["section"] for entry in entries}):
        raise ValueError("duplicate training section")
    if set(section_budgets) != set(allowed_training_ids) or set(section_budgets) != {e["section"] for e in entries}:
        raise ValueError("budgets and fields must match exactly the authorized training IDs")
    items = []
    for entry in entries:
        section = entry["section"]
        count = section_budgets[section]
        if entry["role"] != "train" or type(count) is not int or count <= 0:
            raise ValueError("each training section requires an explicit positive exposure budget")
        if entry["protocol_id"] not in (0, 1):
            raise ValueError("unknown protocol")
        if type(entry["fields"]) is not int or count > entry["fields"]:
            raise ValueError("section pool must cover one epoch's quota")
        if sampling_mode == "fresh" and epoch * count > entry["fields"]:
            raise ValueError(f"fresh FOV pool exhausted: {section}; no silent recycling")
        previous_cycle, permutation = None, None
        for draw in range((epoch - 1) * count, epoch * count):
            index = draw
            cycle_draw = (sampling_mode == "shuffled_cycles" or
                          sampling_mode == "fresh_then_shuffled_cycles" and draw >= entry["fields"])
            if cycle_draw:
                cycle, position = divmod(draw, entry["fields"])
                if cycle != previous_cycle:
                    permutation = np.random.default_rng(stable_seed(seed, section, cycle,
                        "v030_field_cycle")).permutation(entry["fields"])
                    previous_cycle = cycle
                index = int(permutation[position])
            if "field_indices" in entry:
                if len(entry["field_indices"]) != entry["fields"]:
                    raise ValueError("eligible training pool size differs")
                index = int(entry["field_indices"][index])
            items.append(dict(sample=section, field_index=index, protocol_id=entry["protocol_id"],
                mask_seed=stable_seed(seed, section, index, "random_owner"), coverage_stratum=draw % 3))
            if cycle_draw:
                items[-1].update(sampling_cycle=cycle,
                    mask_seed=stable_seed(seed, section, index, cycle, "random_owner"))
    rng = np.random.default_rng(stable_seed(seed, epoch, "v030_batch_order"))
    return [items[i] for i in rng.permutation(len(items))]


def exposure_plan(section_budgets, *, batch_size, max_epochs, minimum_per_section):
    """Report exact optimizer budget; reject dilution relative to explicit floors."""
    if type(batch_size) is not int or batch_size <= 0 or (max_epochs is not None
            and (type(max_epochs) is not int or max_epochs <= 0)):
        raise ValueError("positive integer batch size and optional positive epoch cap required")
    if not section_budgets or set(section_budgets) != set(minimum_per_section):
        raise ValueError("provide an exposure floor for every section")
    for section, count in section_budgets.items():
        floor = minimum_per_section[section]
        if type(count) is not int or type(floor) is not int or floor <= 0 or count < floor:
            raise ValueError(f"section exposure below its approved floor: {section}")
    total = sum(section_budgets.values())
    steps = math.ceil(total / batch_size)
    return dict(fovs_per_epoch=total, steps_per_epoch=steps,
        maximum_optimizer_steps=None if max_epochs is None else steps * max_epochs,
        maximum_exposure_per_section={key: None if max_epochs is None else value * max_epochs for key, value in section_budgets.items()},
        last_batch_size=(total - 1) % batch_size + 1)
