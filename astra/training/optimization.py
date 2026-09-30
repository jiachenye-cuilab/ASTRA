"""Optimization schedule, bounded selection, and stopping thresholds."""

import math


def thresholds(value):
    tasks = ("HD16", "Spot55")
    values = value if isinstance(value, dict) else dict.fromkeys(tasks, value)
    if set(values) != set(tasks):
        raise ValueError("thresholds must specify HD16 and Spot55")
    result = {task: float(values[task]) for task in tasks}
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError("task thresholds must be finite and nonnegative")
    return result


def bounded_selection(previous, scores, *, epoch, minimum_improvement, maximum_regression,
                      patience_checks, minimum_epochs):
    """Keep the original strict task tolerances and fixed first-validation guard."""
    tasks = ("HD16", "Spot55")
    if set(scores) != set(tasks) or not all(math.isfinite(float(scores[t])) for t in tasks):
        raise ValueError("finite scores for both tasks are required")
    if previous is not None and epoch <= previous["last_validation_epoch"]:
        raise ValueError("validation epochs must advance")
    improvements, regressions = thresholds(minimum_improvement), thresholds(maximum_regression)
    current = {task: float(scores[task]) for task in tasks}
    best = None if previous is None else previous["best_scores"]
    anchor = current if previous is None else previous["anchor_scores"]
    accepted = best is None or (
        all(current[t] <= best[t] if regressions[t] == 0 else current[t] < best[t] + regressions[t]
            for t in tasks)
        and all(current[t] <= anchor[t] if regressions[t] == 0 else current[t] < anchor[t] + regressions[t]
                for t in tasks)
        and any(best[t] - current[t] > improvements[t] for t in tasks))
    stale = 0 if accepted else previous["stale_checks"] + 1
    return dict(best_scores=current if accepted else dict(best), anchor_scores=dict(anchor),
                best_epoch=epoch if accepted else previous["best_epoch"],
                last_validation_epoch=epoch, stale_checks=stale, accepted=accepted,
                should_stop=epoch >= minimum_epochs and stale >= patience_checks)


def cosine_multiplier(updates, settings):
    """Match the historical LambdaLR value before each optimizer update."""
    warmup = settings["warmup_optimizer_steps"]
    effective = settings["batch_size"] * settings["gradient_accumulation_steps"]
    maximum = math.ceil(settings["fields_per_epoch"] / effective) * settings["max_epochs"]
    completed = updates + 1
    if warmup and completed <= warmup:
        return completed / warmup
    progress = min((completed - warmup) / max(1, maximum - warmup), 1.)
    minimum = settings["minimum_learning_rate_ratio"]
    return minimum + (1. - minimum) * .5 * (1. + math.cos(math.pi * progress))
