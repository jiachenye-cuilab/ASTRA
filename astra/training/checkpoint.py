"""Complete training checkpoints with strict architecture and gene identity checks."""

from copy import deepcopy
from pathlib import Path

import torch

from astra.training.model_factory import model_class, model_family
from astra.training.panels import GenePanels


def _safe_training_state(value):
    """Reject objects that cannot round-trip through weights_only=True loading."""
    if value is None or isinstance(value, (str, bool, int, float, torch.Tensor)):
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, (str, int, float, bool)) for key in value):
            raise TypeError("training_state dictionary keys must be primitive values")
        return {key: _safe_training_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_safe_training_state(item) for item in value)
    raise TypeError("training_state must contain only tensors and Python primitives; convert numpy RNG arrays")


def save_checkpoint(path, model, *, optimizer=None, step=0, training_state=None):
    """Save the complete core and optional branches without overwriting any file.

    The caller can store epoch, scheduler, scaler and RNG state in training_state.
    Optimizer restoration remains explicit via payload["optimizer"]. No external
    weight resource or older version is needed to restore the checkpoint.
    """
    family = model_family(model)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if training_state is not None and not isinstance(training_state, dict):
        raise TypeError("training_state must be a dictionary or None")
    payload = dict(version="v033", kwargs=model.model_kwargs,
                   panels=model.panels.as_dict(), model=model.state_dict(),
                   step=step, training=model.training)
    if family != "v033":
        payload["model_family"] = family
    if training_state is not None:
        payload["training_state"] = _safe_training_state(training_state)
    if optimizer is not None:
        owned = {id(parameter) for parameter in model.parameters()}
        optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        identities = [id(parameter) for parameter in optimized]
        if len(set(identities)) != len(identities) or not set(identities).issubset(owned):
            raise ValueError("optimizer must contain unique parameters belonging to this v033 model")
        payload["optimizer"] = optimizer.state_dict()
    path = Path(path)
    # Exclusive creation keeps existing training and failed-attempt artifacts intact.
    with path.open("xb") as stream:
        try:
            torch.save(payload, stream)
        except Exception:
            path.unlink()
            raise


def load_checkpoint(path, *, expected_panels=None, device="cpu"):
    """Restore only v033 payloads, validating ordered genes and component flags."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("version") != "v033":
        raise ValueError("expected a complete v033 checkpoint")
    if not isinstance(payload.get("kwargs"), dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("v033 checkpoint must include constructor arguments and full model state")
    kwargs = deepcopy(payload["kwargs"])
    family = payload.get("model_family", "v033")
    constructor = model_class(family)
    # Only this newly introduced field has a legacy default; every weight and
    # the remaining metadata still undergo strict loading below.
    if family == "v033":
        kwargs.setdefault("allocation_post_center_norm", "layernorm")
    panels = GenePanels(kwargs.get("input_gene_ids", ()), kwargs.get("output_gene_ids", ()))
    if payload.get("panels") != panels.as_dict():
        raise ValueError("v033 checkpoint gene identities/order disagree with constructor arguments")
    if expected_panels is not None:
        if isinstance(expected_panels, GenePanels):
            expected = expected_panels
        elif isinstance(expected_panels, dict):
            expected = GenePanels(**expected_panels)
        else:
            from astra.model.model import GenePanels as ControlPanels
            if not isinstance(expected_panels, ControlPanels):
                raise TypeError("expected_panels must be GenePanels or an ordered gene-panel dictionary")
            expected = GenePanels(**expected_panels.as_dict())
        if expected != panels:
            raise ValueError("v033 checkpoint gene identities/order differ from expected_panels")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("v033 checkpoint step must be a nonnegative integer")
    if not isinstance(payload.get("training"), bool):
        raise ValueError("v033 checkpoint must record a boolean training mode")
    model = constructor(**kwargs)
    # Check metadata before copying any weights, including independent branch flags.
    model.set_extra_state(payload["model"].get("_extra_state"))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.train(payload["training"])
    return model, payload


__all__ = ["save_checkpoint", "load_checkpoint"]
