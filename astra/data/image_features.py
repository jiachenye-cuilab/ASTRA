"""Six-channel local image features; original absolute and relative semantics."""
import torch

def image_features_from_raw(
    density: torch.Tensor,
    valid_area: torch.Tensor,
    field_valid: torch.Tensor,
    *,
    valid_area_scale: float = 65535.0,
    relative_log_epsilon: float = 1e-6,
    relative_log_clip: float = 5.0,
) -> torch.Tensor:
    """Create the six frozen H&E channels on CPU or GPU after batch transfer."""

    values = torch.as_tensor(density)
    area = torch.as_tensor(valid_area, device=values.device)
    valid = torch.as_tensor(field_valid, device=values.device, dtype=torch.bool)
    if (
        values.ndim != 4
        or values.shape[-1] != 3
        or area.shape != values.shape[:3]
        or valid.shape != values.shape[:3]
        or values.dtype != torch.float32
        or valid_area_scale <= 0
        or relative_log_epsilon <= 0
        or relative_log_clip <= 0
    ):
        raise ValueError("ASTRA raw H&E feature tensors differ")
    weight = area.to(dtype=torch.float64) / float(valid_area_scale)
    denominator = weight.sum(dim=(1, 2)).clamp_min(torch.finfo(torch.float64).tiny)
    mean = (
        values.to(dtype=torch.float64) * weight[..., None]
    ).sum(dim=(1, 2)) / denominator[:, None]
    epsilon = float(relative_log_epsilon)
    absolute = torch.log1p(values)
    relative = torch.log(
        (values.to(dtype=torch.float64) + epsilon) / (mean[:, None, None] + epsilon)
    ).clamp(-float(relative_log_clip), float(relative_log_clip)).to(dtype=torch.float32)
    result = torch.cat((absolute, relative), dim=-1)
    return torch.where(valid[..., None], result, torch.zeros_like(result))
