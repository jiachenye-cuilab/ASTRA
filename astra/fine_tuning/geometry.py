from dataclasses import dataclass
from typing import Any
import torch
from astra.model.ownership import _spatial_valid, validate_owner_batch

def masks_to_owner(
    masks: torch.Tensor,
    *,
    field_valid: torch.Tensor | None = None,
    check_connectivity: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert external binary masks to canonical owner ids, rejecting overlap."""

    value = torch.as_tensor(masks)
    squeeze = value.ndim == 3
    if squeeze:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.dtype != torch.bool:
        raise TypeError("v033 masks must be bool [batch,parents,rows,columns]")
    batch, parents, rows, columns = value.shape
    valid_field = (
        torch.ones((batch, rows, columns), dtype=torch.bool, device=value.device)
        if field_valid is None
        else _spatial_valid(field_valid, (batch, rows, columns))
    )
    if bool(torch.any(value.sum(dim=1) > 1)):
        raise ValueError("v033 parent masks overlap")
    if bool(torch.any(value & ~valid_field[:, None])):
        raise ValueError("v033 parent mask includes invalid field cells")
    parent_valid = value.flatten(2).any(dim=-1)
    ids = torch.arange(parents, dtype=torch.long, device=value.device)
    owner = torch.where(
        value,
        ids[None, :, None, None],
        torch.full((), -1, dtype=torch.long, device=value.device),
    ).amax(dim=1)
    validate_owner_batch(
        owner,
        parent_valid,
        valid_field,
        check_connectivity=check_connectivity,
    )
    return (owner[0], parent_valid[0]) if squeeze else (owner, parent_valid)


@dataclass(frozen=True)
class OwnerMapSample:
    owner_map: torch.Tensor
    parent_valid: torch.Tensor
    parameters: dict[str, Any]

    @property
    def parents(self) -> int:
        return int(self.parent_valid.sum())

    @property
    def coverage(self) -> float:
        return float((self.owner_map >= 0).to(dtype=torch.float64).mean())


def axis_aligned_square_owner(
    *,
    side_um: float,
    phase_um: tuple[float, float] = (0.0, 0.0),
    field_shape: tuple[int, int] = (128, 128),
    cell_um: float = 2.0,
    device: torch.device | str = "cpu",
) -> OwnerMapSample:
    """Rasterize all full, non-overlapping phased square parents."""

    rows, columns = (int(value) for value in field_shape)
    side = float(side_um)
    phase_y, phase_x = (float(value) for value in phase_um)
    if side <= 0 or cell_um <= 0 or not 0 <= phase_y < side or not 0 <= phase_x < side:
        raise ValueError("v033 square side/phase differs")
    fov_y, fov_x = rows * cell_um, columns * cell_um
    starts_y = torch.arange(phase_y, fov_y - side + 1e-12, side, dtype=torch.float64)
    starts_x = torch.arange(phase_x, fov_x - side + 1e-12, side, dtype=torch.float64)
    if starts_y.numel() == 0 or starts_x.numel() == 0:
        raise ValueError("v033 phased square program has no full parent")
    yy = (torch.arange(rows, dtype=torch.float64) + 0.5) * cell_um
    xx = (torch.arange(columns, dtype=torch.float64) + 0.5) * cell_um
    masks = []
    for start_y in starts_y:
        for start_x in starts_x:
            masks.append(
                ((yy[:, None] >= start_y) & (yy[:, None] < start_y + side))
                & ((xx[None, :] >= start_x) & (xx[None, :] < start_x + side))
            )
    mask_tensor = torch.stack(masks).to(device=device)
    owner, valid = masks_to_owner(mask_tensor, check_connectivity=True)
    return OwnerMapSample(
        owner_map=owner,
        parent_valid=valid,
        parameters={
            "program": "axis_aligned_square",
            "side_um": side,
            "phase_um": [phase_y, phase_x],
            "parents": int(valid.sum()),
            "actual_coverage": float((owner >= 0).double().mean()),
        },
    )


def canonical_hd16_owner(
    *,
    field_shape: tuple[int, int] = (128, 128),
    cell_um: float = 2.0,
    device: torch.device | str = "cpu",
) -> OwnerMapSample:
    return axis_aligned_square_owner(
        side_um=16.0,
        phase_um=(0.0, 0.0),
        field_shape=field_shape,
        cell_um=cell_um,
        device=device,
    )


def core_supervision(*, batch_size=1, device="cpu", core_um=192):
    if type(core_um) is not int or not 0 < core_um <= 256 or core_um % 16:
        raise ValueError("centered supervision must align to the 8um output grid")
    mask = torch.zeros((batch_size, 128, 128), dtype=torch.bool, device=device)
    margin = (256 - core_um) // 4
    mask[:, margin:128-margin, margin:128-margin] = True
    return mask
