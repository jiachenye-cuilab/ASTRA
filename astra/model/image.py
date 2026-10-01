"""Masked H&E image encoders and residual blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaskedLocalResidualBlock(nn.Module):
    """A local image block that cannot repopulate invalid field locations."""

    def __init__(self, channels: int, *, groups: int) -> None:
        super().__init__()
        channels = int(channels)
        groups = int(groups)
        if channels <= 0 or groups <= 0 or channels % groups:
            raise ValueError("ASTRA image residual block dimensions differ")
        self.normalization = nn.GroupNorm(groups, channels)
        self.first = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.second = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or mask.shape != value.shape[:1] + (1,) + value.shape[2:]:
            raise ValueError("ASTRA image residual mask shape differs")
        update = self.normalization(value)
        update = self.first(update) * mask
        update = F.gelu(update)
        update = self.second(update) * mask
        return (value + update) * mask


class ExpandedMaskedImageEncoder(nn.Module):
    """Three dilated convolutions followed by two local residual blocks."""

    def __init__(
        self,
        *,
        input_channels: int,
        hidden_channels: int,
        dilations: tuple[int, ...] = (1, 2, 4),
        residual_blocks: int = 2,
        normalization_groups: int = 8,
    ) -> None:
        super().__init__()
        input_channels = int(input_channels)
        hidden_channels = int(hidden_channels)
        dilations = tuple(int(value) for value in dilations)
        residual_blocks = int(residual_blocks)
        if (
            input_channels <= 0
            or hidden_channels <= 0
            or not dilations
            or any(value <= 0 for value in dilations)
            or residual_blocks <= 0
        ):
            raise ValueError("ASTRA image encoder dimensions differ")
        channels = (input_channels,) + (hidden_channels,) * len(dilations)
        self.layers = nn.ModuleList(
            nn.Conv2d(
                channels[index],
                channels[index + 1],
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
            for index, dilation in enumerate(dilations)
        )
        self.local_blocks = nn.ModuleList(
            MaskedLocalResidualBlock(
                hidden_channels, groups=int(normalization_groups)
            )
            for _ in range(residual_blocks)
        )

    def forward(self, image: torch.Tensor, field_valid: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or field_valid.shape != image.shape[:1] + image.shape[2:]:
            raise ValueError("ASTRA image/valid field shapes differ")
        mask = field_valid[:, None].to(dtype=image.dtype)
        hidden = image * mask
        for layer in self.layers:
            hidden = layer(hidden) * mask
            hidden = F.gelu(hidden)
        for block in self.local_blocks:
            hidden = block(hidden, mask)
        return hidden
