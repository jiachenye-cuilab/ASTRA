"""Frozen spatial UNI features loaded from explicit local weights and config."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def tokens_to_grid(tokens: torch.Tensor, *, prefix_tokens: int,
                   grid_shape: tuple[int, int]) -> torch.Tensor:
    """Remove CLS/register prefixes and retain row-major patch positions."""
    height, width = grid_shape
    if (tokens.ndim != 3 or prefix_tokens < 0 or min(height, width) <= 0
            or tokens.shape[1] != prefix_tokens + height * width):
        raise ValueError("UNI token count does not match its patch grid and prefix tokens")
    patches = tokens[:, prefix_tokens:]
    return patches.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], height, width).contiguous()


class FrozenUNI(nn.Module):
    """UNI1 ViT-L/16 using a complete square physical FOV without cropping.

    Floating RGB in [0,1] is resized to 224 pixels and normalized with ImageNet
    statistics. Output shape is [B,1024,14,14]; token pitch is FOV extent / 14.
    """

    def __init__(self, checkpoint_path: str | Path, *, config_path: str | Path):
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("FrozenUNI requires timm; no download or installation is attempted") from error
        checkpoint = Path(checkpoint_path).expanduser().absolute()
        config_path = Path(config_path).expanduser().absolute()
        if not checkpoint.is_file() or not config_path.is_file():
            raise FileNotFoundError("UNI requires explicit local weights and config files")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected = dict(architecture="vit_large_patch16_224", patch_size=16, img_size=224,
                        init_values=1.0, num_classes=0, num_features=1024,
                        global_pool="token", dynamic_img_size=True)
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("local UNI config is not the supported UNI1 ViT-L/16 architecture")
        preprocessing = config["pretrained_cfg"]
        if (tuple(preprocessing["mean"]) != IMAGENET_MEAN
                or tuple(preprocessing["std"]) != IMAGENET_STD
                or preprocessing["input_size"] != [3, 224, 224]
                or preprocessing["interpolation"] != "bilinear"):
            raise ValueError("local UNI preprocessing differs from the declared RGB normalization")
        self.input_size = (224, 224)
        self.patch_size = (16, 16)
        self.channels = 1024
        # Meta construction avoids a second weight allocation; pretrained=False
        # keeps model creation independent of network access.
        self.backbone = timm.create_model(
            config["architecture"], pretrained=False, img_size=224, patch_size=16,
            init_values=1.0, num_classes=0, global_pool="token", dynamic_img_size=True,
            weight_init="skip", device="meta",
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.backbone.load_state_dict(state, strict=True, assign=True)
        if (any(p.is_meta for p in self.backbone.parameters())
                or tuple(self.backbone.patch_embed.patch_size) != self.patch_size
                or self.backbone.num_features != self.channels):
            raise ValueError("local UNI weights did not materialize the declared architecture")
        self.register_buffer("rgb_mean", torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1), persistent=False)
        self.train(False)

    def train(self, mode: bool = True):
        # Containing pipelines must not enable UNI dropout or gradients.
        super().train(False)
        self.backbone.requires_grad_(False)
        return self

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor, *, extent_um: float) -> torch.Tensor:
        extent = float(extent_um)
        if not math.isfinite(extent) or extent <= 0:
            raise ValueError("extent_um must state the positive square FOV side length")
        if (rgb.ndim != 4 or rgb.shape[0] < 1 or rgb.shape[1] != 3
                or rgb.shape[2] != rgb.shape[3] or rgb.shape[2] < 1
                or not rgb.is_floating_point()):
            raise ValueError("UNI expects a complete square RGB FOV as floating [B,3,H,H], not OD/H/E")
        if not bool(torch.isfinite(rgb).all()) or bool(torch.any((rgb < 0) | (rgb > 1))):
            raise ValueError("UNI RGB values must be finite and in [0,1]")
        if rgb.device != self.rgb_mean.device:
            raise ValueError("move FrozenUNI and its RGB batch to the same device")
        self.train(False)
        image = F.interpolate(rgb.to(self.rgb_mean.dtype), size=self.input_size,
                              mode="bilinear", align_corners=False, antialias=True)
        image = (image-self.rgb_mean) / self.rgb_std
        tokens = self.backbone.forward_features(image)
        if not isinstance(tokens, torch.Tensor) or tokens.shape[-1] != self.channels:
            raise ValueError("UNI forward_features did not return the declared token features")
        grid = tokens_to_grid(tokens, prefix_tokens=int(self.backbone.num_prefix_tokens),
                              grid_shape=tuple(i // p for i, p in zip(self.input_size, self.patch_size)))
        if not bool(torch.isfinite(grid).all()):
            raise ValueError("UNI spatial features contain nonfinite values")
        return grid
