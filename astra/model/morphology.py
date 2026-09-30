"""Shared multiscale morphology with private protocol residuals and frozen spatial inputs.

The local and pooled branches learn from scratch; they are not pretrained pathology
semantics. Optional pathology features must come from a separately frozen encoder.
"""

import torch
from torch import nn
from torch.nn import functional as F

from astra.model.image import ExpandedMaskedImageEncoder


def _masked_pool(value, valid, scale):
    kernel = (min(scale, value.shape[-2]), min(scale, value.shape[-1]))
    mask = valid[:, None]
    clean = torch.where(mask, value, 0.0)
    support = F.avg_pool2d(mask.to(value.dtype), kernel, stride=kernel, ceil_mode=True, count_include_pad=False)
    pooled = F.avg_pool2d(clean, kernel, stride=kernel, ceil_mode=True, count_include_pad=False)
    return pooled / torch.where(support > 0, support, 1.0), support[:, 0] > 0


def _masked_resize(value, valid, size):
    mask = valid[:, None]
    support = F.interpolate(mask.to(value.dtype), size=size, mode="bilinear", align_corners=False)
    resized = F.interpolate(torch.where(mask, value, 0.0), size=size, mode="bilinear", align_corners=False)
    return resized / torch.where(support > 0, support, 1.0)


class MultiScaleMorphology(nn.Module):
    def __init__(self, input_channels=6, image_channels=64, dilations=(1, 2, 4),
                 residual_blocks=2, normalization_groups=8, pathology_channels=0,
                 use_pathology=True, use_multiscale=True, adaptive_fusion=False):
        super().__init__()
        if pathology_channels < 0:
            raise ValueError("pathology_channels must be nonnegative")
        self.input_channels = int(input_channels)
        self.image_channels = int(image_channels)
        self.pathology_channels = int(pathology_channels)
        self.use_pathology = bool(use_pathology)
        self.use_multiscale = bool(use_multiscale)
        self.local_encoder = ExpandedMaskedImageEncoder(
            input_channels=self.input_channels, hidden_channels=self.image_channels,
            dilations=tuple(dilations), residual_blocks=residual_blocks,
            normalization_groups=normalization_groups,
        )
        self.scale_branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.image_channels, self.image_channels, 3, padding=1, bias=False), nn.GELU())
            for _ in (4, 16)
        ])
        if self.pathology_channels:
            self.pathology_projection = nn.Sequential(
                nn.Conv2d(self.pathology_channels, self.image_channels, 1, bias=False), nn.GELU(),
            )
        else:
            self.pathology_projection = None
        branches = 3 + int(self.pathology_channels > 0)
        self.fusion = nn.Conv2d(branches * self.image_channels, self.image_channels, 1, bias=False)
        bottleneck = max(1, self.image_channels // 4)
        self.protocol_adapters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.image_channels, bottleneck, 1, bias=False), nn.GELU(),
                          nn.Conv2d(bottleneck, self.image_channels, 1, bias=False))
            for _ in (0, 1)
        ])
        for adapter in self.protocol_adapters:
            nn.init.zeros_(adapter[-1].weight)
        if not self.use_pathology and self.pathology_projection is not None:
            self.pathology_projection.requires_grad_(False)
        if not self.use_multiscale:
            self.scale_branches.requires_grad_(False)
        self.fusion_gate = nn.Conv2d(branches * self.image_channels, branches, 1) if adaptive_fusion else None
        if self.fusion_gate is not None:
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.zeros_(self.fusion_gate.bias)

    def forward(self, image_features_2um, field_valid, protocol_id, *,
                pathology_features=None, pathology_valid=None):
        """Return [B,C,H,W] features, with invalid 2um locations exactly zero.

        Pathology input is a floating [B,Cp,Hs,Ws] spatial grid with boolean
        [B,Hs,Ws] validity. Its full physical extent and orientation must match
        the same FOV; CLS or globally pooled features are not spatial inputs.
        Cached features are detached, while their projection remains trainable.
        """
        image = image_features_2um
        if (image.ndim != 4 or image.shape[1] != self.input_channels or not image.is_floating_point()
                or field_valid.shape != image.shape[:1] + image.shape[2:] or field_valid.dtype != torch.bool):
            raise ValueError("morphology requires floating BCHW images and matching boolean BHW validity")
        if (protocol_id.shape != image.shape[:1] or protocol_id.dtype != torch.long
                or bool(torch.any((protocol_id < 0) | (protocol_id > 1)))):
            raise ValueError("morphology protocol_id must be normalized long [B] values 0 or 1")
        if field_valid.device != image.device or protocol_id.device != image.device:
            raise ValueError("morphology image, validity and protocol devices differ")
        if not self.pathology_channels and (pathology_features is not None or pathology_valid is not None):
            raise ValueError("pathology inputs supplied while pathology_channels is zero")
        if self.pathology_channels and self.use_pathology:
            if pathology_features is None or pathology_valid is None:
                raise ValueError("configured pathology branch requires frozen spatial features and validity")
            if (pathology_features.ndim != 4 or pathology_features.shape[:2] != (image.shape[0], self.pathology_channels)
                    or min(pathology_features.shape[2:]) <= 0 or not pathology_features.is_floating_point()
                    or pathology_valid.shape != pathology_features.shape[:1] + pathology_features.shape[2:]
                    or pathology_valid.dtype != torch.bool or pathology_features.device != image.device
                    or pathology_valid.device != image.device):
                raise ValueError("pathology features must be an aligned floating BCHW grid with boolean BHW validity")
        mask = field_valid[:, None]
        # The reused encoder multiplies by its mask, so remove invalid NaNs first.
        local = self.local_encoder(torch.where(mask, image, 0.0), field_valid)
        branches = [local]
        for scale, branch in zip((4, 16), self.scale_branches):
            if self.use_multiscale:
                pooled, valid = _masked_pool(local, field_valid, scale)
                coarse = torch.where(valid[:, None], branch(pooled), 0.0)
                branches.append(_masked_resize(coarse, valid, image.shape[2:]))
            else:
                branches.append(torch.zeros_like(local))
        if self.pathology_projection is not None:
            if self.use_pathology:
                frozen = pathology_features.detach().to(dtype=image.dtype)
                projected = self.pathology_projection(torch.where(pathology_valid[:, None], frozen, 0.0))
                projected = torch.where(pathology_valid[:, None], projected, 0.0)
                branches.append(_masked_resize(projected, pathology_valid, image.shape[2:]))
            else:
                branches.append(torch.zeros_like(local))
        if self.fusion_gate is not None:
            gates = 1.0 + torch.tanh(self.fusion_gate(torch.cat(branches, dim=1)))
            branches = [branch * gates[:, index:index+1] for index, branch in enumerate(branches)]
        shared = torch.where(mask, self.fusion(torch.cat(branches, dim=1)), 0.0)
        result = shared
        for protocol, adapter in enumerate(self.protocol_adapters):
            indices = torch.nonzero(protocol_id == protocol, as_tuple=False).flatten()
            if indices.numel():
                selected = shared.index_select(0, indices)
                result = result.index_copy(0, indices, selected + adapter(selected))
        return torch.where(mask, result, 0.0)
