"""Registered RGB sampling and area-integrated optical-density/H&E features."""

from __future__ import annotations

import math

import numpy as np
import torch


_STAIN_MATRIX = np.asarray(
    [[0.650, 0.072, 0.268], [0.704, 0.990, 0.570], [0.286, 0.105, 0.776]],
    dtype=np.float64,
)
_STAIN_MATRIX /= np.linalg.norm(_STAIN_MATRIX, axis=0, keepdims=True)
_STAIN_INVERSE = np.linalg.inv(_STAIN_MATRIX)
DENSITY_CHANNELS = ("rgb_optical_density", "hematoxylin", "eosin")
VALID_FRACTION_SCALE = np.iinfo(np.uint16).max


def rgb_density_maps(
    rgb: np.ndarray,
    *,
    row_chunk: int = 256,
) -> dict[str, np.ndarray]:
    """Compute float32 optical-density and H&E maps with bounded memory."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape [rows, cols, 3]")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    optical_mean = np.empty(rgb.shape[:2], dtype=np.float32)
    hematoxylin = np.empty(rgb.shape[:2], dtype=np.float32)
    eosin = np.empty(rgb.shape[:2], dtype=np.float32)
    inverse = _STAIN_INVERSE.astype(np.float32)
    for start in range(0, rgb.shape[0], row_chunk):
        stop = min(start + row_chunk, rgb.shape[0])
        optical = -np.log((rgb[start:stop].astype(np.float32) + 1.0) / 256.0)
        concentrations = np.einsum("...c,kc->...k", optical, inverse)
        optical_mean[start:stop] = optical.mean(axis=-1)
        hematoxylin[start:stop] = np.clip(concentrations[..., 0], 0.0, None)
        eosin[start:stop] = np.clip(concentrations[..., 1], 0.0, None)
    return {
        "rgb_optical_density": optical_mean,
        "hematoxylin": hematoxylin,
        "eosin": eosin,
    }


def read_rgb_region(
    reader,
    *,
    x_start: int,
    y_start: int,
    width: int,
    height: int,
    fill_value: int = 255,
) -> np.ndarray:
    """Read an RGB rectangle, padding out-of-image pixels without wraparound."""
    x_start = int(x_start)
    y_start = int(y_start)
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    output = np.full((height, width, 3), fill_value, dtype=np.uint8)
    x0 = max(x_start, 0)
    y0 = max(y_start, 0)
    x1 = min(x_start + width, int(reader.width))
    y1 = min(y_start + height, int(reader.height))
    if x0 >= x1 or y0 >= y1:
        return output
    tile_col_start = x0 // int(reader.tile_width)
    tile_col_stop = (x1 - 1) // int(reader.tile_width) + 1
    tile_row_start = y0 // int(reader.tile_height)
    tile_row_stop = (y1 - 1) // int(reader.tile_height) + 1
    for tile_row in range(tile_row_start, tile_row_stop):
        for tile_col in range(tile_col_start, tile_col_stop):
            tile_index = tile_row * int(reader.tile_columns) + tile_col
            tile = reader.read_tile(tile_index)
            tile_x0 = tile_col * int(reader.tile_width)
            tile_y0 = tile_row * int(reader.tile_height)
            overlap_x0 = max(x0, tile_x0)
            overlap_y0 = max(y0, tile_y0)
            overlap_x1 = min(x1, tile_x0 + int(reader.tile_width))
            overlap_y1 = min(y1, tile_y0 + int(reader.tile_height))
            output[
                overlap_y0 - y_start : overlap_y1 - y_start,
                overlap_x0 - x_start : overlap_x1 - x_start,
            ] = tile[
                overlap_y0 - tile_y0 : overlap_y1 - tile_y0,
                overlap_x0 - tile_x0 : overlap_x1 - tile_x0,
            ]
    return output


def project_points(
    transform: np.ndarray, columns: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("spot-to-microscope transform differs")
    columns = np.asarray(columns, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    denominator = (
        matrix[2, 0] * columns + matrix[2, 1] * rows + matrix[2, 2]
    )
    if np.any(np.abs(denominator) <= 0.99):
        raise ValueError("spot-to-microscope transform denominator is unstable")
    x = (
        matrix[0, 0] * columns + matrix[0, 1] * rows + matrix[0, 2]
    ) / denominator
    y = (
        matrix[1, 0] * columns + matrix[1, 1] * rows + matrix[1, 2]
    ) / denominator
    return x, y


def _mapped_bounds(
    transform: np.ndarray,
    *,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
) -> tuple[float, float, float, float]:
    columns = np.asarray(
        [column_start - 0.5, column_stop - 0.5, column_start - 0.5, column_stop - 0.5],
        dtype=np.float64,
    )
    rows = np.asarray(
        [row_start - 0.5, row_start - 0.5, row_stop - 0.5, row_stop - 0.5],
        dtype=np.float64,
    )
    x, y = project_points(transform, columns, rows)
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def full_valid_cells(
    transform: np.ndarray,
    *,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    rows = np.arange(row_start, row_stop, dtype=np.float64)[:, None, None]
    columns = np.arange(column_start, column_stop, dtype=np.float64)[None, :, None]
    row_corners = rows + np.asarray([-0.5, -0.5, 0.5, 0.5])[None, None, :]
    column_corners = columns + np.asarray([-0.5, 0.5, -0.5, 0.5])[None, None, :]
    x, y = project_points(transform, column_corners, row_corners)
    return (
        (x >= -0.5)
        & (x < float(image_width) - 0.5)
        & (y >= -0.5)
        & (y < float(image_height) - 0.5)
    ).all(axis=-1)


def aggregate_density_region(
    reader,
    transform: np.ndarray,
    *,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
    samples_per_cell_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate OD/H/E over spot-grid cells using deterministic midpoint quadrature."""

    k = int(samples_per_cell_axis)
    if (
        k <= 0
        or row_start < 0
        or column_start < 0
        or row_stop <= row_start
        or column_stop <= column_start
    ):
        raise ValueError("area-integration region differs")
    x_min, x_max, y_min, y_max = _mapped_bounds(
        transform,
        row_start=row_start,
        row_stop=row_stop,
        column_start=column_start,
        column_stop=column_stop,
    )
    source_x0 = max(0, int(math.floor(x_min)) - 2)
    source_y0 = max(0, int(math.floor(y_min)) - 2)
    source_x1 = min(int(reader.width), int(math.ceil(x_max)) + 3)
    source_y1 = min(int(reader.height), int(math.ceil(y_max)) + 3)
    output_rows = row_stop - row_start
    output_columns = column_stop - column_start
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return (
            np.zeros((output_rows, output_columns, 3), dtype=np.float32),
            np.zeros((output_rows, output_columns), dtype=np.float32),
        )
    rgb = read_rgb_region(
        reader,
        x_start=source_x0,
        y_start=source_y0,
        width=source_x1 - source_x0,
        height=source_y1 - source_y0,
    )
    source_density = rgb_density_maps(rgb)
    del rgb

    sample_columns = (
        column_start
        - 0.5
        + (np.arange(output_columns * k, dtype=np.float64) + 0.5) / k
    )
    sample_rows = (
        row_start
        - 0.5
        + (np.arange(output_rows * k, dtype=np.float64) + 0.5) / k
    )
    grid_columns = sample_columns[None, :]
    grid_rows = sample_rows[:, None]
    x, y = project_points(transform, grid_columns, grid_rows)
    inside = (
        (x >= -0.5)
        & (x < float(reader.width) - 0.5)
        & (y >= -0.5)
        & (y < float(reader.height) - 0.5)
    )
    clipped_x = np.clip(x, 0.0, float(reader.width - 1))
    clipped_y = np.clip(y, 0.0, float(reader.height - 1))
    x0 = np.floor(clipped_x).astype(np.int64)
    y0 = np.floor(clipped_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, int(reader.width - 1))
    y1 = np.minimum(y0 + 1, int(reader.height - 1))
    if (
        x0.min() < source_x0
        or x1.max() >= source_x1
        or y0.min() < source_y0
        or y1.max() >= source_y1
    ):
        raise ValueError("RGB source crop does not cover interpolation support")
    local_x0 = x0 - source_x0
    local_x1 = x1 - source_x0
    local_y0 = y0 - source_y0
    local_y1 = y1 - source_y0
    fraction_x = clipped_x - x0
    fraction_y = clipped_y - y0
    w00 = (1.0 - fraction_x) * (1.0 - fraction_y)
    w01 = fraction_x * (1.0 - fraction_y)
    w10 = (1.0 - fraction_x) * fraction_y
    w11 = fraction_x * fraction_y
    coverage = inside.reshape(output_rows, k, output_columns, k).mean(
        axis=(1, 3), dtype=np.float64
    )
    density = np.zeros((output_rows, output_columns, 3), dtype=np.float32)
    for channel_index, channel in enumerate(DENSITY_CHANNELS):
        source = source_density[channel]
        sampled = (
            source[local_y0, local_x0] * w00
            + source[local_y0, local_x1] * w01
            + source[local_y1, local_x0] * w10
            + source[local_y1, local_x1] * w11
        )
        sampled[~inside] = 0.0
        numerator = sampled.reshape(output_rows, k, output_columns, k).mean(
            axis=(1, 3), dtype=np.float64
        )
        np.divide(
            numerator,
            coverage,
            out=density[..., channel_index],
            where=coverage > 0,
        )
    if not np.isfinite(density).all() or not np.isfinite(coverage).all():
        raise ValueError("area-integrated image contains non-finite values")
    return density, coverage.astype(np.float32, copy=False)


def quantize_valid_fraction(values: np.ndarray) -> np.ndarray:
    fractions = np.asarray(values, dtype=np.float64)
    if not np.isfinite(fractions).all() or np.any(fractions < 0) or np.any(fractions > 1):
        raise ValueError("valid-area fraction differs")
    return np.rint(fractions * VALID_FRACTION_SCALE).astype(np.uint16)


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
        raise ValueError("raw H&E feature tensors differ")
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


def sample_registered_rgb(reader, transform, origin_2um, output_size=224):
    """Bilinearly sample full-FOV pixel centers; outside-image samples are white."""
    if not isinstance(output_size, int) or output_size <= 0:
        raise ValueError("RGB output_size must be a positive integer")
    start = np.asarray(origin_2um)
    if start.shape != (2,) or not np.issubdtype(start.dtype, np.integer) or np.any(start < 0):
        raise ValueError("RGB FOV origin must be a nonnegative integer row/column pair")
    row_start, column_start = map(int, start)
    offsets = -0.5 + (np.arange(output_size, dtype=np.float64) + 0.5) * 128 / output_size
    x, y = project_points(transform, column_start + offsets[None, :], row_start + offsets[:, None])
    valid = ((x >= -0.5) & (x < reader.width - 0.5) & (y >= -0.5) & (y < reader.height - 0.5))
    output = np.ones((output_size, output_size, 3), dtype=np.float32)
    if not valid.any():
        return output, valid
    xmin, xmax, ymin, ymax = _mapped_bounds(
        transform, row_start=row_start, row_stop=row_start + 128,
        column_start=column_start, column_stop=column_start + 128,
    )
    source_x0, source_y0 = max(0, math.floor(xmin) - 1), max(0, math.floor(ymin) - 1)
    source_x1, source_y1 = min(reader.width, math.ceil(xmax) + 2), min(reader.height, math.ceil(ymax) + 2)
    rgb = read_rgb_region(reader, x_start=source_x0, y_start=source_y0,
                          width=source_x1 - source_x0, height=source_y1 - source_y0)
    px, py = np.clip(x[valid], 0, reader.width - 1), np.clip(y[valid], 0, reader.height - 1)
    x0, y0 = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, reader.width - 1), np.minimum(y0 + 1, reader.height - 1)
    if (x0.min() < source_x0 or x1.max() >= source_x1 or y0.min() < source_y0 or y1.max() >= source_y1):
        raise ValueError("registered RGB crop does not contain bilinear support")
    fx, fy = (px - x0)[:, None], (py - y0)[:, None]
    output[valid] = (
        rgb[y0 - source_y0, x0 - source_x0] * (1 - fx) * (1 - fy)
        + rgb[y0 - source_y0, x1 - source_x0] * fx * (1 - fy)
        + rgb[y1 - source_y0, x0 - source_x0] * (1 - fx) * fy
        + rgb[y1 - source_y0, x1 - source_x0] * fx * fy
    ) / 255.0
    return output, valid
