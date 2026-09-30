"""Native TIFF readers, registered RGB sampling, and bounded OD/H/E integration."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import math
import numpy as np
from PIL import Image

class TiledRgbBtfReader:
    def __init__(self, path: str | Path, cache_tiles: int = 64):
        self.path = Path(path)
        self.cache_tiles = int(cache_tiles)
        if self.cache_tiles <= 0:
            raise ValueError("cache_tiles must be positive")
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(self.path) as image:
            tags = image.tag_v2
            self.width = int(tags[256])
            self.height = int(tags[257])
            self.bits_per_sample = tuple(int(value) for value in tags[258])
            self.compression = int(tags[259])
            self.photometric = int(tags[262])
            self.orientation = int(tags.get(274, 1))
            self.samples_per_pixel = int(tags[277])
            self.planar_configuration = int(tags[284])
            self.tile_width = int(tags[322])
            self.tile_height = int(tags[323])
            self.tile_offsets = tuple(int(value) for value in tags[324])
            self.tile_byte_counts = tuple(int(value) for value in tags[325])
        if self.compression != 1:
            raise ValueError("only uncompressed TIFF tiles are supported")
        if self.photometric != 2 or self.samples_per_pixel != 3:
            raise ValueError("only RGB TIFF images are supported")
        if self.bits_per_sample != (8, 8, 8) or self.planar_configuration != 1:
            raise ValueError("only chunky 8-bit RGB TIFF images are supported")
        if self.orientation != 1:
            raise ValueError("only top-left TIFF orientation is supported")
        self.tile_columns = (self.width + self.tile_width - 1) // self.tile_width
        self.tile_rows = (self.height + self.tile_height - 1) // self.tile_height
        if len(self.tile_offsets) != self.tile_rows * self.tile_columns:
            raise ValueError("TIFF tile table length does not match image dimensions")
        expected_bytes = self.tile_height * self.tile_width * self.samples_per_pixel
        if any(value != expected_bytes for value in self.tile_byte_counts):
            raise ValueError("variable or packed tile byte counts are not supported")
        self._file = None
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __enter__(self) -> "TiledRgbBtfReader":
        self._file = self.path.open("rb")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        self._cache.clear()

    def read_tile(self, tile_index: int) -> np.ndarray:
        if self._file is None:
            raise RuntimeError("TiledRgbBtfReader must be used as a context manager")
        tile_index = int(tile_index)
        if tile_index < 0 or tile_index >= len(self.tile_offsets):
            raise IndexError("tile index outside image")
        if tile_index in self._cache:
            tile = self._cache.pop(tile_index)
            self._cache[tile_index] = tile
            return tile
        self._file.seek(self.tile_offsets[tile_index])
        payload = self._file.read(self.tile_byte_counts[tile_index])
        if len(payload) != self.tile_byte_counts[tile_index]:
            raise IOError("short read while loading TIFF tile")
        tile = np.frombuffer(payload, dtype=np.uint8).reshape(
            self.tile_height, self.tile_width, self.samples_per_pixel
        )
        self._cache[tile_index] = tile
        while len(self._cache) > self.cache_tiles:
            self._cache.popitem(last=False)
        return tile


class StripRgbTiffReader:
    """Read uncompressed chunky RGB strips without materializing a whole slide."""

    def __init__(self, path: str | Path, cache_strips: int = 8) -> None:
        self.path = Path(path)
        self.cache_strips = int(cache_strips)
        if self.cache_strips <= 0:
            raise ValueError("cache_strips must be positive")
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(self.path) as image:
            tags = image.tag_v2
            self.width = int(tags[256])
            self.height = int(tags[257])
            self.bits_per_sample = tuple(int(value) for value in tags[258])
            self.compression = int(tags[259])
            self.photometric = int(tags[262])
            self.orientation = int(tags.get(274, 1))
            self.samples_per_pixel = int(tags[277])
            self.rows_per_strip = int(tags[278])
            self.strip_offsets = tuple(int(value) for value in tags[273])
            self.strip_byte_counts = tuple(int(value) for value in tags[279])
            self.planar_configuration = int(tags[284])
        if not (
            self.bits_per_sample == (8, 8, 8)
            and self.compression == 1
            and self.photometric == 2
            and self.orientation == 1
            and self.samples_per_pixel == 3
            and self.planar_configuration == 1
            and self.rows_per_strip > 0
        ):
            raise ValueError("v030 only supports uncompressed top-left chunky RGB strips")
        expected_strips = (self.height + self.rows_per_strip - 1) // self.rows_per_strip
        if len(self.strip_offsets) != expected_strips or len(self.strip_byte_counts) != expected_strips:
            raise ValueError("v030 TIFF strip table length differs")
        self._file = None
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __enter__(self) -> "StripRgbTiffReader":
        self._file = self.path.open("rb")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        self._cache.clear()

    def read_strip(self, index: int) -> np.ndarray:
        if self._file is None:
            raise RuntimeError("StripRgbTiffReader must be used as a context manager")
        index = int(index)
        if index in self._cache:
            value = self._cache.pop(index)
            self._cache[index] = value
            return value
        if index < 0 or index >= len(self.strip_offsets):
            raise IndexError("TIFF strip index outside image")
        rows = min(self.rows_per_strip, self.height - index * self.rows_per_strip)
        row_bytes = self.width * self.samples_per_pixel
        stored_bytes = self.strip_byte_counts[index]
        if stored_bytes % row_bytes or stored_bytes // row_bytes not in {
            rows,
            self.rows_per_strip,
        }:
            raise ValueError("v030 TIFF strip byte count differs")
        self._file.seek(self.strip_offsets[index])
        payload = self._file.read(stored_bytes)
        if len(payload) != stored_bytes:
            raise IOError("short read while loading TIFF strip")
        value = np.frombuffer(payload, dtype=np.uint8).reshape(
            stored_bytes // row_bytes, self.width, 3
        )[:rows]
        self._cache[index] = value
        while len(self._cache) > self.cache_strips:
            self._cache.popitem(last=False)
        return value


class _AreaStripReader(StripRgbTiffReader):
    """Expose strip TIFFs through the small tile interface used by v008."""

    @property
    def tile_width(self) -> int:
        return self.width

    @property
    def tile_height(self) -> int:
        return self.rows_per_strip

    @property
    def tile_columns(self) -> int:
        return 1

    def read_tile(self, index: int) -> np.ndarray:
        return self.read_strip(index)


_STAIN_MATRIX = np.asarray(
    [
        [0.650, 0.072, 0.268],
        [0.704, 0.990, 0.570],
        [0.286, 0.105, 0.776],
    ],
    dtype=np.float64,
)
_STAIN_MATRIX /= np.linalg.norm(_STAIN_MATRIX, axis=0, keepdims=True)
_STAIN_INVERSE = np.linalg.inv(_STAIN_MATRIX)


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


DENSITY_CHANNELS = ("rgb_optical_density", "hematoxylin", "eosin")
VALID_FRACTION_SCALE = np.iinfo(np.uint16).max


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


class SourceDensityCache:
    """Per-reader LRU of pixelwise OD/H/E after a fixed native RGB transform.

    Cache only within one preparation run. Blocks use source-image coordinates,
    so overlapping or rotated FOVs reuse identical pixels before interpolation.
    The byte limit covers retained density arrays, excluding region/work arrays.
    """

    def __init__(self, reader, *, rgb_transform=None, block_size=256, max_bytes=512 * 2**20):
        self.reader = reader
        self.rgb_transform = rgb_transform
        self.block_size, self.max_bytes = int(block_size), int(max_bytes)
        if self.block_size <= 0 or self.max_bytes < self.block_size**2 * 3 * 4:
            raise ValueError('density cache must hold at least one full float32 block')
        self._cache = OrderedDict()
        self.retained_bytes = self.peak_bytes = 0
        self.hits = self.misses = self.transformed_pixels = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cache.clear()
        self.retained_bytes = 0

    def _block(self, by, bx):
        key = (by, bx)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        size = self.block_size
        x, y = bx * size, by * size
        width = min(size, self.reader.width - x)
        height = min(size, self.reader.height - y)
        needed = width * height * 3 * 4
        while self.retained_bytes + needed > self.max_bytes:
            _, oldest = self._cache.popitem(last=False)
            self.retained_bytes -= sum(v.nbytes for v in oldest.values())
        rgb = read_rgb_region(self.reader, x_start=x, y_start=y, width=width, height=height)
        if self.rgb_transform is not None:
            rgb = self.rgb_transform(rgb)
        density = rgb_density_maps(rgb)
        self._cache[key] = density
        self.retained_bytes += needed
        self.peak_bytes = max(self.peak_bytes, self.retained_bytes)
        self.misses += 1
        self.transformed_pixels += width * height
        return density

    def read_region(self, *, x_start, y_start, width, height):
        if (x_start < 0 or y_start < 0 or width <= 0 or height <= 0
                or x_start + width > self.reader.width or y_start + height > self.reader.height):
            raise ValueError('density cache region must lie inside the source image')
        output = {name: np.empty((height, width), dtype=np.float32) for name in DENSITY_CHANNELS}
        size = self.block_size
        for by in range(y_start // size, (y_start + height - 1) // size + 1):
            for bx in range(x_start // size, (x_start + width - 1) // size + 1):
                block = self._block(by, bx)
                x0, y0 = max(x_start, bx * size), max(y_start, by * size)
                x1, y1 = min(x_start + width, (bx + 1) * size), min(y_start + height, (by + 1) * size)
                src = np.s_[y0 - by * size:y1 - by * size, x0 - bx * size:x1 - bx * size]
                dst = np.s_[y0 - y_start:y1 - y_start, x0 - x_start:x1 - x_start]
                for name in DENSITY_CHANNELS:
                    output[name][dst] = block[name][src]
        return output

    def statistics(self):
        return dict(block_size=self.block_size, max_bytes=self.max_bytes, peak_bytes=self.peak_bytes,
                    hits=self.hits, misses=self.misses, transformed_pixels=self.transformed_pixels)


def aggregate_density_region(
    reader,
    transform: np.ndarray,
    *,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
    samples_per_cell_axis: int,
    rgb_transform=None,
    density_cache=None,
    integration_rows=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate OD/H/E; optional row blocks preserve each cell's original reduction."""

    if density_cache is not None and (density_cache.reader is not reader or rgb_transform is not None):
        raise ValueError('use the density cache with its own reader and fixed RGB transform')
    k = int(samples_per_cell_axis)
    if (
        k <= 0
        or row_start < 0
        or column_start < 0
        or row_stop <= row_start
        or column_stop <= column_start
    ):
        raise ValueError("area-integration region differs")
    if integration_rows is not None and (
        isinstance(integration_rows, bool) or not isinstance(integration_rows, (int, np.integer))
        or integration_rows <= 0
    ):
        raise ValueError('integration_rows must be a positive integer or None')
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
    region = dict(x_start=source_x0, y_start=source_y0,
                  width=source_x1 - source_x0, height=source_y1 - source_y0)
    if density_cache is not None:
        source_density = density_cache.read_region(**region)
    else:
        rgb = read_rgb_region(reader, **region)
        if rgb_transform is not None:
            rgb = rgb_transform(rgb)
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
    density = np.zeros((output_rows, output_columns, 3), dtype=np.float32)
    coverage = np.empty((output_rows, output_columns), dtype=np.float64)
    rows = output_rows if integration_rows is None else int(integration_rows)
    # Slice the original coordinates: rebuilding coordinates from a block origin
    # can change floating-point rounding, especially for non-binary sampling steps.
    for begin in range(0, output_rows, rows):
        end = min(begin + rows, output_rows)
        grid_rows = sample_rows[begin * k:end * k, None]
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
            raise ValueError("BTF source crop does not cover interpolation support")
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
        chunk_coverage = inside.reshape(end - begin, k, output_columns, k).mean(
            axis=(1, 3), dtype=np.float64
        )
        coverage[begin:end] = chunk_coverage
        for channel_index, channel in enumerate(DENSITY_CHANNELS):
            source = source_density[channel]
            sampled = (
                source[local_y0, local_x0] * w00
                + source[local_y0, local_x1] * w01
                + source[local_y1, local_x0] * w10
                + source[local_y1, local_x1] * w11
            )
            sampled[~inside] = 0.0
            numerator = sampled.reshape(end - begin, k, output_columns, k).mean(
                axis=(1, 3), dtype=np.float64
            )
            np.divide(numerator, chunk_coverage, out=density[begin:end, :, channel_index],
                      where=chunk_coverage > 0)
    if not np.isfinite(density).all() or not np.isfinite(coverage).all():
        raise ValueError("area-integrated image contains non-finite values")
    return density, coverage.astype(np.float32, copy=False)


def quantize_valid_fraction(values: np.ndarray) -> np.ndarray:
    fractions = np.asarray(values, dtype=np.float64)
    if not np.isfinite(fractions).all() or np.any(fractions < 0) or np.any(fractions > 1):
        raise ValueError("valid-area fraction differs")
    return np.rint(fractions * VALID_FRACTION_SCALE).astype(np.uint16)


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
