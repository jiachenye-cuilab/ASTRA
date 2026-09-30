"""Geometry-preserving RGB decoding with bounded TIFF segments and small rasters."""
from collections import OrderedDict
from pathlib import Path
import warnings

import numpy as np
from PIL import Image

MAX_DECODE_BYTES = 64 * 2**20


def registered_transform(config):
    if 'spot_to_image' not in config:
        raise ValueError('missing physical calibration: supply spot_to_image from 2um capture centers to image pixels; dimensions and DPI are insufficient')
    matrix = np.asarray(config['spot_to_image'], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('spot_to_image must be a finite invertible calibrated 3x3 matrix')
    return matrix


class TiffRgbReader:
    """Decode only requested segments; uncompressed data retain the direct byte path."""
    def __init__(self, path, *, max_cache_bytes=MAX_DECODE_BYTES):
        try:
            import tifffile
        except ImportError as error:
            raise ImportError('TIFF reading requires the section extra (tifffile); no whole-image fallback is used') from error
        self.path = Path(path)
        self.max_cache_bytes = int(max_cache_bytes)
        if self.max_cache_bytes < 0:
            raise ValueError('max_cache_bytes must be nonnegative')
        with tifffile.TiffFile(self.path) as image:
            if len(image.pages) != 1:
                raise ValueError('multipage TIFF is unsupported; supply a registered single base IFD (subIFD pyramids use native level 0 only)')
            page = image.pages[0]
            orientation = page.tags.get(274)
            if orientation is not None and int(orientation.value) != 1:
                raise ValueError('only top-left orientation=1 is supported; image and coordinates must not be rotated independently')
            if page.dtype != np.uint8 or page.bitspersample != 8:
                raise ValueError('only uint8, 8-bit TIFF is supported; no bit-depth conversion')
            if int(page.photometric) != 2 or page.samplesperpixel != 3 or int(page.planarconfig) != 1 or page.shaped[:2] != (1, 1):
                raise ValueError('only 2D chunky RGB TIFF with three channels and no alpha is supported')
            if 34675 in page.tags:
                raise ValueError('embedded TIFF ICC profiles are unsupported; no implicit color conversion')
            self.width, self.height = int(page.imagewidth), int(page.imagelength)
            self.tile_width = int(page.tilewidth) if page.is_tiled else self.width
            self.tile_height = int(page.tilelength) if page.is_tiled else min(int(page.rowsperstrip), self.height)
            self.tile_columns = (self.width + self.tile_width - 1) // self.tile_width
            self.tile_rows = (self.height + self.tile_height - 1) // self.tile_height
            if self.tile_width * self.tile_height * 3 > MAX_DECODE_BYTES:
                raise ValueError('TIFF segment exceeds 64 MiB decoded RGB limit; use lossless tiled/smaller-strip TIFF with unchanged geometry')
            self.offsets, self.byte_counts = tuple(page.dataoffsets), tuple(page.databytecounts)
            if len(self.offsets) != self.tile_rows * self.tile_columns or any(n <= 0 or n > MAX_DECODE_BYTES for n in self.byte_counts):
                raise ValueError('invalid or oversized TIFF segment table')
            self.compression = int(page.compression)
            self.direct = self.compression == 1 and int(page.predictor) == 1
            try:
                self.decode = None if self.direct else page.decode
                self.jpeg_tables = page.jpegtables
            except (ImportError, ValueError) as error:
                raise ValueError(f'TIFF compression {self.compression} backend unavailable: {error}; some codecs require optional imagecodecs') from error
        self._file = None
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self.decoded_segments = 0

    def __enter__(self):
        self._file = self.path.open('rb')
        return self

    def __exit__(self, *_):
        if self._file is not None:
            self._file.close()
            self._file = None
        self._cache.clear()
        self._cache_bytes = 0

    def read_tile(self, index):
        if self._file is None:
            raise RuntimeError('reader must be used as a context manager')
        if not 0 <= index < len(self.offsets):
            raise IndexError('TIFF segment outside image')
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        self._file.seek(self.offsets[index])
        payload = self._file.read(self.byte_counts[index])
        if len(payload) != self.byte_counts[index]:
            raise IOError('short read while loading TIFF segment')
        if self.direct:
            row_bytes = self.tile_width * 3
            if len(payload) % row_bytes or len(payload) // row_bytes > self.tile_height:
                raise ValueError('uncompressed TIFF segment byte count differs')
            tile = np.frombuffer(payload, np.uint8).reshape(-1, self.tile_width, 3)
        else:
            try:
                decoded, _, _ = self.decode(payload, index, jpegtables=self.jpeg_tables)
            except (ImportError, ValueError, NotImplementedError) as error:
                raise ValueError(f'cannot decode TIFF compression {self.compression}: {error}; optional imagecodecs may be required; no full-image fallback') from error
            if decoded is None or decoded.shape[0] != 1:
                raise ValueError('unsupported TIFF segment geometry')
            tile = decoded[0]
        rows = min(self.tile_height, self.height - index // self.tile_columns * self.tile_height)
        if tile.dtype != np.uint8 or tile.ndim != 3 or tile.shape[1:] != (self.tile_width, 3) or tile.shape[0] < rows or tile.nbytes > MAX_DECODE_BYTES:
            raise ValueError('decoded TIFF segment dtype, channels or shape differs')
        self.decoded_segments += 1
        while self._cache and self._cache_bytes + tile.nbytes > self.max_cache_bytes:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes
        if tile.nbytes <= self.max_cache_bytes:
            self._cache[index] = tile
            self._cache_bytes += tile.nbytes
        return tile


class SmallRgbReader:
    """Pillow PNG/JPEG decoder, capped before load; never a WSI fallback."""
    def __init__(self, path):
        self.path = Path(path)
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(self.path) as image:
                if image.format not in ('PNG', 'JPEG'):
                    raise ValueError('only TIFF/BTF, PNG and JPEG are supported')
                self.width, self.height = image.size
                if self.width * self.height * 3 > MAX_DECODE_BYTES:
                    raise ValueError('PNG/JPEG exceeds 64 MiB decoded RGB limit; use registered lossless tiled TIFF; whole-slide fallback is disabled')
                if image.mode != 'RGB' or getattr(image, 'bits', 8) != 8:
                    raise ValueError('only 8-bit three-channel RGB PNG/JPEG is supported; no alpha, grayscale, palette or CMYK conversion')
                if image.getexif().get(274, 1) != 1:
                    raise ValueError('only top-left orientation=1 is supported; no automatic EXIF transpose')
                if image.info.get('icc_profile'):
                    raise ValueError('embedded ICC profiles are unsupported; no implicit color conversion')
                if image.format == 'PNG':
                    with self.path.open('rb') as handle:
                        header = handle.read(29)
                    if header[24:26] != bytes((8, 2)):
                        raise ValueError('only 8-bit truecolor RGB PNG is supported; no silent bit-depth conversion')
                    if 'transparency' in image.info:
                        raise ValueError('PNG transparency is unsupported; no alpha compositing')
                    if 'gamma' in image.info and not np.isclose(image.info['gamma'], 0.45455, atol=1e-5):
                        raise ValueError('non-sRGB PNG gamma is unsupported')
                    if 'chromaticity' in image.info and not np.allclose(image.info['chromaticity'], (.3127,.329,.64,.33,.3,.6,.15,.06), atol=1e-5):
                        raise ValueError('non-sRGB PNG chromaticity is unsupported')
                image.load()
                self._pixels = np.array(image, copy=True)
        self.tile_width, self.tile_height, self.tile_columns = self.width, self.height, 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._pixels = None

    def read_tile(self, index):
        if self._pixels is None:
            raise RuntimeError('reader is closed')
        if index != 0:
            raise IndexError('raster segment outside image')
        return self._pixels


def open_rgb_image(path):
    with Path(path).open('rb') as handle:
        signature = handle.read(4)
    if signature in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+'):
        return TiffRgbReader(path)
    return SmallRgbReader(path)
