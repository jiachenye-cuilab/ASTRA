# Image input support

Raw `prepare-section` decoding uses [image_readers.py](../astra/data/image_readers.py)
and existing registered sampling/OD/H/E integration. Weights, scientific stain
processing, coordinates and UNI preprocessing do not change.

| Input | Decoder/access | Verification |
| --- | --- | --- |
| Uncompressed uint8 chunky RGB TIFF/BTF | Direct requested tile/strip bytes, retaining the fast path | Exact pixels/regions for tiles and strips |
| Deflate uint8 chunky RGB TIFF/BTF | `tifffile.TiffPage.decode`, requested segments only | Exact pixels/regions for tiles and strips |
| Other TIFF compression | Requires a working `tifffile` segment codec, sometimes optional `imagecodecs` | Not generally validated; missing-codec error tested |
| 8-bit truecolor RGB PNG | Pillow, capped small-image full decode | Lossless regions; 16-bit/alpha/palette rejection |
| 8-bit RGB JPEG | Pillow, capped small-image full decode | Equals Pillow decoded pixels; differs from source RGB |
| Multipage TIFF, CMYK/YCbCr TIFF, grayscale, 16-bit, alpha | Rejected | No implicit conversion |

Pillow is required by base requirements. TIFF needs the existing section
dependencies (`tifffile`, SciPy, h5py). No new production dependency is added;
optional TIFF codecs are not installed automatically. Backend/codec errors
identify missing support; no whole-image fallback is attempted.

Accept only **2D uint8, three-channel RGB**. No alpha compositing, channel or
bit-depth conversion, EXIF transpose, resize, crop or resampling occurs in
decoding. Orientation must be absent/1 (top-left). Embedded ICC profiles are
rejected; untagged RGB is the user's supplied display RGB. PNG gamma/chromaticity
must be absent or sRGB-compatible. JPEG's usual YCbCr-to-RGB decoder conversion
is part of JPEG decoding. A JPEG remains lossy even when decoded pixels are
read exactly.

TIFF uses native base IFD **level 0**. SubIFD pyramids may be present but are not
selected/downsampled. Multiple top-level pages/associated-image layouts are
rejected rather than silently selecting an image. Pyramid/vendor preprocessing
has not been validated end-to-end.
NDPI/SVS have no dedicated backend or validation here. A file's extension alone
does not establish support: TIFF-signature files must satisfy the strict RGB,
base-IFD and segment contract above; no general vendor WSI support is claimed.

## Memory bounds

TIFF never calls whole-page `asarray`. Encoded and decoded segments are each at
most **64 MiB**, with a 64 MiB default decoded-segment LRU cache. Larger segments
are rejected before decode; use lossless TIFF with smaller tiles/strips while
retaining pixel geometry/calibration. Reading memory includes the requested
region, one segment/encoded buffer and cache. Existing OD/H/E density caching
has its own 512 MiB limit.

PNG/JPEG are **not WSI region decoders**. `width*height*3 <= 64 MiB` is checked
before `Image.load`. The Pillow buffer and NumPy copy can together use roughly
twice that RGB size, plus codec workspace. Larger rasters are rejected. Region
reading preserves coordinates and explicit white padding.

The large synthetic TIFF represents 768 MiB decoded RGB using compressed
repeated tiles without allocating that array. The test forbids whole-page
`asarray`, checks that only two requested tiles decode, and checks cache bytes
and eviction. Oversized strips/PNG headers are rejected before load. This is
structural bounded-memory evidence, not a real-slide RSS benchmark.

## Calibration and input distinctions

Raw sections require an explicit physically calibrated `spot_to_image`; missing
calibration raises a clear error. Dimensions, generic DPI and TIFF resolution
tags never imply microns-per-pixel. The transform maps 2 µm capture pixel-center
indices `(col,row)` to image pixel centers and must include measured scale,
registration and origin. For aligned physical edges with isotropic `mpp`, use
`[[2/mpp,0,1/mpp-0.5],[0,2/mpp,1/mpp-0.5],[0,0,1]]`.
The identity template is valid only for aligned **2 µm/pixel** data.

No decode coordinate transform occurs; non-top-left orientation is rejected.
Existing homography sampling uses the same native convention for RGB, OD/H/E
and validity. Bilinear sampling/integration is existing scientific preprocessing,
distinct from decoding.

FOV NPZ arrays are already registered/processed numeric inputs. Cached UNI
features do not decode raw images. Decode tests do not validate new formats
through gated UNI weights or all downstream platforms. Raw base-training
preparation now uses the same bounded reader contract; this does not establish
end-to-end training validation for every newly decoded format.
