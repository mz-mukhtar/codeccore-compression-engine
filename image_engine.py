"""
image_engine.py — Lossy & Lossless Image Processing Pipeline
CodecCore: Multi-Codec Compression Laboratory — Phase 5

Pipeline overview
─────────────────

  Encode
  ──────
    1. Load image via Pillow → RGB numpy array (H × W × 3, uint8).
    2. [Optional, lossy] Grayscale conversion:
         Y = 0.299·R  + 0.587·G  + 0.114·B
       Result: 2-D luminance array (H × W, uint8).
    3. [Optional, lossy] Spatial quantization:
         shift = 8 − quantize_bits
         pixel = (pixel >> shift) << shift
       Drops the lowest-significance bits, reducing unique pixel counts
       and creating longer runs for RLE (introduces intentional colour banding).
    4. Flatten the pixel array → 1-D uint8 sequence.
    5. Run-length encode (RLE) the flat sequence with BitWriter packing.
       Each run: 8 bits for the pixel value + 16 bits for the run count.
    6. Assemble self-describing payload (metadata block + RLE block).
    7. Return (payload_bytes, 0).  The RLE block embeds its own internal
       bit-padding byte so the external padding is always 0.

  Decode
  ──────
    1. Read metadata block: H, W, channels, quantize_bits (10 bytes).
    2. Read RLE block (self-describing: embeds padding at byte [8]).
    3. Reconstruct the flat pixel array via RLE decode.
    4. Reshape to (H, W) or (H, W, 3) and save as PNG via Pillow.
    5. Return PNG bytes.

Payload layout (written after the BinaryHeader in the .abc archive)
─────────────────────────────────────────────────────────────────────
  Metadata block — 10 bytes (byte-aligned):
    [0–3]   H                  (uint32 BE) — image height in pixels
    [4–7]   W                  (uint32 BE) — image width in pixels
    [8]     channels           (uint8)     — 1 = grayscale, 3 = RGB
    [9]     quantize_bits      (uint8)     — effective bit-depth 1–8

  RLE block — variable length (self-describing):
    [0–3]   total_pixels       (uint32 BE) — H × W × channels
    [4–7]   n_runs             (uint32 BE) — number of (value, count) runs
    [8]     internal_padding   (uint8, 0–7) — trailing zero bits in last byte
                                              of the bit-packed run stream
    [9…]    bit-packed runs    — per run: 8 bits (value) + 16 bits (count)

Total minimum: 10 + 9 = 19 bytes (for an image with 0 pixels or 0 runs).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from bit_io import BitWriter, BitReader


# ─────────────────────────────────────────────────────────────────────────────
# Public constants
# ─────────────────────────────────────────────────────────────────────────────

#: File extensions automatically recognised as images by the app UI.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    "jpg", "jpeg", "png", "bmp", "gif",
    "tiff", "tif", "webp", "avif",
})

#: Maximum run count representable in 16 bits.
_MAX_RUN: int = 65535


# ─────────────────────────────────────────────────────────────────────────────
# RLE helpers  (private)
# ─────────────────────────────────────────────────────────────────────────────

def _rle_encode(flat: np.ndarray) -> bytes:
    """Run-length encode a 1-D array of uint8 pixel values.

    Scans *flat* sequentially, grouping consecutive identical values into
    ``(pixel_value, count)`` pairs.  Each pair is packed by ``BitWriter``
    as 8 bits for the value and 16 bits for the count (max 65 535 per run).

    Payload layout (self-describing RLE block):
    ::

        [0–3]   total_pixels     (uint32 BE)
        [4–7]   n_runs           (uint32 BE)
        [8]     internal_padding (uint8, 0–7)
        [9…]    bit-packed runs  — per run: 8-bit value + 16-bit count

    Parameters
    ----------
    flat:
        1-D numpy array, dtype uint8.  May be empty.

    Returns
    -------
    bytes
        Self-describing RLE block.  Outer padding is always 0 (embedded).
    """
    n: int = len(flat)

    if n == 0:
        # 4-byte pixel count + 4-byte run count + 1-byte padding = 9 bytes
        return (0).to_bytes(4, "big") + (0).to_bytes(4, "big") + b"\x00"

    # ── Build the run list ────────────────────────────────────────────────────
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        val   = int(flat[i])
        count = 1
        # Coalesce identical consecutive pixels; cap at _MAX_RUN per run.
        while i + count < n and int(flat[i + count]) == val and count < _MAX_RUN:
            count += 1
        runs.append((val, count))
        i += count

    # ── Bit-pack each (value, count) pair ─────────────────────────────────────
    writer = BitWriter()
    for val, count in runs:
        writer.write_bits(format(val,   "08b"))   #  8 bits: pixel value 0–255
        writer.write_bits(format(count, "016b"))  # 16 bits: run length 1–65535
    packed, internal_padding = writer.flush()

    # ── Assemble self-describing block ────────────────────────────────────────
    header = bytearray()
    header += n.to_bytes(4, "big")            # total_pixels
    header += len(runs).to_bytes(4, "big")    # n_runs
    header += bytes([internal_padding])       # embedded bit-padding for decoder

    return bytes(header) + packed


def _rle_decode(rle_block: bytes) -> np.ndarray:
    """Decode an RLE block produced by :func:`_rle_encode`.

    Reads the self-describing header (total_pixels, n_runs, internal_padding)
    and then reconstructs the flat pixel array from the bit-packed run stream.

    Parameters
    ----------
    rle_block:
        Bytes produced by :func:`_rle_encode`.

    Returns
    -------
    numpy.ndarray
        1-D uint8 array of length *total_pixels*.

    Raises
    ------
    ValueError
        If the block is too short to contain the 9-byte header.
    """
    if len(rle_block) < 9:
        raise ValueError(
            f"RLE block too short: need ≥9 bytes for the header, "
            f"got {len(rle_block)}."
        )

    total_pixels     = int.from_bytes(rle_block[0:4], "big")
    n_runs           = int.from_bytes(rle_block[4:8], "big")
    internal_padding = rle_block[8]
    bit_data         = rle_block[9:]

    result = np.zeros(total_pixels, dtype=np.uint8)

    if n_runs == 0:
        return result

    # Materialise all logical bits into a flat list for indexed access.
    reader = BitReader(bit_data, internal_padding)
    bits   = list(reader.read_bits())
    pos    = 0

    def _rbits(n: int) -> int:
        """Read *n* bits from the pre-materialised list."""
        nonlocal pos
        v = 0
        for _ in range(n):
            v = (v << 1) | (bits[pos] if pos < len(bits) else 0)
            pos += 1
        return v

    pix_pos = 0
    for _ in range(n_runs):
        val   = _rbits(8)
        count = _rbits(16)
        end   = min(pix_pos + count, total_pixels)
        result[pix_pos:end] = val
        pix_pos += count

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def encode_image(
    image_bytes:   bytes,
    apply_gray:    bool = False,
    quantize_bits: int  = 8,
) -> tuple[bytes, int]:
    """Compress image bytes through the Phase 4 lossy→RLE pipeline.

    Pipeline
    --------
    1. Load via Pillow; normalise to RGB uint8 (H × W × 3).
    2. Grayscale conversion (optional, lossy):
       ``Y = 0.299·R + 0.587·G + 0.114·B``  →  2-D uint8 array.
    3. Spatial quantization (optional, lossy):
       Drop ``8 − quantize_bits`` least-significant bits then restore scale.
       Quantize to ``2^quantize_bits`` discrete levels.  Increases run lengths
       dramatically, boosting RLE efficiency.
    4. Flatten → RLE encode with ``BitWriter`` packing.
    5. Prepend 10-byte metadata block.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the source image file (any format Pillow can open).
    apply_gray:
        ``True``  → convert to single-channel luminance (reduces pixel count
        by 3× for RGB inputs, significant size reduction).
        ``False`` → keep all three RGB channels.
    quantize_bits:
        Effective bit-depth per channel (1–8).  ``8`` = lossless pass-through.
        ``4`` = 16 discrete levels per channel (heavy colour banding, high
        RLE gain).  ``1`` = binary (black/white per channel).

    Returns
    -------
    (payload_bytes, 0)
        *payload_bytes* — fully self-describing; embed in the .abc archive
        immediately after the serialised ``BinaryHeader``.
        The second element is always ``0`` (external padding bits).

    Raises
    ------
    Exception
        Propagates any ``PIL.UnidentifiedImageError`` or numpy shape error
        when *image_bytes* cannot be parsed as a valid image.
    """
    quantize_bits = max(1, min(8, int(quantize_bits)))

    # ── Step 1: Load and normalise to RGB ─────────────────────────────────────
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.uint8)     # shape (H, W, 3)
    H, W = arr.shape[:2]

    # ── Step 2: Grayscale conversion (lossy) ──────────────────────────────────
    if apply_gray:
        # Use float32 to avoid uint8 intermediate overflow during weighted sum.
        arr_f = arr.astype(np.float32)
        gray  = (
            0.299 * arr_f[:, :, 0]
            + 0.587 * arr_f[:, :, 1]
            + 0.114 * arr_f[:, :, 2]
        )
        arr      = np.clip(gray, 0.0, 255.0).astype(np.uint8)  # (H, W)
        channels = 1
    else:
        channels = 3

    # ── Step 3: Spatial quantization (lossy) ──────────────────────────────────
    if quantize_bits < 8:
        shift = 8 - quantize_bits
        # Compute (arr >> shift) << shift in uint16 to avoid left-shift overflow
        # at the uint8 boundary, then cast back to uint8.
        arr = (
            (arr.astype(np.uint16) >> shift) << shift
        ).astype(np.uint8)

    # ── Step 4: RLE compress the flattened pixel array ────────────────────────
    flat     = arr.flatten()                         # 1-D uint8
    rle_block = _rle_encode(flat)

    # ── Step 5: Assemble final payload ────────────────────────────────────────
    # Metadata block — 10 bytes:
    #   [0–3]  H                (uint32 BE)
    #   [4–7]  W                (uint32 BE)
    #   [8]    channels         (uint8: 1 or 3)
    #   [9]    quantize_bits    (uint8: 1–8)
    meta = bytearray()
    meta += H.to_bytes(4, "big")
    meta += W.to_bytes(4, "big")
    meta += bytes([channels & 0xFF])
    meta += bytes([quantize_bits & 0xFF])

    return bytes(meta) + rle_block, 0
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # External padding is always 0; the RLE block self-describes its own
    # internal bit-padding at byte [8] of the RLE block (offset [18] total).


def decode_image(payload: bytes, padding_bits: int = 0) -> bytes:
    """Reconstruct the image from a Phase 4 RLE-compressed payload.

    Parameters
    ----------
    payload:
        Bytes produced by :func:`encode_image` (metadata block + RLE block).
    padding_bits:
        Ignored — the RLE block is self-describing (internal padding stored at
        RLE block byte [8]).  Present only for API symmetry.

    Returns
    -------
    bytes
        PNG-encoded image bytes ready for download or display in Streamlit.

    Raises
    ------
    ValueError
        If the payload is shorter than 19 bytes (minimum valid size), or if
        the channel count is not 1 or 3, or if the flat array cannot be
        reshaped into the dimensions declared in the metadata.
    """
    _METADATA_LEN = 10

    if len(payload) < _METADATA_LEN:
        raise ValueError(
            f"Image payload too short: need ≥{_METADATA_LEN} bytes for "
            f"the metadata block, got {len(payload)}."
        )

    # ── Read metadata block ───────────────────────────────────────────────────
    H            = int.from_bytes(payload[0:4], "big")
    W            = int.from_bytes(payload[4:8], "big")
    channels     = payload[8]
    # quantize_bits = payload[9]  # informational only; not needed for decode

    if channels not in (1, 3):
        raise ValueError(
            f"Unsupported channel count in image payload: {channels} "
            f"(expected 1 for grayscale or 3 for RGB)."
        )

    # ── Decode RLE block ──────────────────────────────────────────────────────
    rle_block = payload[_METADATA_LEN:]
    flat      = _rle_decode(rle_block)

    # ── Reshape flat array → image ────────────────────────────────────────────
    expected_pixels = H * W * channels
    if len(flat) < expected_pixels:
        raise ValueError(
            f"RLE decoded {len(flat)} pixel values but the metadata "
            f"declares {H}×{W}×{channels}={expected_pixels}."
        )

    if channels == 1:
        arr = flat[:expected_pixels].reshape(H, W).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    else:  # channels == 3
        arr = flat[:expected_pixels].reshape(H, W, 3).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")

    # ── Encode output as PNG ──────────────────────────────────────────────────
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_image_info(payload: bytes) -> Optional[dict]:
    """Extract image metadata from a payload without decoding the full pixel data.

    Parameters
    ----------
    payload:
        Bytes produced by :func:`encode_image`.

    Returns
    -------
    dict or None
        Dict with keys ``height``, ``width``, ``channels``, ``quantize_bits``,
        and ``total_pixels``.  Returns ``None`` if the payload is too short.
    """
    if len(payload) < 10:
        return None

    H             = int.from_bytes(payload[0:4], "big")
    W             = int.from_bytes(payload[4:8], "big")
    channels      = payload[8]
    quantize_bits = payload[9]

    # RLE header starts at offset 10.
    rle_offset = 10
    total_pixels = (
        int.from_bytes(payload[rle_offset : rle_offset + 4], "big")
        if len(payload) >= rle_offset + 4
        else H * W * channels
    )
    n_runs = (
        int.from_bytes(payload[rle_offset + 4 : rle_offset + 8], "big")
        if len(payload) >= rle_offset + 8
        else 0
    )

    return {
        "height":        H,
        "width":         W,
        "channels":      channels,
        "quantize_bits": quantize_bits,
        "total_pixels":  total_pixels,
        "n_runs":        n_runs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING ANALYTICS  (Phase 5)
# ══════════════════════════════════════════════════════════════════════════════
#
# Mean Squared Error (MSE)
# ────────────────────────
#   Quantifies the average squared pixel-level deviation introduced by the
#   lossy pipeline (grayscale conversion and/or spatial quantization):
#
#       MSE = (1 / (H · W · C)) · ΣΣΣ [I_orig(i,j,c) − I_rest(i,j,c)]²
#
#   MSE = 0 when the pipeline is lossless (quant=8, no grayscale).
#   Units: squared intensity (0–255² = 0–65 025).
#
# Peak Signal-to-Noise Ratio (PSNR)
# ─────────────────────────────────
#   Expresses signal fidelity in decibels relative to the maximum possible
#   pixel intensity (MAX_I = 255 for 8-bit images):
#
#       PSNR = 10 · log₁₀(MAX_I² / MSE)   [dB]
#
#   Interpretation guide:
#     > 40 dB  Virtually indistinguishable from original
#     35–40 dB Excellent quality
#     30–35 dB Good quality, minor artifacts
#     25–30 dB Noticeable but acceptable
#     < 25 dB  Significant distortion visible
#
#   Special case: MSE = 0 → PSNR = ∞ (reported as the string "Infinity").
#   The comparison is performed in the same colour-space as the compressed
#   image; for grayscale archives, both original and reconstructed are
#   converted to luminance before the calculation.


@dataclass(frozen=True)
class ImageErrorMetrics:
    """Container for pixel-domain signal-quality metrics.

    All fields are pre-computed and immutable.  Pass an instance directly
    to the Streamlit display layer.

    Attributes
    ----------
    mse : float
        Mean Squared Error in squared-intensity units (0 – 65 025).
        Equals ``0.0`` for a lossless pipeline.
    psnr : float or None
        Peak Signal-to-Noise Ratio in decibels.  ``None`` encodes infinite
        PSNR (MSE = 0).  Use :attr:`psnr_str` for display.
    psnr_str : str
        Human-readable PSNR: either the decimal dB value (e.g. ``"38.47 dB"``)
        or the string ``"Infinity"`` when MSE = 0.
    compression_ratio : float
        Ratio of compressed-payload bytes to original raw image bytes
        (not the .abc archive — just the RLE payload vs. raw pixel bytes).
    n_pixels : int
        Total number of pixel×channel samples compared.
    is_lossless : bool
        ``True`` when MSE = 0 (pipeline was lossless).
    """

    mse:               float
    psnr:              Optional[float]
    psnr_str:          str
    compression_ratio: float
    n_pixels:          int
    is_lossless:       bool


def analyze_image_error(
    original_bytes:      bytes,
    reconstructed_bytes: bytes,
    width:               int,
    height:              int,
    is_grayscale:        bool,
    compressed_payload:  bytes = b"",
) -> ImageErrorMetrics:
    """Compute MSE, PSNR, and compression ratio for the lossy image pipeline.

    Both *original_bytes* and *reconstructed_bytes* are PNG byte strings as
    produced by Pillow.  The function converts them to numpy arrays internally
    so that no intermediate arrays need to be threaded through the call stack.

    For grayscale archives the original RGB image is converted to the same
    single-channel luminance space (Y = 0.299R + 0.587G + 0.114B) before the
    comparison, ensuring that the error measurement is in the same domain as
    the compressed representation.

    Parameters
    ----------
    original_bytes : bytes
        Raw bytes of the *original* source image file (any Pillow-supported
        format: PNG, JPEG, BMP, …).  This is the file that was uploaded
        *before* any lossy transformation.
    reconstructed_bytes : bytes
        PNG bytes returned by :func:`decode_image` for the same archive.
    width : int
        Image width in pixels (from :func:`get_image_info`).
    height : int
        Image height in pixels (from :func:`get_image_info`).
    is_grayscale : bool
        ``True`` if the archive stores a single luminance channel
        (``channels == 1`` in the payload metadata).
    compressed_payload : bytes, optional
        The RLE payload bytes (for computing the compression ratio).  If
        omitted or empty the ratio field is set to ``0.0``.

    Returns
    -------
    ImageErrorMetrics
        Frozen dataclass with MSE, PSNR, compression ratio, pixel count,
        and losslessness flag.

    Raises
    ------
    ValueError
        If the arrays cannot be broadcast to the same shape after cropping
        to the declared dimensions.
    """
    # ── Step 1: Decode both images to numpy arrays ────────────────────────────
    orig_img = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    rest_img = Image.open(io.BytesIO(reconstructed_bytes)).convert(
        "L" if is_grayscale else "RGB"
    )

    orig_arr = np.array(orig_img, dtype=np.float64)   # (H_src, W_src, 3)
    rest_arr = np.array(rest_img, dtype=np.float64)   # (H, W[, 1 or 3])

    # ── Step 2: Project original to the same colour-space as the archive ───────
    if is_grayscale:
        # Apply the same luminance formula used in encode_image() to the
        # original so we compare apples-to-apples (both in Y-space).
        orig_arr = (
            0.299 * orig_arr[:, :, 0]
            + 0.587 * orig_arr[:, :, 1]
            + 0.114 * orig_arr[:, :, 2]
        )   # shape (H_src, W_src) — float64

    # ── Step 3: Crop both arrays to the declared (H, W) ────────────────────
    # Pillow may return a slightly different size if the source image had
    # EXIF orientation data; crop to the metadata-declared size.
    orig_arr = orig_arr[:height, :width]   # crop rows then cols
    rest_arr = rest_arr[:height, :width]

    # ── Step 4: MSE ──────────────────────────────────────────────────────────
    diff       = orig_arr.astype(np.float64) - rest_arr.astype(np.float64)
    n_pixels   = int(diff.size)       # total pixel×channel samples
    mse        = float(np.sum(diff ** 2) / n_pixels) if n_pixels > 0 else 0.0

    # ── Step 5: PSNR ─────────────────────────────────────────────────────────
    MAX_I = 255.0
    if mse == 0.0:
        psnr     = None
        psnr_str = "Infinity"
    else:
        psnr     = 10.0 * math.log10((MAX_I ** 2) / mse)
        psnr_str = f"{psnr:.4f} dB"

    # ── Step 6: Compression ratio ───────────────────────────────────────────
    # raw_pixel_bytes: the uncompressed image data (H × W × channels × 1 byte).
    channels_n  = 1 if is_grayscale else 3
    raw_bytes_n = height * width * channels_n
    if raw_bytes_n > 0 and len(compressed_payload) > 0:
        compression_ratio = len(compressed_payload) / raw_bytes_n
    else:
        compression_ratio = 0.0

    return ImageErrorMetrics(
        mse               = mse,
        psnr              = psnr,
        psnr_str          = psnr_str,
        compression_ratio = compression_ratio,
        n_pixels          = n_pixels,
        is_lossless       = (mse == 0.0),
    )
