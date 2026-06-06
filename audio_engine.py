"""
audio_engine.py — Audio Waveform Coding Engine
CodecCore: Data Compression Engine — Phase 6

Pipeline overview
─────────────────
This module implements four classical digital audio coding schemes as described
in Jayant & Noll (1984) "Digital Coding of Waveforms" and Proakis & Manolakis
(2007) "Digital Signal Processing":

  1. PCM  — Uniform Pulse-Code Modulation.
  2. DM   — Delta Modulation (1-bit differential coding).
  3. ADM  — Adaptive Delta Modulation (Jayant's algorithm).
  4. DPCM — Differential PCM with first-order linear predictor.

All implementations operate on normalised float32 amplitude arrays in [-1, 1].
The interface to the Streamlit UI is purely through Python bytes objects; no
audio files are opened or closed inside this module.

Encoding contract
──────────────────
Every encoder returns:
    (payload_bytes: bytes, padding_bits: int)

The payload_bytes are the raw bit-packed stream.  padding_bits is the number
of trailing zero bits appended to the last byte to align it to a byte boundary;
this value must be stored in the archive header (AudioHeader.padding_bits) so
the decoder can recover the exact logical bit-stream length.

Decoding contract
──────────────────
Every decoder accepts:
    (payload_bytes: bytes, padding_bits: int, original_length: int, **params)

and returns:
    signal: np.ndarray  — float32 array in [-1, 1], length == original_length
                          (after upsampling / interpolation when sampling_factor > 1)

Internal representation
──────────────────────────
  • "signal"     — float32 array, values in [-1.0, +1.0] (normalised PCM).
  • "packed"     — bytes object from BitWriter.flush().
  • "staircase"  — float accumulator tracking the DM/ADM approximation.
  • "residual"   — float array of DPCM prediction errors.

All amplitude values are quantised to integer levels and bit-packed via the
BitWriter class from bit_io.py.  No external compression frameworks are used.
"""

from __future__ import annotations

import math
import struct
import io
from typing import Optional

import numpy as np

from bit_io import BitWriter, BitReader


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _uniform_quantize(
    value:    float,
    n_levels: int,
    v_min:    float,
    v_max:    float,
) -> int:
    """Map a scalar float value to an integer quantization index.

    Uses a mid-rise uniform quantiser with ``n_levels`` levels spanning
    the closed interval ``[v_min, v_max]``.

    Parameters
    ----------
    value:
        The amplitude sample to quantise.  Clamped to [v_min, v_max] internally.
    n_levels:
        Number of quantisation levels (e.g. 256 for 8-bit, 16 for 4-bit).
    v_min, v_max:
        The amplitude range to span with ``n_levels`` uniform steps.

    Returns
    -------
    int
        Integer index in [0, n_levels - 1].
    """
    value   = max(v_min, min(v_max, value))   # hard clip
    span    = v_max - v_min
    if span == 0.0:
        return 0
    # Normalise to [0, 1) then scale to [0, n_levels - 1].
    norm    = (value - v_min) / span
    # Clamp after scaling to avoid index == n_levels on value == v_max.
    idx     = int(norm * n_levels)
    return min(idx, n_levels - 1)


def _uniform_reconstruct(
    index:    int,
    n_levels: int,
    v_min:    float,
    v_max:    float,
) -> float:
    """Reconstruct a float amplitude from a uniform quantisation index.

    Returns the *mid-point* of the quantisation interval (mid-rise convention).

    Parameters
    ----------
    index:
        Integer in [0, n_levels - 1].
    n_levels, v_min, v_max:
        Same parameters used in :func:`_uniform_quantize`.

    Returns
    -------
    float
        Reconstructed amplitude.
    """
    span     = v_max - v_min
    step     = span / n_levels
    midpoint = v_min + (index + 0.5) * step
    return midpoint


def _write_fixed_width(writer: BitWriter, value: int, width: int) -> None:
    """Write a non-negative integer as a fixed-width MSB-first bit string.

    Parameters
    ----------
    writer : BitWriter
    value  : int   — non-negative integer.
    width  : int   — number of bits (1–32).
    """
    for shift in range(width - 1, -1, -1):
        writer.write_bit((value >> shift) & 1)


def _read_fixed_width(bits: list[int], pos: int, width: int) -> tuple[int, int]:
    """Read `width` bits from a pre-materialised bit list starting at `pos`.

    Returns
    -------
    (value, new_pos) — the reconstructed integer and the updated read position.
    """
    value = 0
    for _ in range(width):
        bit   = bits[pos] if pos < len(bits) else 0
        value = (value << 1) | bit
        pos  += 1
    return value, pos


def _materialize_bits(payload: bytes, padding_bits: int) -> list[int]:
    """Convert a packed byte payload to a flat list of logical bits."""
    reader = BitReader(payload, padding_bits)
    return list(reader.read_bits())


# ─────────────────────────────────────────────────────────────────────────────
# Waveform I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_audio_to_array(file_bytes: bytes, extension: str) -> tuple[int, np.ndarray]:
    """Load a multi-format audio file from raw bytes and return (sample_rate, signal).

    The signal is normalised to float32 in [-1.0, +1.0].
    Multi-channel signals are mixed down to mono.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of a valid audio file.
    extension : str
        The original file extension (e.g., 'wav', 'mp3', 'flac', 'm4a', 'ogg').

    Returns
    -------
    sample_rate : int
    signal      : np.ndarray  shape (n_samples,), float32

    Raises
    ------
    ImportError
        If pydub is not installed.
    Exception
        If pydub/ffmpeg cannot parse the bytes.
    """
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise ImportError(
            "pydub is required for audio loading (pip install pydub)."
        ) from exc

    fmt = extension.lower()
    if fmt == 'm4a':
        fmt = 'mp4'

    buf = io.BytesIO(file_bytes)
    audio = AudioSegment.from_file(buf, format=fmt)

    sample_rate = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())

    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))

    max_val = float(1 << (8 * audio.sample_width - 1))
    signal = samples.astype(np.float32) / max_val
    signal = np.clip(signal, -1.0, 1.0)

    # Always return mono array for processing
    return sample_rate, mix_to_mono(signal)


def export_array_to_format(signal: np.ndarray, sample_rate: int, extension: str) -> bytes:
    """Convert a float32 mono signal array to a specific audio format byte string.

    Parameters
    ----------
    signal      : np.ndarray  float32, values in [-1, 1], 1-D.
    sample_rate : int
    extension   : str         The target file extension.

    Returns
    -------
    bytes  — Valid audio file bytes suitable for st.audio() or download.
    """
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise ImportError("pydub is required (pip install pydub).") from exc

    # Clip, convert to int16.
    pcm = np.clip(signal, -1.0, 1.0)
    pcm_int16 = (pcm * 32767.0).astype(np.int16)

    audio = AudioSegment(
        pcm_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit is 2 bytes
        channels=1
    )

    fmt = extension.lower()
    if fmt == 'm4a':
        fmt = 'mp4'

    buf = io.BytesIO()
    audio.export(buf, format=fmt)
    return buf.getvalue()


def mix_to_mono(signal: np.ndarray) -> np.ndarray:
    """Average multi-channel audio to mono float32.

    Parameters
    ----------
    signal : np.ndarray
        Shape (n_samples,) for mono or (n_samples, n_channels) for multi-channel.

    Returns
    -------
    np.ndarray  shape (n_samples,), float32, values in [-1, 1].
    """
    if signal.ndim == 1:
        return signal.astype(np.float32)
    return signal.mean(axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# §1  PCM  —  Uniform Pulse-Code Modulation
# ─────────────────────────────────────────────────────────────────────────────
#
# PCM encodes each audio sample independently as a fixed-width integer.
# Steps:
#   1. Downsample by `sampling_factor` (keep every N-th sample).
#   2. Determine the global amplitude range [v_min, v_max] of the downsampled
#      signal so the decoder can reconstruct the correct scale.
#   3. Quantise each sample to one of 2^bits uniform levels.
#   4. Pack each quantised integer as a `bits`-wide MSB-first bit string.
#
# Payload layout (bytes):
#   [0–3]   v_min  as IEEE 754 float32 (4 bytes, big-endian)
#   [4–7]   v_max  as IEEE 754 float32 (4 bytes, big-endian)
#   [8…]    packed quantised samples — `bits` bits per sample
#
# After the 8-byte range header the bit-stream follows immediately; its
# padding_bits value is stored in AudioHeader.padding_bits.

def pcm_encode(
    signal:          np.ndarray,
    sampling_factor: int,
    bits:            int,
) -> tuple[bytes, int]:
    """Encode a float32 signal using uniform PCM quantisation.

    Parameters
    ----------
    signal : np.ndarray
        Float32 array, values in [-1.0, +1.0].  Must be 1-D (mono).
    sampling_factor : int
        Downsample ratio (1 = no downsample, 2 = keep every 2nd sample, etc.).
        Must be 1, 2, or 4.
    bits : int
        Quantisation bit-depth per sample (1–8).  Yields 2^bits levels.

    Returns
    -------
    (payload_bytes, padding_bits)
        payload_bytes — 8-byte range header + bit-packed quantised samples.
        padding_bits  — 0–7 trailing zero bits in the last byte.
    """
    if sampling_factor < 1:
        raise ValueError(f"sampling_factor must be ≥ 1, got {sampling_factor}.")
    if not (1 <= bits <= 8):
        raise ValueError(f"bits must be 1–8, got {bits}.")

    # ── Step 1: Downsample ────────────────────────────────────────────────────
    downsampled: np.ndarray = signal[::sampling_factor].astype(np.float32)

    # ── Step 2: Determine amplitude range ─────────────────────────────────────
    if len(downsampled) == 0:
        # Edge case: empty signal → return empty payload.
        v_min_f = v_max_f = 0.0
    else:
        v_min_f = float(downsampled.min())
        v_max_f = float(downsampled.max())
        if v_min_f == v_max_f:
            # Constant signal — avoid zero-span quantiser.
            v_min_f -= 1e-6
            v_max_f += 1e-6

    n_levels: int = 1 << bits   # 2^bits

    # ── Step 3 & 4: Quantise and bit-pack ─────────────────────────────────────
    writer = BitWriter()
    for sample in downsampled:
        idx = _uniform_quantize(float(sample), n_levels, v_min_f, v_max_f)
        _write_fixed_width(writer, idx, bits)

    packed, padding_bits = writer.flush()

    # ── Assemble payload: 8-byte range header + packed samples ────────────────
    range_header = struct.pack(">ff", v_min_f, v_max_f)   # 4+4 = 8 bytes, BE
    return range_header + packed, padding_bits


def pcm_decode(
    payload:         bytes,
    padding_bits:    int,
    bits:            int,
    original_length: int,
    sampling_factor: int,
) -> np.ndarray:
    """Decode a PCM payload back to a float32 signal array.

    Parameters
    ----------
    payload : bytes
        Bytes produced by :func:`pcm_encode` (8-byte header + packed samples).
    padding_bits : int
        Trailing padding bits in the last byte of the packed stream.
    bits : int
        Quantisation bit-depth used during encoding.
    original_length : int
        Number of samples in the *original* (pre-downsample) signal.
    sampling_factor : int
        Downsample factor used during encoding (used to allocate the output).

    Returns
    -------
    np.ndarray
        Float32 array of length ``original_length``, amplitude in [-1, 1].
        Upsampling (nearest-neighbour) is applied when sampling_factor > 1.
    """
    _HEADER = 8   # bytes for the float32 range header

    if len(payload) < _HEADER:
        raise ValueError(
            f"PCM payload too short: need ≥ {_HEADER} bytes, got {len(payload)}."
        )

    v_min_f, v_max_f = struct.unpack(">ff", payload[:_HEADER])
    packed_bytes = payload[_HEADER:]

    n_levels = 1 << bits

    # ── Reconstruct quantised samples from bit-stream ─────────────────────────
    bits_list = _materialize_bits(packed_bytes, padding_bits)
    pos       = 0
    decoded   = []
    while pos + bits <= len(bits_list):
        idx, pos = _read_fixed_width(bits_list, pos, bits)
        amp      = _uniform_reconstruct(idx, n_levels, v_min_f, v_max_f)
        decoded.append(amp)

    # ── Upsample to original_length (nearest-neighbour repeat) ───────────────
    if len(decoded) == 0:
        return np.zeros(original_length, dtype=np.float32)

    decoded_arr = np.array(decoded, dtype=np.float32)
    if sampling_factor == 1:
        # Trim or zero-pad to original_length.
        out = np.zeros(original_length, dtype=np.float32)
        n   = min(len(decoded_arr), original_length)
        out[:n] = decoded_arr[:n]
        return out

    # Nearest-neighbour upsampling: each decoded sample fills sampling_factor
    # output positions.
    upsampled = np.repeat(decoded_arr, sampling_factor)
    out       = np.zeros(original_length, dtype=np.float32)
    n         = min(len(upsampled), original_length)
    out[:n]   = upsampled[:n]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# §2  DM  —  Delta Modulation
# ─────────────────────────────────────────────────────────────────────────────
#
# Delta Modulation is a 1-bit differential coding scheme that tracks the input
# signal using a staircase approximation.
#
# Encoder state machine (per sample n):
#   if signal[n] > staircase[n-1]:
#       emit bit = 1  →  staircase[n] = staircase[n-1] + delta_step
#   else:
#       emit bit = 0  →  staircase[n] = staircase[n-1] - delta_step
#
# The staircase is clamped to [-1, 1] to prevent runaway accumulation.
#
# Payload layout:
#   [0–3]  initial staircase value as float32 BE  (always 0.0)
#   [4…]   1-bit stream (one bit per sample, MSB-first packed)

def dm_encode(
    signal:     np.ndarray,
    delta_step: float,
) -> tuple[bytes, int]:
    """Encode a float32 signal using standard (non-adaptive) Delta Modulation.

    Parameters
    ----------
    signal : np.ndarray
        Float32, 1-D, values in [-1, 1].
    delta_step : float
        Fixed step size Δ for the staircase accumulator.
        Typical values: 0.01 – 0.1.  Larger Δ reduces slope overload but
        increases granular noise.

    Returns
    -------
    (payload_bytes, padding_bits)
    """
    if delta_step <= 0.0:
        raise ValueError(f"delta_step must be > 0, got {delta_step}.")

    staircase: float = 0.0   # accumulator, starts at mid-range
    writer = BitWriter()

    for sample in signal:
        s = float(sample)
        if s > staircase:
            writer.write_bit(1)
            staircase = min(1.0, staircase + delta_step)
        else:
            writer.write_bit(0)
            staircase = max(-1.0, staircase - delta_step)

    packed, padding_bits = writer.flush()
    # 4-byte float32 header carrying the initial staircase value (0.0).
    init_header = struct.pack(">f", 0.0)
    return init_header + packed, padding_bits


def dm_decode(
    payload:         bytes,
    padding_bits:    int,
    delta_step:      float,
    original_length: int,
) -> np.ndarray:
    """Decode a DM payload back to a float32 signal.

    Parameters
    ----------
    payload : bytes
        Bytes from :func:`dm_encode`.
    padding_bits : int
    delta_step : float
        Must match the value used during encoding.
    original_length : int
        Expected output length.

    Returns
    -------
    np.ndarray  float32, shape (original_length,)
    """
    _HEADER = 4
    if len(payload) < _HEADER:
        raise ValueError(
            f"DM payload too short: need ≥ {_HEADER} bytes, got {len(payload)}."
        )

    (init_staircase,) = struct.unpack(">f", payload[:_HEADER])
    packed_bytes = payload[_HEADER:]

    staircase: float = float(init_staircase)
    bits_list = _materialize_bits(packed_bytes, padding_bits)

    decoded = np.zeros(len(bits_list), dtype=np.float32)
    for i, bit in enumerate(bits_list):
        if bit == 1:
            staircase = min(1.0, staircase + delta_step)
        else:
            staircase = max(-1.0, staircase - delta_step)
        decoded[i] = staircase

    # Trim / zero-pad to original_length.
    out   = np.zeros(original_length, dtype=np.float32)
    n     = min(len(decoded), original_length)
    out[:n] = decoded[:n]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# §3  ADM  —  Adaptive Delta Modulation  (Jayant's algorithm)
# ─────────────────────────────────────────────────────────────────────────────
#
# ADM adapts the step size Δ[n] each sample based on the previous two output
# bits.  Jayant (1974) describes two operating states:
#
#   Slope-overload condition:   two consecutive identical bits.
#       The signal is changing faster than the staircase can track.
#       Response: multiply Δ by the adaptation multiplier M > 1.
#
#   Granular noise condition:   two consecutive alternating bits.
#       The signal is near-stationary; the staircase is hunting around it.
#       Response: divide Δ by M (shrink the step to reduce granular noise).
#
# Jayant's step-size update rule:
#   Δ[n] = Δ[n-1] × M^(2·b[n-1]·b[n-2] - 1)
#
# In practice this reduces to:
#   if b[n] == b[n-1]:   Δ[n+1] = Δ[n] × M
#   else:                Δ[n+1] = Δ[n] / M
#
# Delta is clamped to [delta_min, delta_max] to prevent explosion or vanishing.
#
# Payload layout:
#   Same as DM: 4-byte float32 initial staircase + packed 1-bit stream.
#   The initial delta and multiplier are carried in the AudioHeader, not inline.

_ADM_DELTA_MIN: float = 1e-5
_ADM_DELTA_MAX: float = 1.0


def adm_encode(
    signal:        np.ndarray,
    initial_delta: float,
    multiplier:    float = 1.5,
) -> tuple[bytes, int]:
    """Encode a float32 signal using Adaptive Delta Modulation (Jayant).

    Parameters
    ----------
    signal : np.ndarray
        Float32, 1-D, values in [-1, 1].
    initial_delta : float
        Starting step size Δ[0].  Recommended range: 0.01 – 0.1.
    multiplier : float
        Jayant adaptation multiplier M > 1.  Default 1.5.
        Larger values: faster adaptation, more granular noise.

    Returns
    -------
    (payload_bytes, padding_bits)
    """
    if initial_delta <= 0.0:
        raise ValueError(f"initial_delta must be > 0, got {initial_delta}.")
    if multiplier <= 1.0:
        raise ValueError(f"multiplier must be > 1, got {multiplier}.")

    staircase:  float = 0.0
    delta:      float = initial_delta
    prev_bit:   int   = 0       # b[n-1]; initialised to 0 (no previous bit)
    first_sample: bool = True   # skip adaptation for the very first sample

    writer = BitWriter()

    for sample in signal:
        s = float(sample)
        if s > staircase:
            bit = 1
            staircase = min(1.0, staircase + delta)
        else:
            bit = 0
            staircase = max(-1.0, staircase - delta)

        writer.write_bit(bit)

        # ── Jayant adaptation (applied AFTER emitting the current bit) ────────
        if not first_sample:
            if bit == prev_bit:
                # Slope-overload: step up.
                delta = min(_ADM_DELTA_MAX, delta * multiplier)
            else:
                # Granular noise: step down.
                delta = max(_ADM_DELTA_MIN, delta / multiplier)
        first_sample = False
        prev_bit     = bit

    packed, padding_bits = writer.flush()
    init_header = struct.pack(">f", 0.0)   # initial staircase value
    return init_header + packed, padding_bits


def adm_decode(
    payload:         bytes,
    padding_bits:    int,
    initial_delta:   float,
    multiplier:      float,
    original_length: int,
) -> np.ndarray:
    """Decode an ADM payload back to a float32 signal.

    Parameters
    ----------
    payload : bytes
    padding_bits : int
    initial_delta : float
        Must match encoding parameter.
    multiplier : float
        Must match encoding parameter.
    original_length : int

    Returns
    -------
    np.ndarray  float32, shape (original_length,)
    """
    _HEADER = 4
    if len(payload) < _HEADER:
        raise ValueError(
            f"ADM payload too short: need ≥ {_HEADER} bytes, got {len(payload)}."
        )

    (init_staircase,) = struct.unpack(">f", payload[:_HEADER])
    packed_bytes = payload[_HEADER:]

    staircase:    float = float(init_staircase)
    delta:        float = initial_delta
    prev_bit:     int   = 0
    first_sample: bool  = True

    bits_list = _materialize_bits(packed_bytes, padding_bits)
    decoded   = np.zeros(len(bits_list), dtype=np.float32)

    for i, bit in enumerate(bits_list):
        if bit == 1:
            staircase = min(1.0, staircase + delta)
        else:
            staircase = max(-1.0, staircase - delta)

        decoded[i] = staircase

        # Adaptation (mirrored exactly from encoder)
        if not first_sample:
            if bit == prev_bit:
                delta = min(_ADM_DELTA_MAX, delta * multiplier)
            else:
                delta = max(_ADM_DELTA_MIN, delta / multiplier)
        first_sample = False
        prev_bit     = bit

    out   = np.zeros(original_length, dtype=np.float32)
    n     = min(len(decoded), original_length)
    out[:n] = decoded[:n]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# §4  DPCM  —  Differential PCM with First-Order Predictor
# ─────────────────────────────────────────────────────────────────────────────
#
# DPCM exploits sample-to-sample correlation by encoding the prediction
# residual e[n] rather than the raw amplitude x[n].
#
# Predictor (first-order auto-regressive):
#   x_hat[n] = alpha * x_reconstructed[n-1]
#
# where x_reconstructed[n] is the locally decoded value (using the quantised
# residual) so that the encoder and decoder share identical state and never
# drift apart (closed-loop DPCM).
#
# Encoder loop per sample n:
#   1. Compute prediction:  x_hat[n] = alpha * x_rec[n-1]
#   2. Compute residual:    e[n]     = x[n] - x_hat[n]
#   3. Quantise residual to bits-wide index e_q[n]
#      (range: fixed window [-e_range, +e_range] = [-2.0, 2.0] to handle
#       worst-case initial transient residuals)
#   4. Reconstruct quantised residual: e_rec[n] = mid-point of quantised level
#   5. Update local decoder: x_rec[n] = x_hat[n] + e_rec[n], clamp to [-1,1]
#   6. Emit e_q[n] as bits-wide bit string
#
# Payload layout:
#   Same as PCM (no range header needed — e_range is fixed at 2.0):
#   The packed residual bit-stream only (no header; all params from AudioHeader).

# Fixed residual amplitude window — must cover [-2.0, +2.0] because the
# residual e[n] = x[n] - alpha*x_rec[n-1] can range from -2 to +2 in the
# worst case (x[n]=+1, x_rec[n-1]=-1, alpha≈1).
_DPCM_E_RANGE: float = 2.0


def dpcm_encode(
    signal: np.ndarray,
    bits:   int,
    alpha:  float,
) -> tuple[bytes, int]:
    """Encode a float32 signal using closed-loop Differential PCM.

    Parameters
    ----------
    signal : np.ndarray
        Float32, 1-D, values in [-1, 1].
    bits : int
        Quantisation bit-depth for the residual (1–8).
    alpha : float
        First-order predictor coefficient (0 < alpha ≤ 1, recommended 0.9–0.99).

    Returns
    -------
    (payload_bytes, padding_bits)
    """
    if not (1 <= bits <= 8):
        raise ValueError(f"bits must be 1–8, got {bits}.")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}.")

    n_levels: int   = 1 << bits
    e_min:    float = -_DPCM_E_RANGE
    e_max:    float = +_DPCM_E_RANGE

    x_rec: float = 0.0   # locally reconstructed amplitude (shared encoder state)

    writer = BitWriter()

    for sample in signal:
        x = float(sample)

        # Step 1: Linear prediction from previous reconstructed sample.
        x_hat: float = alpha * x_rec

        # Step 2: Compute residual.
        e: float = x - x_hat

        # Step 3: Quantise residual into integer index.
        e_q: int = _uniform_quantize(e, n_levels, e_min, e_max)

        # Step 4: Reconstruct quantised residual (mid-point of level).
        e_rec: float = _uniform_reconstruct(e_q, n_levels, e_min, e_max)

        # Step 5: Update local decoder state (closed loop — prevents drift).
        x_rec = max(-1.0, min(1.0, x_hat + e_rec))

        # Step 6: Emit fixed-width quantised residual.
        _write_fixed_width(writer, e_q, bits)

    packed, padding_bits = writer.flush()
    return packed, padding_bits


def dpcm_decode(
    payload:         bytes,
    padding_bits:    int,
    bits:            int,
    alpha:           float,
    original_length: int,
) -> np.ndarray:
    """Decode a DPCM payload back to a float32 signal.

    Parameters
    ----------
    payload : bytes
    padding_bits : int
    bits : int
        Must match the encoding bit-depth.
    alpha : float
        Must match the encoding predictor coefficient.
    original_length : int

    Returns
    -------
    np.ndarray  float32, shape (original_length,)
    """
    if not (1 <= bits <= 8):
        raise ValueError(f"bits must be 1–8, got {bits}.")

    n_levels: int   = 1 << bits
    e_min:    float = -_DPCM_E_RANGE
    e_max:    float = +_DPCM_E_RANGE

    bits_list = _materialize_bits(payload, padding_bits)
    pos       = 0

    x_rec:  float       = 0.0
    decoded: list[float] = []

    while pos + bits <= len(bits_list):
        # Read fixed-width residual index.
        e_q, pos = _read_fixed_width(bits_list, pos, bits)

        # Reconstruct residual.
        e_rec: float = _uniform_reconstruct(e_q, n_levels, e_min, e_max)

        # Predict and reconstruct sample (identical state machine as encoder).
        x_hat  = alpha * x_rec
        x_rec  = max(-1.0, min(1.0, x_hat + e_rec))
        decoded.append(x_rec)

    decoded_arr = np.array(decoded, dtype=np.float32)
    out   = np.zeros(original_length, dtype=np.float32)
    n     = min(len(decoded_arr), original_length)
    out[:n] = decoded_arr[:n]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatch API
# ─────────────────────────────────────────────────────────────────────────────

# Mapping from algo ID to human-readable string (for display only).
_AUDIO_ALGO_NAMES: dict[int, str] = {
    0x05: "PCM",
    0x06: "DM",
    0x07: "ADM",
    0x08: "DPCM",
}


def encode_audio(
    signal:          np.ndarray,
    algorithm_id:    int,
    sampling_factor: int  = 1,
    bits:            int  = 8,
    delta_step:      float = 0.05,
    multiplier:      float = 1.5,
    alpha:           float = 0.9,
) -> tuple[bytes, int]:
    """Dispatch to the appropriate audio encoder.

    Parameters
    ----------
    signal : np.ndarray
        1-D float32 mono signal, values in [-1, 1].
    algorithm_id : int
        One of: ALGO_PCM (0x05), ALGO_DM (0x06), ALGO_ADM (0x07),
        ALGO_DPCM (0x08).
    sampling_factor : int
        Downsample factor for PCM (1, 2, or 4).  Ignored by DM/ADM/DPCM.
    bits : int
        Quantisation bit-depth (1–8).  Ignored by DM/ADM.
    delta_step : float
        Step size for DM.  Also used as initial_delta for ADM.
    multiplier : float
        Jayant multiplier for ADM.  Ignored by PCM/DM/DPCM.
    alpha : float
        Predictor coefficient for DPCM.  Ignored by PCM/DM/ADM.

    Returns
    -------
    (payload_bytes, padding_bits)
    """
    from bit_io import ALGO_PCM, ALGO_DM, ALGO_ADM, ALGO_DPCM

    if algorithm_id == ALGO_PCM:
        return pcm_encode(signal, sampling_factor, bits)
    if algorithm_id == ALGO_DM:
        return dm_encode(signal, delta_step)
    if algorithm_id == ALGO_ADM:
        return adm_encode(signal, delta_step, multiplier)
    if algorithm_id == ALGO_DPCM:
        return dpcm_encode(signal, bits, alpha)
    raise ValueError(
        f"Unknown audio algorithm_id 0x{algorithm_id:02X}.  "
        f"Valid: 0x05 PCM, 0x06 DM, 0x07 ADM, 0x08 DPCM."
    )


def decode_audio(
    payload:         bytes,
    padding_bits:    int,
    algorithm_id:    int,
    original_length: int,
    bits:            int   = 8,
    delta_step:      float = 0.05,
    multiplier:      float = 1.5,
    alpha:           float = 0.9,
    sampling_factor: int   = 1,
) -> np.ndarray:
    """Dispatch to the appropriate audio decoder.

    Parameters
    ----------
    payload : bytes
    padding_bits : int
    algorithm_id : int
    original_length : int
        Number of samples in the original source signal (before downsampling).
    bits : int
        PCM / DPCM bit-depth.
    delta_step : float
        DM / ADM step size.
    multiplier : float
        ADM Jayant multiplier.
    alpha : float
        DPCM predictor coefficient.
    sampling_factor : int
        PCM downsample factor (used to calculate upsampling on decode).

    Returns
    -------
    np.ndarray  float32, shape (original_length,)
    """
    from bit_io import ALGO_PCM, ALGO_DM, ALGO_ADM, ALGO_DPCM

    if algorithm_id == ALGO_PCM:
        return pcm_decode(payload, padding_bits, bits, original_length, sampling_factor)
    if algorithm_id == ALGO_DM:
        return dm_decode(payload, padding_bits, delta_step, original_length)
    if algorithm_id == ALGO_ADM:
        return adm_decode(payload, padding_bits, delta_step, multiplier, original_length)
    if algorithm_id == ALGO_DPCM:
        return dpcm_decode(payload, padding_bits, bits, alpha, original_length)
    raise ValueError(
        f"Unknown audio algorithm_id 0x{algorithm_id:02X}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Signal Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_sqnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute the Signal-to-Quantisation-Noise Ratio in dB.

    SQNR = 10 · log₁₀(Σ x²[n]  /  Σ (x[n] - x_rec[n])²)

    Parameters
    ----------
    original : np.ndarray       Float32, the source signal.
    reconstructed : np.ndarray  Float32, the decoded signal (same length).

    Returns
    -------
    float  SQNR in dB.  Returns +inf if error == 0 (perfect reconstruction).
    """
    n = min(len(original), len(reconstructed))
    if n == 0:
        return 0.0
    sig   = original[:n].astype(np.float64)
    rec   = reconstructed[:n].astype(np.float64)
    noise = sig - rec
    sig_power   = float(np.sum(sig   ** 2))
    noise_power = float(np.sum(noise ** 2))
    if noise_power == 0.0:
        return math.inf
    if sig_power == 0.0:
        return 0.0
    return 10.0 * math.log10(sig_power / noise_power)


def compute_theoretical_sqnr(bits: int) -> float:
    """Return the theoretical SQNR for uniform b-bit PCM (SQNR = 6.02b + 1.76 dB).

    Parameters
    ----------
    bits : int  Quantisation bit-depth.

    Returns
    -------
    float  Theoretical SQNR in dB.
    """
    return 6.02 * bits + 1.76
