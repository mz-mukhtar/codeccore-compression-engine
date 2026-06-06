"""
bit_io.py — Bit-Level I/O Engine & Binary Header Protocol
CodecCore: Data Compression Engine

Architecture
────────────
Three cooperating classes implement the low-level binary serialisation layer:

  BinaryHeader  — Serialise / deserialise the variable-length magic file header.
                  A fixed 8-byte block carries type/algorithm/padding/ext-len,
                  followed by N ASCII extension bytes.

  AudioHeader   — Subclass of BinaryHeader that appends a fixed 16-byte
                  parameter block immediately after the extension bytes.  Stores
                  all encoding parameters needed for lossless round-trip decode
                  of PCM, DM, ADM, and DPCM bitstreams:
                    [0]     sampling_factor  (uint8)
                    [1]     bits             (uint8)
                    [2–3]   delta_step_u16   (uint16 BE = round(delta_step × 1000))
                    [4–5]   alpha_u16        (uint16 BE = round(alpha × 10000))
                    [6–9]   original_length  (uint32 BE — samples before downsample)
                    [10–13] sample_rate      (uint32 BE — Hz)
                    [14–15] n_channels       (uint16 BE)

  BitWriter     — Stream arbitrary bits (0/1) into a packed byte buffer.
                  Maintains an 8-bit shift register; flushes complete bytes to
                  an internal bytearray.  `.flush()` returns (bytes, padding_count).

  BitReader     — Stream individual bits back out of a packed byte buffer in
                  MSB-first order.  Respects padding_bits so the logical
                  bit-stream boundary is exact.

All bit manipulation uses native Python bitwise operators (<<, >>, |, &).
No third-party binary-packing libraries are used.

Magic number layout (bytes 0-3):
  0x41 0x52 0x43 0x48  →  ASCII 'A' 'R' 'C' 'H'

Base header layout (variable length):
  [0]        'A'  (0x41)
  [1]        'R'  (0x52)
  [2]        'C'  (0x43)
  [3]        'H'  (0x48)
  [4]        File-type flag    : 0x01 = Picture  |  0x02 = Binary/PDF
                                 0x03 = Audio
  [5]        Algorithm ID      : 0x01 = Huffman  |  0x02 = LZ78
                                 0x03 = Chained  |  0x04 = RLE+Quantization
                                 0x05 = PCM      |  0x06 = DM
                                 0x07 = ADM      |  0x08 = DPCM
  [6]        Padding bits      : 0–7  (trailing zero bits in last payload byte)
  [7]        Extension length  : N   (number of ASCII bytes that follow)
  [8..8+N-1] Extension bytes   : N ASCII chars of the original extension

For AudioHeader, 16 additional parameter bytes follow the extension bytes.

Total sizes:
  BinaryHeader  = 8 + N bytes.
  AudioHeader   = 8 + N + 16 bytes.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Public constants
# ──────────────────────────────────────────────────────────────────────────────

#: The 4-byte magic signature that must appear at the start of every .abc file.
MAGIC: bytes = b"ARCH"

#: Size of the fixed-length portion of the header (before the extension field).
#: Bytes [0..7]: ARCH magic (4) + file_type (1) + algorithm_id (1)
#:              + padding_bits (1) + ext_len (1).
HEADER_FIXED_SIZE: int = 8

# Maximum number of ASCII characters permitted in the extension field.
# ext_len is stored in a single uint8, so the ceiling is 255.
EXTENSION_MAX_LEN: int = 255

# File-type flag values.
FILE_TYPE_PICTURE: int = 0x01
FILE_TYPE_BINARY:  int = 0x02
FILE_TYPE_AUDIO:   int = 0x03   # Phase 6 — waveform coding.

# Algorithm-ID flag values (lossless / image).
ALGO_HUFFMAN:   int = 0x01
ALGO_LZ78:      int = 0x02
ALGO_CHAINED:   int = 0x03
ALGO_RLE_QUANT: int = 0x04   # RLE + Spatial Quantization (image pipeline).

# Algorithm-ID flag values (audio — Phase 6).
ALGO_PCM:  int = 0x05   # Uniform PCM quantization.
ALGO_DM:   int = 0x06   # Delta Modulation.
ALGO_ADM:  int = 0x07   # Adaptive Delta Modulation (Jayant).
ALGO_DPCM: int = 0x08   # Differential PCM with first-order predictor.

#: Fixed size (bytes) of the AudioHeader parameter block that follows the
#: extension bytes in an audio archive.
AUDIO_PARAM_SIZE: int = 16

# Human-readable labels (used by the UI and the status board).
FILE_TYPE_LABELS: dict[int, str] = {
    FILE_TYPE_PICTURE: "Picture",
    FILE_TYPE_BINARY:  "Binary File / PDF",
    FILE_TYPE_AUDIO:   "Audio Waveform",
}

ALGORITHM_LABELS: dict[int, str] = {
    ALGO_HUFFMAN:   "Huffman",
    ALGO_LZ78:      "LZ78",
    ALGO_CHAINED:   "Chained (LZ78 + Huffman)",
    ALGO_RLE_QUANT: "RLE + Spatial Quantization",
    ALGO_PCM:       "PCM (Uniform Quantization)",
    ALGO_DM:        "Delta Modulation (DM)",
    ALGO_ADM:       "Adaptive Delta Modulation (ADM)",
    ALGO_DPCM:      "Differential PCM (DPCM)",
}

# Reverse-lookup: UI label → algorithm ID (used when building the header).
ALGORITHM_IDS: dict[str, int] = {v: k for k, v in ALGORITHM_LABELS.items()}

# Reverse-lookup: UI label → file-type ID.
FILE_TYPE_IDS: dict[str, int] = {v: k for k, v in FILE_TYPE_LABELS.items()}


# ──────────────────────────────────────────────────────────────────────────────
# BinaryHeader
# ──────────────────────────────────────────────────────────────────────────────

class BinaryHeader:
    """Encapsulates the variable-length metadata prefix of every .abc archive.

    Phase 3 Header Layout
    ─────────────────────
    Fixed portion (always 8 bytes):
        [0–3]  b'ARCH'         — 4-byte magic number
        [4]    file_type       — 0x01 (Picture) or 0x02 (Binary/PDF)
        [5]    algorithm_id    — 0x01 Huffman | 0x02 LZ78 | 0x03 Chained
        [6]    padding_bits    — 0–7 trailing zero-bits in last payload byte
        [7]    ext_len         — N, number of ASCII chars in extension (0–255)

    Variable portion (N bytes):
        [8 .. 8+N-1]           — ASCII characters of the original file
                                  extension WITHOUT the leading dot
                                  (e.g., 3 bytes "pdf" for a .pdf file).

    Total serialised size = 8 + N bytes (accessible via ``.total_size``).

    Parameters
    ----------
    file_type:
        Integer flag identifying the source data category.
    algorithm_id:
        Integer flag identifying the compression codec.
    padding_bits:
        Number of trailing zero bits (0–7) appended to the compressed payload.
    extension:
        The original file extension string WITHOUT the leading dot
        (e.g. ``"pdf"``, ``"txt"``, ``"sh"``).  Empty string if the file
        has no extension.  Maximum ``EXTENSION_MAX_LEN`` (255) characters.

    Examples
    --------
    Round-trip serialisation::

        hdr = BinaryHeader(FILE_TYPE_PICTURE, ALGO_HUFFMAN, 3, "png")
        raw = hdr.serialize()          # 8 + 3 = 11 bytes
        out = BinaryHeader.deserialize(raw)
        assert out.extension   == "png"
        assert out.padding_bits == 3
        assert out.total_size   == 11
    """

    def __init__(
        self,
        file_type:    int,
        algorithm_id: int,
        padding_bits: int,
        extension:    str = "",
    ) -> None:
        # ── Validation ────────────────────────────────────────────────────────
        if padding_bits < 0 or padding_bits > 7:
            raise ValueError(
                f"padding_bits must be 0–7, got {padding_bits!r}."
            )
        if file_type not in FILE_TYPE_LABELS:
            raise ValueError(
                f"Unknown file_type 0x{file_type:02X}. "
                f"Valid: {list(FILE_TYPE_LABELS)!r}."
            )
        if algorithm_id not in ALGORITHM_LABELS:
            raise ValueError(
                f"Unknown algorithm_id 0x{algorithm_id:02X}. "
                f"Valid: {list(ALGORITHM_LABELS)!r}."
            )
        if len(extension) > EXTENSION_MAX_LEN:
            raise ValueError(
                f"Extension is {len(extension)} characters; "
                f"maximum allowed is {EXTENSION_MAX_LEN}."
            )
        try:
            extension.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"Extension {extension!r} contains non-ASCII characters: {exc}"
            ) from exc

        self.file_type:    int = file_type
        self.algorithm_id: int = algorithm_id
        self.padding_bits: int = padding_bits
        self.extension:    str = extension

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_size(self) -> int:
        """Total byte count of the serialised header (fixed + variable parts)."""
        return HEADER_FIXED_SIZE + len(self.extension)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def serialize(self) -> bytes:
        """Pack all header fields into a variable-length byte sequence.

        Layout::

            [0-3]       b'ARCH'          (4 bytes — magic)
            [4]         file_type        (1 byte)
            [5]         algorithm_id     (1 byte)
            [6]         padding_bits     (1 byte, 0–7)
            [7]         ext_len          (1 byte, N)
            [8..8+N-1]  extension bytes  (N bytes, ASCII)

        Returns
        -------
        bytes
            Exactly ``self.total_size`` bytes.
        """
        ext_bytes: bytes = self.extension.encode("ascii")
        ext_len:   int   = len(ext_bytes)

        buf = bytearray(HEADER_FIXED_SIZE + ext_len)

        # Bytes 0–3: ARCH magic.
        buf[0] = 0x41  # 'A'
        buf[1] = 0x52  # 'R'
        buf[2] = 0x43  # 'C'
        buf[3] = 0x48  # 'H'

        # Byte 4: file-type flag.
        buf[4] = self.file_type & 0xFF

        # Byte 5: algorithm-ID flag.
        buf[5] = self.algorithm_id & 0xFF

        # Byte 6: padding-bits count.
        buf[6] = self.padding_bits & 0xFF

        # Byte 7: extension length.
        buf[7] = ext_len & 0xFF

        # Bytes 8..(8+N-1): extension ASCII bytes.
        buf[8 : 8 + ext_len] = ext_bytes

        return bytes(buf)

    # ── Deserialisation ───────────────────────────────────────────────────────

    @classmethod
    def deserialize(cls, data: bytes) -> "BinaryHeader":
        """Parse the leading bytes of an .abc file and return a ``BinaryHeader``.

        The method reads the fixed 8-byte portion to discover ``ext_len``, then
        reads the subsequent N extension bytes.  The caller must supply at least
        ``8 + ext_len`` bytes.  Typically you pass the entire file buffer; the
        method only consumes ``total_size`` bytes and the rest are the payload.

        Parameters
        ----------
        data:
            A bytes-like object containing the serialised header at the start.
            Must be at least ``HEADER_FIXED_SIZE`` (8) bytes long.

        Returns
        -------
        BinaryHeader
            A fully instantiated header object with ``total_size`` set to the
            exact number of bytes consumed.

        Raises
        ------
        ValueError
            If the data is shorter than 8 bytes, the ARCH magic is absent or
            corrupt, any flag value is outside its valid range, or the
            extension bytes are not valid ASCII.
        """
        if len(data) < HEADER_FIXED_SIZE:
            raise ValueError(
                f"Data must be at least {HEADER_FIXED_SIZE} bytes to contain "
                f"the fixed header portion; got {len(data)} bytes."
            )

        # ── Verify ARCH magic (bytes 0–3) ─────────────────────────────────────
        if (
            data[0] != 0x41
            or data[1] != 0x52
            or data[2] != 0x43
            or data[3] != 0x48
        ):
            found = data[:4]
            raise ValueError(
                f"Invalid magic number: expected b'ARCH' "
                f"(0x41 0x52 0x43 0x48), "
                f"found {found!r} ({' '.join(f'0x{b:02X}' for b in found)})."
            )

        file_type    = data[4]
        algorithm_id = data[5]
        padding_bits = data[6]
        ext_len      = data[7]

        # ── Validate fixed-portion fields ─────────────────────────────────────
        if file_type not in FILE_TYPE_LABELS:
            raise ValueError(
                f"Unrecognised file_type 0x{file_type:02X} in header."
            )
        if algorithm_id not in ALGORITHM_LABELS:
            raise ValueError(
                f"Unrecognised algorithm_id 0x{algorithm_id:02X} in header."
            )
        if padding_bits > 7:
            raise ValueError(
                f"Corrupt padding_bits {padding_bits} in header (max 7)."
            )

        # ── Read extension bytes (variable portion) ───────────────────────────
        total_needed = HEADER_FIXED_SIZE + ext_len
        if len(data) < total_needed:
            raise ValueError(
                f"Header declares extension_length={ext_len} but only "
                f"{len(data) - HEADER_FIXED_SIZE} bytes remain after the "
                f"fixed portion."
            )

        ext_raw: bytes = data[HEADER_FIXED_SIZE : HEADER_FIXED_SIZE + ext_len]
        try:
            extension: str = ext_raw.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Extension field contains non-ASCII bytes {ext_raw!r}: {exc}"
            ) from exc

        return cls(
            file_type=file_type,
            algorithm_id=algorithm_id,
            padding_bits=padding_bits,
            extension=extension,
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"BinaryHeader("
            f"file_type=0x{self.file_type:02X}"
            f" ({FILE_TYPE_LABELS.get(self.file_type, '?')}), "
            f"algorithm_id=0x{self.algorithm_id:02X}"
            f" ({ALGORITHM_LABELS.get(self.algorithm_id, '?')}), "
            f"padding_bits={self.padding_bits}, "
            f"extension={self.extension!r}, "
            f"total_size={self.total_size})"
        )



# ──────────────────────────────────────────────────────────────────────────────
# AudioHeader
# ──────────────────────────────────────────────────────────────────────────────

class AudioHeader(BinaryHeader):
    """Extends BinaryHeader with a 16-byte fixed parameter block for audio archives.

    Layout (follows immediately after the extension bytes):
    ::

        [0]     sampling_factor  (uint8)   — downsample ratio: 1, 2, or 4
        [1]     bits             (uint8)   — quantization bit-depth: 1–8
        [2–3]   delta_step_u16   (uint16 BE) — delta step × 1000 (integer encode)
        [4–5]   alpha_u16        (uint16 BE) — predictor α × 10000
        [6–9]   original_length  (uint32 BE) — samples before downsampling
        [10–13] sample_rate      (uint32 BE) — source sample rate in Hz
        [14–15] n_channels       (uint16 BE) — number of audio channels

    Parameters
    ----------
    algorithm_id : int
        One of ALGO_PCM, ALGO_DM, ALGO_ADM, ALGO_DPCM.
    padding_bits : int
        Trailing padding bits in the compressed payload (0–7).
    extension : str
        Original file extension without leading dot (e.g. ``"wav"``).
    sampling_factor : int
        Downsample factor applied during encoding (1 = no downsample).
    bits : int
        Quantization bit-depth (1–8).
    delta_step : float
        Step size Δ for DM / ADM.  Stored as round(Δ × 1000) in uint16.
        Ignored (stored as 0) for PCM and DPCM.
    alpha : float
        Predictor coefficient α for DPCM (0 < α ≤ 1).  Stored as
        round(α × 10000) in uint16.  Ignored (stored as 0) for PCM / DM / ADM.
    original_length : int
        Total number of samples in the source signal *before* downsampling.
    sample_rate : int
        Source sample rate in Hz (e.g. 44100, 22050).
    n_channels : int
        Number of audio channels (1 = mono, 2 = stereo).

    Examples
    --------
    Round-trip serialisation::

        hdr = AudioHeader(
            algorithm_id=ALGO_DPCM, padding_bits=3, extension="wav",
            sampling_factor=2, bits=4, delta_step=0.0, alpha=0.9,
            original_length=44100, sample_rate=44100, n_channels=1,
        )
        raw  = hdr.serialize()
        out  = AudioHeader.deserialize_audio(raw)
        assert out.alpha == 0.9
        assert out.sample_rate == 44100
    """

    def __init__(
        self,
        algorithm_id:    int,
        padding_bits:    int,
        extension:       str   = "wav",
        sampling_factor: int   = 1,
        bits:            int   = 8,
        delta_step:      float = 0.0,
        alpha:           float = 0.0,
        original_length: int   = 0,
        sample_rate:     int   = 44100,
        n_channels:      int   = 1,
    ) -> None:
        super().__init__(
            file_type    = FILE_TYPE_AUDIO,
            algorithm_id = algorithm_id,
            padding_bits = padding_bits,
            extension    = extension,
        )
        if sampling_factor not in (1, 2, 4):
            raise ValueError(
                f"sampling_factor must be 1, 2, or 4; got {sampling_factor}."
            )
        if not (1 <= bits <= 8):
            raise ValueError(f"bits must be 1–8; got {bits}.")
        if original_length < 0:
            raise ValueError(f"original_length must be ≥ 0; got {original_length}.")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0; got {sample_rate}.")
        if n_channels not in (1, 2):
            raise ValueError(f"n_channels must be 1 or 2; got {n_channels}.")

        self.sampling_factor: int   = sampling_factor
        self.bits:            int   = bits
        self.delta_step:      float = float(delta_step)
        self.alpha:           float = float(alpha)
        self.original_length: int   = original_length
        self.sample_rate:     int   = sample_rate
        self.n_channels:      int   = n_channels

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_size(self) -> int:
        """Total byte count: base fixed (8) + ext (N) + audio param block (16)."""
        return HEADER_FIXED_SIZE + len(self.extension) + AUDIO_PARAM_SIZE

    # ── Serialisation ─────────────────────────────────────────────────────────

    def serialize(self) -> bytes:
        """Pack the base header bytes, then append the 16-byte audio parameter block.

        Layout after the base extension bytes:
        ::

            [0]     sampling_factor  uint8
            [1]     bits             uint8
            [2–3]   round(delta_step × 1000)   uint16 BE
            [4–5]   round(alpha × 10000)        uint16 BE
            [6–9]   original_length             uint32 BE
            [10–13] sample_rate                 uint32 BE
            [14–15] n_channels                  uint16 BE

        Returns
        -------
        bytes
            Exactly ``self.total_size`` bytes.
        """
        # Build the base 8-byte fixed block + N extension bytes, but we need
        # to override ext_len accounting to exclude the audio param block from
        # the base class logic.  We re-implement the full serialisation here
        # so that ext_len in byte[7] still correctly counts only the extension
        # (not the audio param block) — ensuring BinaryHeader.deserialize can
        # still parse the fixed portion without knowing about AudioHeader.
        ext_bytes: bytes = self.extension.encode("ascii")
        ext_len:   int   = len(ext_bytes)

        buf = bytearray(HEADER_FIXED_SIZE + ext_len + AUDIO_PARAM_SIZE)

        # Base header bytes 0–7
        buf[0] = 0x41   # 'A'
        buf[1] = 0x52   # 'R'
        buf[2] = 0x43   # 'C'
        buf[3] = 0x48   # 'H'
        buf[4] = FILE_TYPE_AUDIO & 0xFF
        buf[5] = self.algorithm_id & 0xFF
        buf[6] = self.padding_bits & 0xFF
        buf[7] = ext_len & 0xFF

        # Extension bytes 8..(8+N-1)
        buf[8 : 8 + ext_len] = ext_bytes

        # ── Audio parameter block — 16 bytes at offset (8 + ext_len) ──────────
        p = 8 + ext_len   # base offset of parameter block

        # [0] sampling_factor — uint8
        buf[p + 0] = self.sampling_factor & 0xFF

        # [1] bits — uint8
        buf[p + 1] = self.bits & 0xFF

        # [2–3] delta_step × 1000, clamped to uint16 range
        ds_u16 = min(65535, max(0, round(self.delta_step * 1000)))
        buf[p + 2] = (ds_u16 >> 8) & 0xFF
        buf[p + 3] = ds_u16 & 0xFF

        # [4–5] alpha × 10000, clamped to uint16 range
        al_u16 = min(65535, max(0, round(self.alpha * 10000)))
        buf[p + 4] = (al_u16 >> 8) & 0xFF
        buf[p + 5] = al_u16 & 0xFF

        # [6–9] original_length — uint32 BE
        ol = self.original_length
        buf[p + 6]  = (ol >> 24) & 0xFF
        buf[p + 7]  = (ol >> 16) & 0xFF
        buf[p + 8]  = (ol >>  8) & 0xFF
        buf[p + 9]  =  ol        & 0xFF

        # [10–13] sample_rate — uint32 BE
        sr = self.sample_rate
        buf[p + 10] = (sr >> 24) & 0xFF
        buf[p + 11] = (sr >> 16) & 0xFF
        buf[p + 12] = (sr >>  8) & 0xFF
        buf[p + 13] =  sr        & 0xFF

        # [14–15] n_channels — uint16 BE
        nc = self.n_channels
        buf[p + 14] = (nc >> 8) & 0xFF
        buf[p + 15] =  nc       & 0xFF

        return bytes(buf)

    # ── Deserialisation ───────────────────────────────────────────────────────

    @classmethod
    def deserialize_audio(cls, data: bytes) -> "AudioHeader":
        """Parse an audio .abc archive header into an AudioHeader instance.

        Reads the standard BinaryHeader fixed portion, then reads the additional
        16-byte audio parameter block that follows the extension bytes.

        Parameters
        ----------
        data:
            Raw bytes of the entire .abc file (the method only consumes
            ``total_size`` bytes).

        Returns
        -------
        AudioHeader

        Raises
        ------
        ValueError
            If the magic number is wrong, the file is too short for the
            declared extension + parameter block, or any field is out of range.
        """
        if len(data) < HEADER_FIXED_SIZE:
            raise ValueError(
                f"Data too short for ARCH fixed header: need "
                f"{HEADER_FIXED_SIZE} bytes, got {len(data)}."
            )

        # Verify magic
        if data[:4] != b"ARCH":
            raise ValueError(
                f"Invalid magic: expected b'ARCH', "
                f"found {data[:4]!r}."
            )

        algorithm_id = data[5]
        padding_bits = data[6]
        ext_len      = data[7]

        total_needed = HEADER_FIXED_SIZE + ext_len + AUDIO_PARAM_SIZE
        if len(data) < total_needed:
            raise ValueError(
                f"AudioHeader needs {total_needed} bytes (8 fixed + "
                f"{ext_len} ext + {AUDIO_PARAM_SIZE} params), "
                f"got {len(data)}."
            )

        # Extension
        ext_raw   = data[HEADER_FIXED_SIZE : HEADER_FIXED_SIZE + ext_len]
        extension = ext_raw.decode("ascii")

        # Audio parameter block
        p = HEADER_FIXED_SIZE + ext_len

        sampling_factor  = data[p + 0]
        bits             = data[p + 1]
        ds_u16           = (data[p + 2] << 8) | data[p + 3]
        al_u16           = (data[p + 4] << 8) | data[p + 5]
        original_length  = (
            (data[p +  6] << 24)
            | (data[p +  7] << 16)
            | (data[p +  8] <<  8)
            |  data[p +  9]
        )
        sample_rate = (
            (data[p + 10] << 24)
            | (data[p + 11] << 16)
            | (data[p + 12] <<  8)
            |  data[p + 13]
        )
        n_channels = (data[p + 14] << 8) | data[p + 15]

        delta_step = ds_u16 / 1000.0
        alpha      = al_u16 / 10000.0

        return cls(
            algorithm_id    = algorithm_id,
            padding_bits    = padding_bits,
            extension       = extension,
            sampling_factor = sampling_factor,
            bits            = bits,
            delta_step      = delta_step,
            alpha           = alpha,
            original_length = original_length,
            sample_rate     = sample_rate,
            n_channels      = n_channels,
        )

    def __repr__(self) -> str:
        return (
            f"AudioHeader("
            f"algo={ALGORITHM_LABELS.get(self.algorithm_id,'?')}, "
            f"bits={self.bits}, sf={self.sampling_factor}, "
            f"delta={self.delta_step:.4f}, alpha={self.alpha:.4f}, "
            f"orig_len={self.original_length}, sr={self.sample_rate}, "
            f"ch={self.n_channels}, pad={self.padding_bits}, "
            f"ext={self.extension!r})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# BitWriter
# ──────────────────────────────────────────────────────────────────────────────

class BitWriter:
    """Stream arbitrary bits into a packed byte buffer (MSB-first packing).

    Internally maintains a single-byte shift register (``_current_byte``) and a
    bit-position counter (``_bit_count``).  Every time the register fills to 8
    bits it is appended to ``_buffer`` and both state variables reset.

    Usage pattern::

        writer = BitWriter()
        writer.write_bits("110100")
        writer.write_bit(1)
        payload_bytes, padding_count = writer.flush()

    The returned ``padding_count`` is the number of zero bits appended to the
    final byte to align it to a byte boundary.  Store this value in the
    ``BinaryHeader`` so that ``BitReader`` can discard those bits during
    decompression.
    """

    def __init__(self) -> None:
        # Internal accumulation buffer; complete bytes land here.
        self._buffer: bytearray = bytearray()

        # Shift register: accumulates bits one-by-one until a full byte forms.
        # Only the lower 8 bits are ever used.
        self._current_byte: int = 0

        # Number of bits currently loaded into ``_current_byte`` (0–7).
        # When this reaches 8, the byte is committed and the register resets.
        self._bit_count: int = 0

    # ── Core write primitives ─────────────────────────────────────────────────

    def write_bit(self, bit: int) -> None:
        """Append a single bit to the stream.

        Parameters
        ----------
        bit:
            Must be 0 or 1.  Any other value raises ``ValueError``.

        Notes
        -----
        Bits are packed MSB-first: the first bit written becomes bit 7 of the
        first byte, the second bit becomes bit 6, and so on.

        Mechanically:
        1. Shift ``_current_byte`` left by 1, making room for the new bit.
        2. OR the new bit into the least-significant position.
        3. Increment ``_bit_count``.
        4. When ``_bit_count`` reaches 8, commit the byte and reset.
        """
        if bit not in (0, 1):
            raise ValueError(f"write_bit expects 0 or 1, got {bit!r}.")

        self._current_byte = ((self._current_byte << 1) | bit) & 0xFF
        self._bit_count += 1

        if self._bit_count == 8:
            self._buffer.append(self._current_byte)
            self._current_byte = 0
            self._bit_count    = 0

    def write_bits(self, bit_string: str) -> None:
        """Write a sequence of bits from a string of '0'/'1' characters.

        Parameters
        ----------
        bit_string:
            String consisting solely of ``'0'`` and ``'1'``.  Empty string is
            a valid no-op.  Any other character raises ``ValueError``.
        """
        for ch in bit_string:
            if ch == "0":
                self.write_bit(0)
            elif ch == "1":
                self.write_bit(1)
            else:
                raise ValueError(
                    f"write_bits expects '0'/'1' characters; "
                    f"found {ch!r} in {bit_string!r}."
                )

    # ── Flush ─────────────────────────────────────────────────────────────────

    def flush(self) -> tuple[bytes, int]:
        """Finalise the stream and return the packed byte payload.

        If the final byte in the shift register is only partially filled, the
        remaining bit positions are padded with zeros on the right (LSB side).
        The padding count is also returned so it can be stored in the header.

        Returns
        -------
        tuple[bytes, int]
            - ``payload``:      Complete packed byte sequence (immutable).
            - ``padding_bits``: Zero bits appended to the last byte (0–7).
              Zero when the total bit count is an exact multiple of 8.

        Notes
        -----
        The writer's internal state is NOT mutated; calling ``flush()`` twice
        returns identical results (idempotent).  Calling on an empty writer
        returns ``(b'', 0)``.
        """
        if self._bit_count == 0:
            return bytes(self._buffer), 0

        padding_bits: int  = 8 - self._bit_count
        padded_byte:  int  = (self._current_byte << padding_bits) & 0xFF

        final_buffer = bytearray(self._buffer)
        final_buffer.append(padded_byte)

        return bytes(final_buffer), padding_bits

    # ── Inspection ────────────────────────────────────────────────────────────

    @property
    def bits_written(self) -> int:
        """Total bits fed into the writer so far (not counting flush padding)."""
        return len(self._buffer) * 8 + self._bit_count

    def __repr__(self) -> str:
        return (
            f"BitWriter("
            f"complete_bytes={len(self._buffer)}, "
            f"bits_in_register={self._bit_count}, "
            f"total_bits={self.bits_written})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# BitReader
# ──────────────────────────────────────────────────────────────────────────────

class BitReader:
    """Stream individual bits out of a packed byte buffer (MSB-first).

    Respects the ``padding_bits`` count so that the generator stops at the true
    logical end of the bit-stream — trailing zero padding bits are never yielded.

    Parameters
    ----------
    payload:
        The raw compressed byte buffer (the data after the ``BinaryHeader``).
    padding_bits:
        Number of trailing bits in the *last byte* of ``payload`` that are
        meaningless padding.  Must be 0–7.  Comes directly from
        ``BinaryHeader.padding_bits``.

    Usage::

        reader = BitReader(payload_bytes, padding_bits=3)
        for bit in reader.read_bits():
            ...   # bit is 0 or 1

    Notes
    -----
    * Bits within each byte are yielded MSB-first (bit 7 → bit 0), matching
      the packing order of ``BitWriter.write_bit()``.
    * Empty payload → generator yields nothing immediately.
    * Single-pass generator; create a new instance to re-iterate.
    """

    def __init__(self, payload: bytes, padding_bits: int) -> None:
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError(
                f"payload must be bytes or bytearray, "
                f"got {type(payload).__name__!r}."
            )
        if not isinstance(padding_bits, int) or padding_bits < 0 or padding_bits > 7:
            raise ValueError(
                f"padding_bits must be an integer 0–7, got {padding_bits!r}."
            )

        self._payload:      bytes = bytes(payload)
        self._padding_bits: int   = padding_bits

    # ── Generator ─────────────────────────────────────────────────────────────

    def read_bits(self):
        """Yield each logical bit in the payload as an integer (0 or 1).

        Iterates byte-by-byte; for each byte extracts bits MSB-first:

            bit_k = (byte >> (7 - k)) & 1   for k in 0..7

        Stops before the trailing ``_padding_bits`` bits in the final byte.

        Yields
        ------
        int
            0 or 1.
        """
        total_bytes: int = len(self._payload)

        if total_bytes == 0:
            return

        total_logical_bits: int = total_bytes * 8 - self._padding_bits
        bits_yielded: int = 0

        for byte_index in range(total_bytes):
            byte: int = self._payload[byte_index]
            for bit_position in range(8):
                if bits_yielded >= total_logical_bits:
                    return
                yield (byte >> (7 - bit_position)) & 1
                bits_yielded += 1

    # ── Convenience helpers ───────────────────────────────────────────────────

    def reconstruct_bytes(self) -> bytes:
        """Consume all bits and reassemble the original byte sequence.

        Inverse of encoding a byte stream via ``BitWriter`` (one byte at a time,
        8 bits each).  Collects bits in groups of 8 and assembles bytes via
        the same MSB-first shift logic.  Any trailing partial group (malformed
        stream) is left-shifted and appended.

        Returns
        -------
        bytes
            The reconstructed byte payload.
        """
        result:       bytearray = bytearray()
        current_byte: int       = 0
        bit_count:    int       = 0

        for bit in self.read_bits():
            current_byte = ((current_byte << 1) | bit) & 0xFF
            bit_count += 1
            if bit_count == 8:
                result.append(current_byte)
                current_byte = 0
                bit_count    = 0

        if bit_count > 0:
            result.append((current_byte << (8 - bit_count)) & 0xFF)

        return bytes(result)

    def __repr__(self) -> str:
        total_logical_bits = len(self._payload) * 8 - self._padding_bits
        return (
            f"BitReader("
            f"payload_bytes={len(self._payload)}, "
            f"padding_bits={self._padding_bits}, "
            f"logical_bits={total_logical_bits})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers  (Phase 2 passthrough shim — kept for utility / testing)
# ──────────────────────────────────────────────────────────────────────────────

def encode_bytes_to_bitwriter(raw: bytes) -> tuple[bytes, int]:
    """Pass a raw byte sequence through ``BitWriter`` byte-by-byte.

    Each byte is written as an 8-bit MSB-first binary string.  Because input is
    always byte-aligned the returned padding_bits is always 0.

    This function is the Phase 2 passthrough shim and is retained for utility
    and round-trip testing.  The Phase 3 codec engines replace it in the
    application pipeline.
    """
    writer = BitWriter()
    for byte_val in raw:
        writer.write_bits(format(byte_val, "08b"))
    return writer.flush()


def decode_bitreader_to_bytes(payload: bytes, padding_bits: int) -> bytes:
    """Reconstruct the original byte sequence from a bit-packed payload.

    Inverse of ``encode_bytes_to_bitwriter``.  Retained for utility / testing.
    """
    reader = BitReader(payload, padding_bits)
    return reader.reconstruct_bytes()
