"""
codec_engine.py — Core Compression Algorithm Implementations
CodecCore: Multi-Codec Compression Laboratory — Phase 5

Note on module naming
─────────────────────
This file is named ``codec_engine.py`` rather than ``codecs.py`` to avoid
shadowing Python's standard-library ``codecs`` module.  That stdlib module is
loaded at interpreter start-up and permanently cached in ``sys.modules``; a
local file named ``codecs.py`` would never be importable via a plain
``import codecs`` statement since Python would always find the cached stdlib
version first.

Algorithms implemented
──────────────────────
  §1  Huffman Coding          (ALGO_HUFFMAN  = 0x01)
  §2  Lempel-Ziv 78           (ALGO_LZ78     = 0x02)
  §3  Chained: LZ78 → Huffman (ALGO_CHAINED  = 0x03)

Design constraints
──────────────────
  • Pure-Python only: no zlib, gzip, bz2, lzma, or any external compression
    library is imported.  All arithmetic uses native Python integers and native
    bitwise operators (<<, >>, |, &).
  • Both encoder and decoder for each codec share an identical, self-contained
    byte-stream format so no external side-channel (files, databases, network)
    is needed to reconstruct the original data.

Public API
──────────
  compress(data, algorithm_id)                    → (compressed_bytes, padding_bits)
  decompress(payload, algorithm_id, padding_bits)  → original_bytes

  Information Theory (Phase 5):
    analyze_source(data)                           → InformationTheoryMetrics
    shannon_entropy(data)                          → float  (bits / symbol)
    average_code_length(data, compressed, padding) → float  (bits / symbol)
    coding_efficiency(entropy, avg_code_len)       → float  (percent)

  Lower-level entry points:
    huffman_encode / huffman_decode
    lz78_encode    / lz78_decode
    chained_encode / chained_decode
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

from bit_io import (
    BitWriter,
    BitReader,
    ALGO_HUFFMAN,
    ALGO_LZ78,
    ALGO_CHAINED,
    ALGO_RLE_QUANT,
)


# ══════════════════════════════════════════════════════════════════════════════
# §1  HUFFMAN CODEC
# ══════════════════════════════════════════════════════════════════════════════
#
# Huffman coding assigns variable-length, prefix-free binary codes to each
# unique byte symbol in the input, with shorter codes assigned to more frequent
# symbols.  The resulting bit-stream is shorter than the original when the
# input has a skewed frequency distribution.
#
# Self-contained payload format:
#   ┌─────────────────────────────────────────────────────────────┐
#   │  FREQUENCY TABLE BLOCK  (byte-aligned metadata)            │
#   │    [0–1]  N unique symbols (uint16, big-endian)            │
#   │    Per entry (sorted by symbol value):                     │
#   │      [0]    symbol byte value  (uint8)                     │
#   │      [1–4]  frequency count   (uint32, big-endian)         │
#   │    Total: 2 + 5·N bytes                                    │
#   ├─────────────────────────────────────────────────────────────┤
#   │  HUFFMAN BIT-STREAM  (bit-packed, may have trailing zeros) │
#   └─────────────────────────────────────────────────────────────┘
#
# The BinaryHeader's padding_bits refers to the last byte of the bit-stream
# section.  The frequency block is byte-aligned so it does not contribute any
# padding.


class _HuffmanNode:
    """Node in the Huffman binary tree.

    Leaf nodes carry a ``symbol`` (0–255).  Internal nodes have
    ``symbol = None`` and non-None ``left`` / ``right`` children.
    """

    __slots__ = ("symbol", "freq", "left", "right")

    def __init__(
        self,
        symbol: Optional[int]          = None,
        freq:   int                    = 0,
        left:   Optional["_HuffmanNode"] = None,
        right:  Optional["_HuffmanNode"] = None,
    ) -> None:
        self.symbol = symbol
        self.freq   = freq
        self.left   = left
        self.right  = right

    def __lt__(self, other: "_HuffmanNode") -> bool:
        # Fallback comparator required by heapq when two (freq, counter, node)
        # tuples have equal freq AND equal counter — should never occur in
        # practice but needed to satisfy Python's type system.
        return self.freq < other.freq


# ── Private helpers ───────────────────────────────────────────────────────────

def _count_frequencies(data: bytes) -> dict[int, int]:
    """Return a {byte_value: occurrence_count} mapping for *data*."""
    freq: dict[int, int] = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1
    return freq


def _build_huffman_tree(freq: dict[int, int]) -> _HuffmanNode:
    """Construct the canonical Huffman tree from a frequency table.

    Determinism guarantee
    ─────────────────────
    Symbols are inserted into the min-heap in ascending numeric order.  Ties in
    frequency are broken by an insertion counter so the heap never needs to
    compare ``_HuffmanNode`` objects directly.  This guarantees that the encoder
    and the decoder — both operating on the same frequency table — always
    produce the identical tree topology.
    """
    counter = 0
    heap: list[tuple[int, int, _HuffmanNode]] = []

    for sym in sorted(freq):           # ascending symbol order → deterministic
        heapq.heappush(heap, (freq[sym], counter, _HuffmanNode(sym, freq[sym])))
        counter += 1

    while len(heap) > 1:
        f1, _, left  = heapq.heappop(heap)
        f2, _, right = heapq.heappop(heap)
        parent = _HuffmanNode(freq=f1 + f2, left=left, right=right)
        heapq.heappush(heap, (f1 + f2, counter, parent))
        counter += 1

    _, _, root = heap[0]
    return root


def _assign_codes(node: _HuffmanNode, prefix: str = "") -> dict[int, str]:
    """Recursively derive prefix-free binary codes for every leaf.

    Convention: left branch → ``"0"``, right branch → ``"1"``.

    Single-leaf (single unique symbol) edge case: the root IS the leaf, so
    ``prefix`` is empty.  We assign ``"0"`` (one bit per occurrence) by
    convention, matching the encoder's special-case handling.
    """
    if node.left is None and node.right is None:
        return {node.symbol: prefix if prefix else "0"}

    codes: dict[int, str] = {}
    if node.left  is not None:
        codes.update(_assign_codes(node.left,  prefix + "0"))
    if node.right is not None:
        codes.update(_assign_codes(node.right, prefix + "1"))
    return codes


def _serialize_freq_table(freq: dict[int, int]) -> bytes:
    """Serialise the frequency table into a portable byte block.

    Format::

        [0–1]          N unique symbols   (uint16, big-endian; max 256)
        Per entry (sorted ascending by symbol value):
            [0]        symbol byte value  (uint8)
            [1–4]      frequency count    (uint32, big-endian)

    Total: 2 + 5·N bytes.  Maximum (all 256 byte values present): 1 282 bytes.
    """
    buf = bytearray()
    buf += len(freq).to_bytes(2, "big")
    for sym in sorted(freq):
        buf += bytes([sym])
        buf += freq[sym].to_bytes(4, "big")
    return bytes(buf)


def _deserialize_freq_table(
    data: bytes, offset: int
) -> tuple[dict[int, int], int]:
    """Parse the serialised frequency table at *offset* within *data*.

    Returns
    -------
    (freq, new_offset)
        *freq*       — reconstructed {symbol: count} mapping.
        *new_offset* — byte position immediately after the table.

    Raises
    ------
    ValueError
        If *data* is truncated before the table ends.
    """
    if offset + 2 > len(data):
        raise ValueError(
            f"Truncated Huffman frequency-table header at byte offset {offset}."
        )
    n       = int.from_bytes(data[offset : offset + 2], "big")
    offset += 2

    needed = n * 5
    if offset + needed > len(data):
        raise ValueError(
            f"Huffman frequency table claims {n} entries ({needed} bytes) "
            f"but only {len(data) - offset} bytes remain after the header."
        )

    freq: dict[int, int] = {}
    for _ in range(n):
        sym        = data[offset]
        count      = int.from_bytes(data[offset + 1 : offset + 5], "big")
        freq[sym]  = count
        offset    += 5

    return freq, offset


# ── Public API ────────────────────────────────────────────────────────────────

def huffman_encode(data: bytes) -> tuple[bytes, int]:
    """Huffman-compress *data*.

    Parameters
    ----------
    data:
        Raw bytes to compress.  May be empty.

    Returns
    -------
    (compressed_bytes, padding_bits)
        *compressed_bytes* — freq_table_block + bit-packed Huffman stream.
        *padding_bits*     — 0–7 trailing zero bits in the last payload byte.
    """
    if not data:
        # Empty input → just the empty-table header, no bit-stream.
        return _serialize_freq_table({}), 0

    freq             = _count_frequencies(data)
    freq_table_bytes = _serialize_freq_table(freq)

    # ── Single unique symbol: assign code "0" (1 bit per occurrence) ──────────
    # A one-leaf tree cannot be traversed downward, so we handle it explicitly.
    if len(freq) == 1:
        sym    = next(iter(freq))
        count  = freq[sym]
        writer = BitWriter()
        for _ in range(count):
            writer.write_bit(0)
        payload_bits, padding = writer.flush()
        return freq_table_bytes + payload_bits, padding

    # ── General case ──────────────────────────────────────────────────────────
    root  = _build_huffman_tree(freq)
    codes = _assign_codes(root)

    writer = BitWriter()
    for byte_val in data:
        writer.write_bits(codes[byte_val])

    payload_bits, padding = writer.flush()
    return freq_table_bytes + payload_bits, padding


def huffman_decode(compressed_bytes: bytes, padding_bits: int) -> bytes:
    """Reconstruct original data from a Huffman-compressed payload.

    Parameters
    ----------
    compressed_bytes:
        Bytes produced by :func:`huffman_encode`.
    padding_bits:
        From ``BinaryHeader.padding_bits``; applies to the bit-stream section.

    Returns
    -------
    bytes
        The original uncompressed data.

    Raises
    ------
    ValueError
        If the payload is structurally inconsistent (truncated table, wrong
        symbol count, corrupt bit-stream).
    """
    if not compressed_bytes:
        return b""

    freq, data_start = _deserialize_freq_table(compressed_bytes, 0)

    if not freq:
        return b""

    total_symbols = sum(freq.values())

    # ── Single unique symbol: no tree traversal required ──────────────────────
    if len(freq) == 1:
        sym = next(iter(freq))
        return bytes([sym] * total_symbols)

    # ── General case: rebuild tree and decode the bit-stream ──────────────────
    root   = _build_huffman_tree(freq)
    reader = BitReader(compressed_bytes[data_start:], padding_bits)

    result:         bytearray     = bytearray()
    current:        _HuffmanNode  = root
    symbols_decoded: int          = 0

    for bit in reader.read_bits():
        # Navigate the tree; left branch = bit 0, right branch = bit 1.
        current = current.left if bit == 0 else current.right

        # Leaf detection: both children are None.
        if current.left is None and current.right is None:
            result.append(current.symbol)
            symbols_decoded += 1
            if symbols_decoded == total_symbols:
                break          # All symbols decoded; stop consuming bits.
            current = root     # Reset traversal pointer for next code-word.

    if symbols_decoded != total_symbols:
        raise ValueError(
            f"Huffman decode produced {symbols_decoded} symbols; "
            f"the frequency table expected {total_symbols}. "
            f"The payload may be corrupt or truncated."
        )

    return bytes(result)


# ══════════════════════════════════════════════════════════════════════════════
# §2  LZ78 CODEC  (Phase 4: dynamic bit-width packing)
# ══════════════════════════════════════════════════════════════════════════════
#
# Lempel-Ziv 1978 is a dictionary-based lossless compression algorithm.
# It parses the input sequentially, building a dynamic dictionary of previously
# seen byte sequences.  Each output token references the longest already-known
# prefix and appends the new extending byte.
#
# Dictionary structure (encoder)
# ───────────────────────────────
#   A trie keyed by (parent_id, new_byte) → child_id.
#   ID 0 is implicitly reserved for the empty string / root of the trie.
#   Lookups are O(1) average (dict of fixed-size int-pair keys).
#
# Phase 4 self-contained payload format  (replaces fixed 6-byte tokens)
# ──────────────────────────────────────
#   ┌───────────────────────────────────────────────────────────────┐
#   │  [0–3]  TOKEN COUNT      (uint32, big-endian)                │
#   │  [4]    INTERNAL PADDING (uint8, 0–7)                        │
#   │           Trailing zero bits in the last byte of the         │
#   │           bit-packed token stream (self-describing).         │
#   │  [5…]   BIT-PACKED TOKEN STREAM                             │
#   │           Per token (written by BitWriter):                  │
#   │             W bits   dictionary index  (W = _lz78_ibits())  │
#   │             1 bit    has_char flag     (0 = sentinel)        │
#   │             8 bits   char value        (only if has_char=1)  │
#   └───────────────────────────────────────────────────────────────┘
#   W is computed dynamically as the dictionary grows:
#     W = max(1, (next_id − 1).bit_length())
#   where next_id is the count of entries added so far (starts at 1).
#   Both encoder and decoder increment the same counter after each token
#   so their bit-width schedules are always in sync.
#
#   The payload is self-describing: external callers always receive
#   padding_bits=0.  The internal padding is stored at byte [4].
#   This design eliminates the chained-codec lost-padding bug.


def _lz78_ibits(next_id: int) -> int:
    """Minimum bits needed to represent dictionary indices 0 .. next_id-1.

    Examples
    --------
    >>> _lz78_ibits(1)   # only index 0 exists → 1 bit
    1
    >>> _lz78_ibits(2)   # indices 0..1 → 1 bit
    1
    >>> _lz78_ibits(3)   # indices 0..2 → 2 bits
    2
    >>> _lz78_ibits(5)   # indices 0..4 → 3 bits
    3
    """
    if next_id <= 2:          # covers next_id==1 (max idx=0) and next_id==2 (max idx=1)
        return 1
    return (next_id - 1).bit_length()


def lz78_encode(data: bytes) -> tuple[bytes, int]:
    """LZ78-compress *data* using dynamic bit-width index packing.

    Algorithm
    ---------
    Uses a trie keyed by ``(parent_id, byte)`` pairs (constant-size keys) for
    O(1) average-case lookups regardless of match depth.

    1. Scan *data* byte-by-byte; extend the current match as long as the
       ``(current_id, byte)`` pair is already in the trie.
    2. On a miss: emit ``(current_id, byte)``, add the new entry to the trie,
       reset ``current_id`` to 0 (root).
    3. At end-of-stream: if ``current_id ≠ 0`` (a match is pending with no
       extending byte), emit ``(current_id, None)`` — the sentinel token.

    Bit packing
    -----------
    Each token is packed as: W bits (index) + 1 bit (has_char) + 8 bits (char).
    W = ``_lz78_ibits(next_id)`` grows as the dictionary expands, minimising
    wasted bits on small dictionaries where indices are always small numbers.

    Parameters
    ----------
    data:
        Raw bytes to compress.  May be empty.

    Returns
    -------
    (compressed_bytes, 0)
        The payload is *self-describing*: it embeds the internal bit-padding
        count at byte [4], so external callers always receive ``padding_bits=0``.
    """
    if not data:
        # 4-byte zero count + 1-byte zero padding → 5 bytes total.
        return (0).to_bytes(4, "big") + b"\x00", 0

    # ── Phase 1: tokenise ─────────────────────────────────────────────────────
    trie: dict[tuple[int, int], int] = {}   # (parent_id, byte) → child_id
    next_id:    int = 1
    current_id: int = 0                      # 0 = root = empty string
    tokens: list[tuple[int, int | None]] = []

    for byte_val in data:
        key = (current_id, byte_val)
        if key in trie:
            current_id = trie[key]
        else:
            tokens.append((current_id, byte_val))
            trie[key] = next_id
            next_id  += 1
            current_id = 0

    # Trailing unfinished match: input ended while a dictionary hit was active.
    if current_id != 0:
        tokens.append((current_id, None))

    # ── Phase 2: bit-pack with dynamic index widths ───────────────────────────
    writer  = BitWriter()
    n_id    = 1        # mirrors next_id at each emission point during encoding

    for idx, char in tokens:
        bit_w = _lz78_ibits(n_id)
        writer.write_bits(format(idx, f"0{bit_w}b"))  # W bits for index
        if char is None:
            writer.write_bit(0)                        # has_char = 0 → sentinel
        else:
            writer.write_bit(1)                        # has_char = 1
            writer.write_bits(format(char, "08b"))     # 8 bits for char
        n_id += 1  # ALWAYS increment — decoder mirrors this unconditionally

    packed_bits, internal_padding = writer.flush()

    # ── Phase 3: assemble self-describing payload ─────────────────────────────
    count_bytes = len(tokens).to_bytes(4, "big")
    padding_byte = bytes([internal_padding])

    return count_bytes + padding_byte + packed_bits, 0
    #                    ^^^^^^^^^^^^
    # External padding is always 0; internal padding is embedded at byte [4].


def lz78_decode(compressed_bytes: bytes, padding_bits: int = 0) -> bytes:
    """Reconstruct the original byte sequence from an LZ78-compressed payload.

    Parameters
    ----------
    compressed_bytes:
        Bytes produced by :func:`lz78_encode`.
    padding_bits:
        Ignored — the payload is self-describing (internal padding is stored
        at byte [4]).  This parameter exists only for API symmetry with the
        Huffman decoder.  Pass ``0`` or omit it.

    Returns
    -------
    bytes
        The original uncompressed data.

    Raises
    ------
    ValueError
        If the payload is truncated or references an unknown dictionary index.
    """
    if not compressed_bytes:
        return b""

    if len(compressed_bytes) < 5:
        raise ValueError(
            f"LZ78 payload too short: need ≥5 bytes for count + padding, "
            f"got {len(compressed_bytes)}."
        )

    n_tokens         = int.from_bytes(compressed_bytes[:4], "big")
    internal_padding = compressed_bytes[4]             # self-describing
    bit_data         = compressed_bytes[5:]

    if n_tokens == 0:
        return b""

    # ── Unpack the bit-stream into a flat list for random access ──────────────
    reader   = BitReader(bit_data, internal_padding)
    bits     = list(reader.read_bits())
    bit_pos  = 0

    def _rbits(n: int) -> int:
        """Read exactly *n* bits from the pre-materialised list."""
        nonlocal bit_pos
        v = 0
        for _ in range(n):
            v = (v << 1) | (bits[bit_pos] if bit_pos < len(bits) else 0)
            bit_pos += 1
        return v

    # ── Reconstruct byte stream ───────────────────────────────────────────────
    dictionary: dict[int, bytes] = {0: b""}   # ID 0 = empty string (root)
    next_id = 1
    result  = bytearray()

    for _ in range(n_tokens):
        bit_w    = _lz78_ibits(next_id)
        idx      = _rbits(bit_w)
        has_char = _rbits(1)
        char     = _rbits(8) if has_char else None

        base = dictionary.get(idx)
        if base is None:
            raise ValueError(
                f"LZ78 decode error: token references unknown dictionary "
                f"index {idx} (highest defined: {next_id - 1})."
            )
        sequence = base + (bytes([char]) if char is not None else b"")
        result  += sequence
        dictionary[next_id] = sequence
        next_id += 1           # mirrors the encoder's n_id increment

    return bytes(result)


# ══════════════════════════════════════════════════════════════════════════════
# §3  CHAINED CODEC  (LZ78 → Huffman)
# ══════════════════════════════════════════════════════════════════════════════
#
# The chained codec is a two-stage pipeline:
#
#   Encode: raw bytes ──LZ78──► intermediate bytes ──Huffman──► payload
#   Decode:    payload ──Huffman──► intermediate bytes ──LZ78──► raw bytes
#
# The LZ78 stage exploits structural redundancy (repeated byte sequences).
# The Huffman stage then exploits frequency skew in the LZ78 token byte-stream
# (e.g., low dictionary indices → many 0x00 bytes in the token stream that
# Huffman can code with very short codes).
#
# Phase 4 note: the Phase 4 LZ78 payload is self-describing (internal padding
# embedded at byte [4]) so lz78_decode(intermediate, 0) is always correct
# regardless of how many internal padding bits were used.  The chained codec
# no longer needs to thread LZ78 padding through the Huffman layer.
#
# The payload format is identical to the Huffman format (freq_table + bit-stream)
# applied to the self-describing LZ78 byte-stream.  The BinaryHeader
# padding_bits field comes from the outer Huffman flush.


def chained_encode(data: bytes) -> tuple[bytes, int]:
    """Two-stage LZ78 → Huffman compression.

    Pipeline::

        data ──lz78_encode()──► intermediate ──huffman_encode()──► payload

    Parameters
    ----------
    data:
        Raw bytes to compress.  May be empty.

    Returns
    -------
    (compressed_bytes, padding_bits)
        *padding_bits* is from the Huffman outer layer.
    """
    intermediate, _ = lz78_encode(data)          # LZ78 is always byte-aligned
    payload, padding = huffman_encode(intermediate)
    return payload, padding


def chained_decode(compressed_bytes: bytes, padding_bits: int) -> bytes:
    """Decode a Chained-compressed payload (Huffman outer, LZ78 inner).

    Pipeline::

        payload ──huffman_decode()──► intermediate ──lz78_decode()──► data

    Parameters
    ----------
    compressed_bytes:
        Payload produced by :func:`chained_encode`.
    padding_bits:
        From ``BinaryHeader.padding_bits`` (Huffman outer layer).

    Returns
    -------
    bytes
        The reconstructed original data.
    """
    intermediate = huffman_decode(compressed_bytes, padding_bits)
    return lz78_decode(intermediate, padding_bits=0)


# ══════════════════════════════════════════════════════════════════════════════
# §4  DISPATCH HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def compress(data: bytes, algorithm_id: int) -> tuple[bytes, int]:
    """Route *data* through the codec identified by *algorithm_id*.

    Parameters
    ----------
    data:
        Raw bytes to compress.  May be empty.
    algorithm_id:
        One of :data:`bit_io.ALGO_HUFFMAN`, :data:`bit_io.ALGO_LZ78`, or
        :data:`bit_io.ALGO_CHAINED`.

    Returns
    -------
    (compressed_bytes, padding_bits)
        Ready to be concatenated after the serialised ``BinaryHeader``.

    Raises
    ------
    ValueError
        If *algorithm_id* is not mapped to an implemented codec.
    """
    if algorithm_id == ALGO_HUFFMAN:
        return huffman_encode(data)
    if algorithm_id == ALGO_LZ78:
        return lz78_encode(data)
    if algorithm_id == ALGO_CHAINED:
        return chained_encode(data)
    if algorithm_id == ALGO_RLE_QUANT:
        raise ValueError(
            "algorithm_id ALGO_RLE_QUANT (0x04) is handled by image_engine, "
            "not codec_engine.  Route image payloads through "
            "image_engine.encode_image() / decode_image()."
        )
    raise ValueError(
        f"Unknown or unimplemented algorithm_id 0x{algorithm_id:02X}.  "
        f"Phase 4 supports: "
        f"HUFFMAN=0x{ALGO_HUFFMAN:02X}, "
        f"LZ78=0x{ALGO_LZ78:02X}, "
        f"CHAINED=0x{ALGO_CHAINED:02X}, "
        f"RLE_QUANT=0x{ALGO_RLE_QUANT:02X} (image_engine only)."
    )


def decompress(payload: bytes, algorithm_id: int, padding_bits: int) -> bytes:
    """Route *payload* through the appropriate decoder.

    Parameters
    ----------
    payload:
        The compressed bytes after the serialised ``BinaryHeader``.
    algorithm_id:
        From ``BinaryHeader.algorithm_id``.
    padding_bits:
        From ``BinaryHeader.padding_bits``.

    Returns
    -------
    bytes
        The reconstructed original data.

    Raises
    ------
    ValueError
        If *algorithm_id* is not mapped, or if the payload is structurally
        invalid.
    """
    if algorithm_id == ALGO_HUFFMAN:
        return huffman_decode(payload, padding_bits)
    if algorithm_id == ALGO_LZ78:
        return lz78_decode(payload, padding_bits)
    if algorithm_id == ALGO_CHAINED:
        return chained_decode(payload, padding_bits)
    if algorithm_id == ALGO_RLE_QUANT:
        raise ValueError(
            "algorithm_id ALGO_RLE_QUANT (0x04) is handled by image_engine, "
            "not codec_engine.  Route image payloads through "
            "image_engine.encode_image() / decode_image()."
        )
    raise ValueError(
        f"Unknown or unimplemented algorithm_id 0x{algorithm_id:02X}.  "
        f"Phase 4 supports: "
        f"HUFFMAN=0x{ALGO_HUFFMAN:02X}, "
        f"LZ78=0x{ALGO_LZ78:02X}, "
        f"CHAINED=0x{ALGO_CHAINED:02X}, "
        f"RLE_QUANT=0x{ALGO_RLE_QUANT:02X} (image_engine only)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# §5  INFORMATION THEORY ANALYTICS  (Phase 5)
# ══════════════════════════════════════════════════════════════════════════════
#
# All metrics are computed in pure Python (math module only; no external
# libraries).  The formulas are taken directly from Shannon (1948).
#
# Shannon Entropy  H(X)
# ─────────────────────
#   Measures the theoretical minimum average number of bits required to
#   encode one symbol from source X.  Computed over the empirical byte
#   distribution of the raw input data:
#
#       H(X) = − Σ  P(x_i) · log₂ P(x_i)
#
#   where P(x_i) = count(x_i) / total_symbols.
#
#   Range:  0 bits/symbol  (all bytes identical)  to
#           8 bits/symbol  (perfectly uniform over all 256 values).
#
# Average Code Length  L
# ────────────────────────
#   The empirical average number of bits spent per input symbol after
#   compression.  Accounts for the padding bits that are stored in the
#   header (not decoded as data):
#
#       total_compressed_bits = len(payload) * 8 − padding_bits
#       L = total_compressed_bits / len(original)
#
#   Padding bits are subtracted because they carry no information.
#   For LZ78 the external padding is always 0 (self-describing payload),
#   so the formula simplifies to  L = len(payload)*8 / len(original).
#
# Coding Efficiency  η
# ─────────────────────
#   How close the codec is to the Shannon limit:
#
#       η = (H / L) × 100 %
#
#   η = 100 %  means the codec achieves the theoretical minimum (rarely
#               attainable in practice due to integer rounding and headers).
#   η > 100 %  is impossible for a valid lossless codec.
#   η can exceed 100 % in the UI display only when the entropy estimate is
#   slightly higher than L due to very small files; we clamp to 100 % when
#   displaying.


@dataclass(frozen=True)
class InformationTheoryMetrics:
    """Container for the three canonical information-theory codec metrics.

    All fields are pre-computed and immutable.  Pass an instance directly
    to the Streamlit display layer.

    Attributes
    ----------
    entropy : float
        Shannon entropy H(X) in bits per symbol (byte).  Range [0.0, 8.0].
    avg_code_length : float
        Average number of compressed bits spent per input symbol.
        Computed as ``(payload_bits − padding_bits) / n_symbols``.
    efficiency : float
        Coding efficiency η = H/L × 100, expressed as a percentage.
        Values above 100 % are clamped to 100 % for display purposes.
    n_symbols : int
        Total number of source symbols (input bytes).
    n_unique : int
        Number of distinct byte values that appear in the source.
    """

    entropy:        float
    avg_code_length: float
    efficiency:     float
    n_symbols:      int
    n_unique:       int


def shannon_entropy(data: bytes) -> float:
    """Compute the Shannon entropy of *data* in bits per symbol.

    Formula
    -------
    ::

        H(X) = − Σ  P(x_i) · log₂ P(x_i)

    where ``P(x_i) = count(x_i) / len(data)``.

    Parameters
    ----------
    data : bytes
        The raw source byte array.  May be empty.

    Returns
    -------
    float
        Entropy in bits per symbol.  Returns ``0.0`` for empty or
        single-valued inputs (zero uncertainty).

    Notes
    -----
    * The result is bounded by ``[0.0, 8.0]`` for byte alphabets.
    * Uses ``math.log2`` for numerical accuracy.
    """
    n = len(data)
    if n == 0:
        return 0.0

    # Build frequency table in pure Python (no numpy required here).
    freq: dict[int, int] = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1

    h = 0.0
    for count in freq.values():
        p = count / n
        h -= p * math.log2(p)     # − P(x_i) · log₂ P(x_i)

    return h


def average_code_length(
    data:          bytes,
    compressed:    bytes,
    padding_bits:  int = 0,
) -> float:
    """Compute the average number of compressed bits per input symbol.

    Formula
    -------
    ::

        total_bits = len(compressed) * 8 − padding_bits
        L = total_bits / len(data)

    Padding bits are subtracted because they are structural (boundary
    alignment), not information-carrying.

    Parameters
    ----------
    data : bytes
        The original uncompressed input.
    compressed : bytes
        The compressed payload bytes (header-free; only the codec output).
    padding_bits : int, optional
        Number of trailing zero bits in the last byte of *compressed* that
        carry no information (from ``BinaryHeader.padding_bits``).  Default 0.

    Returns
    -------
    float
        Average code length L in bits per symbol.  Returns ``0.0`` if *data*
        is empty.
    """
    n = len(data)
    if n == 0:
        return 0.0

    total_bits = len(compressed) * 8 - padding_bits
    return total_bits / n


def coding_efficiency(entropy: float, avg_code_len: float) -> float:
    """Compute coding efficiency η = (H / L) × 100.

    Parameters
    ----------
    entropy : float
        Shannon entropy H in bits per symbol.
    avg_code_len : float
        Average code length L in bits per symbol.

    Returns
    -------
    float
        Efficiency as a percentage.  Returns ``0.0`` if *avg_code_len* is
        zero (to avoid division by zero on empty inputs).  May theoretically
        exceed 100 % due to finite-sample entropy estimation; callers should
        clamp to 100 % for display.
    """
    if avg_code_len <= 0.0:
        return 0.0
    return (entropy / avg_code_len) * 100.0


def analyze_source(
    data:         bytes,
    compressed:   bytes,
    padding_bits: int = 0,
) -> InformationTheoryMetrics:
    """One-shot computation of all three information-theory metrics.

    Runs :func:`shannon_entropy`, :func:`average_code_length`, and
    :func:`coding_efficiency` and packages the results into an immutable
    :class:`InformationTheoryMetrics` dataclass.

    Parameters
    ----------
    data : bytes
        Original uncompressed source data.
    compressed : bytes
        The compressed payload bytes (codec output, header-free).
    padding_bits : int, optional
        Trailing padding bits in the last byte of *compressed*.  Default 0.

    Returns
    -------
    InformationTheoryMetrics
        Frozen dataclass with ``entropy``, ``avg_code_length``,
        ``efficiency``, ``n_symbols``, and ``n_unique`` fields.

    Examples
    --------
    >>> m = analyze_source(b"AAABBC", b"\x00", 0)
    >>> 0.0 <= m.entropy <= 8.0
    True
    >>> m.n_symbols
    6
    """
    H = shannon_entropy(data)
    L = average_code_length(data, compressed, padding_bits)
    eta = coding_efficiency(H, L)

    # Count unique byte values.
    freq: dict[int, int] = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1

    return InformationTheoryMetrics(
        entropy         = H,
        avg_code_length = L,
        efficiency      = eta,
        n_symbols       = len(data),
        n_unique        = len(freq),
    )
