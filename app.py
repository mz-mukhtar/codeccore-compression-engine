"""
CodecCore: Data Compression Engine
app.py — Streamlit application coordinator (Phase 6)

Architecture
────────────
  bit_io.py        — Bit-level primitives + BinaryHeader / AudioHeader protocol.
  codec_engine.py  — Huffman, LZ78, Chained codecs + Information Theory analytics.
  image_engine.py  — Lossy image pipeline (grayscale, quantization, RLE) + SP analytics.
  audio_engine.py  — PCM, DM, ADM, DPCM waveform coding + SQNR metrics.
  app.py           — Streamlit UI (this file).
"""

from __future__ import annotations

import io as _io

import numpy as np
import streamlit as st

import codec_engine
import image_engine
import audio_engine
from image_engine import IMAGE_EXTENSIONS

from bit_io import (
    BinaryHeader,
    AudioHeader,
    HEADER_FIXED_SIZE,
    AUDIO_PARAM_SIZE,
    MAGIC,
    FILE_TYPE_PICTURE,
    FILE_TYPE_BINARY,
    FILE_TYPE_AUDIO,
    FILE_TYPE_LABELS,
    ALGORITHM_LABELS,
    ALGORITHM_IDS,
    ALGO_HUFFMAN,
    ALGO_LZ78,
    ALGO_CHAINED,
    ALGO_RLE_QUANT,
    ALGO_PCM,
    ALGO_DM,
    ALGO_ADM,
    ALGO_DPCM,
)
from codec_engine import analyze_source, InformationTheoryMetrics
from image_engine import analyze_image_error, ImageErrorMetrics, get_image_info
from audio_engine import (
    load_audio_to_array,
    export_array_to_format,
    mix_to_mono,
    encode_audio,
    decode_audio,
    compute_sqnr,
    compute_theoretical_sqnr,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CodecCore: Data Compression Engine",
    page_icon="🗜️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .block-container { padding-top: 1.5rem; }

        /* Download buttons */
        [data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #1a6fad, #0d4f82);
            color: #fff; border: none; font-weight: 600;
        }
        [data-testid="stDownloadButton"] > button:hover {
            background: linear-gradient(135deg, #1e7fc0, #155d94); color: #fff;
        }

        /* ARCH header status board */
        .header-board {
            background: linear-gradient(135deg, #0d1b2a 0%, #1b2e45 100%);
            border: 1px solid #1a6fad; border-radius: 10px;
            padding: 1.2rem 1.5rem; margin: 0.75rem 0;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.90rem; line-height: 2.1; color: #e8f0fe;
        }
        .header-board .label { color: #7ecfff; font-weight: 700;
                                display: inline-block; min-width: 250px; }
        .header-board .magic { color: #6be56b; font-weight: 800; letter-spacing: 0.06em; }
        .header-board .algo  { color: #ffcf70; font-weight: 600; }
        .header-board .ext   { color: #c084fc; font-weight: 600; }
        .header-board .pad   { color: #ff9f7e; font-weight: 600; }
        .header-board .sub   { color: #94a3b8; font-size: 0.80rem; }

        /* Analytics panels */
        .analytics-panel { border-radius: 10px; padding: 1.25rem 1.5rem 1rem; margin: 0.5rem 0 1.25rem; }
        .analytics-panel.it {
            background: linear-gradient(135deg, #0a1929 0%, #0d2137 60%, #102740 100%);
            border: 1px solid #1e6fa8;
        }
        .analytics-panel.sp {
            background: linear-gradient(135deg, #0f1a0f 0%, #132013 60%, #182818 100%);
            border: 1px solid #1e8a4a;
        }
        .analytics-panel.audio {
            background: linear-gradient(135deg, #1a0f29 0%, #251340 60%, #2a1847 100%);
            border: 1px solid #7c3aed;
        }
        .analytics-panel h4 { margin-top: 0; margin-bottom: 0.75rem;
                               font-size: 0.95rem; letter-spacing: 0.05em; text-transform: uppercase; }
        .analytics-panel.it h4    { color: #7ecfff; }
        .analytics-panel.sp h4    { color: #6be59e; }
        .analytics-panel.audio h4 { color: #c084fc; }

        /* Image / audio pipeline box */
        .pipeline-box {
            border-radius: 8px; padding: 1rem 1.25rem; margin: 0.5rem 0 1rem 0; font-size: 0.88rem;
        }
        .pipeline-box.img {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            border: 1px solid #0ea5e9;
        }
        .pipeline-box.audio {
            background: linear-gradient(135deg, #1a0929 0%, #2d1245 50%, #3b1860 100%);
            border: 1px solid #7c3aed;
        }

        /* Ratio / PSNR colours */
        .ratio-good { color: #4ade80; font-weight: 700; }
        .ratio-bad  { color: #f87171; font-weight: 700; }
        .psnr-inf   { color: #6be59e; font-weight: 800; }
        .psnr-exc   { color: #4ade80; font-weight: 700; }
        .psnr-good  { color: #a3e635; font-weight: 700; }
        .psnr-fair  { color: #facc15; font-weight: 700; }
        .psnr-poor  { color: #fb923c; font-weight: 700; }
        .psnr-bad   { color: #f87171; font-weight: 700; }

        /* Theory tab */
        .theory-section {
            background: linear-gradient(135deg, #0d1b2a 0%, #112233 100%);
            border: 1px solid #1e4d70; border-radius: 10px;
            padding: 1.5rem 2rem; margin: 1rem 0 1.5rem;
        }
        .theory-section h3 { color: #7ecfff; margin-top: 0; }
        .theory-section.audio { border-color: #7c3aed; }
        .theory-section.audio h3 { color: #c084fc; }
        .insight-box {
            background: #060e18; border-left: 4px solid #4ade80;
            border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
            color: #a7f3d0; font-size: 0.90rem; margin: 0.75rem 0;
        }
        .insight-box.audio { border-color: #7c3aed; background: #0e0618; color: #d8b4fe; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Algorithm lists
# ─────────────────────────────────────────────────────────────────────────────
_TEXT_ALGORITHMS: list[str] = [
    ALGORITHM_LABELS[ALGO_HUFFMAN],
    ALGORITHM_LABELS[ALGO_LZ78],
    ALGORITHM_LABELS[ALGO_CHAINED],
]
_AUDIO_ALGORITHMS: list[str] = [
    ALGORITHM_LABELS[ALGO_PCM],
    ALGORITHM_LABELS[ALGO_DM],
    ALGORITHM_LABELS[ALGO_ADM],
    ALGORITHM_LABELS[ALGO_DPCM],
]

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _size_metrics(raw: bytes) -> dict:
    n = len(raw)
    return {"bytes": n, "kb": n / 1_024, "mb": n / (1_024 ** 2)}

def _extract_extension(filename: str) -> str:
    dot = filename.rfind(".")
    return "" if dot == -1 or dot == len(filename) - 1 else filename[dot + 1:].lower()

def _extract_stem(filename: str) -> str:
    dot = filename.rfind(".")
    return filename if dot == -1 else filename[:dot]

def _abc_stem(filename: str) -> str:
    return filename[:-4] if filename.lower().endswith(".abc") else _extract_stem(filename)

def _is_image(ext: str) -> bool:
    return ext in IMAGE_EXTENSIONS

def _is_audio(ext: str) -> bool:
    return ext in ["wav", "mp3", "flac", "m4a", "ogg"]

def _format_ratio(original: int, compressed: int) -> tuple[str, str, str]:
    if original == 0:
        return "N/A", "N/A", ""
    ratio   = compressed / original
    savings = (1.0 - ratio) * 100.0
    sign    = "-" if savings >= 0 else "+"
    css     = "ratio-good" if ratio < 1.0 else "ratio-bad"
    return f"{ratio:.4f}×", f"{sign}{abs(savings):.2f}%", css

def _psnr_css(psnr: float | None) -> str:
    if psnr is None: return "psnr-inf"
    if psnr >= 40:   return "psnr-exc"
    if psnr >= 35:   return "psnr-good"
    if psnr >= 30:   return "psnr-fair"
    if psnr >= 25:   return "psnr-poor"
    return "psnr-bad"

def _psnr_label(psnr: float | None) -> str:
    if psnr is None: return "Lossless (∞)"
    if psnr >= 40:   return "Excellent"
    if psnr >= 35:   return "Very Good"
    if psnr >= 30:   return "Good"
    if psnr >= 25:   return "Acceptable"
    return "Distorted"

def _sqnr_label(sqnr: float) -> str:
    if not isinstance(sqnr, float) or sqnr == float("inf"):
        return "Lossless"
    if sqnr >= 40:  return "Excellent"
    if sqnr >= 30:  return "Good"
    if sqnr >= 20:  return "Acceptable"
    return "Distorted"

# ─────────────────────────────────────────────────────────────────────────────
# Analytics display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_it_panel(metrics: InformationTheoryMetrics, algo_label: str) -> None:
    eta_display = min(metrics.efficiency, 100.0)
    st.markdown(
        f'<div class="analytics-panel it"><h4>🔬 Shannon Channel Analysis — {algo_label}</h4>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Source:** {metrics.n_symbols:,} symbols · **Alphabet:** {metrics.n_unique}/256 unique values"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Source Entropy  H(X)", f"{metrics.entropy:.4f} bits/sym",
              help="H(X) = −Σ P(xᵢ)·log₂P(xᵢ). Theoretical minimum code length.")
    c2.metric("Avg Code Length  L", f"{metrics.avg_code_length:.4f} bits/sym",
              delta=f"{metrics.avg_code_length - metrics.entropy:+.4f} above floor",
              delta_color="inverse")
    c3.metric("Coding Efficiency  η", f"{eta_display:.2f} %",
              delta=f"{eta_display:.2f}% of Shannon limit", delta_color="off")
    gap = metrics.avg_code_length - metrics.entropy
    if eta_display >= 95.0:   icon, msg = "🟢", "Near-optimal."
    elif eta_display >= 80.0: icon, msg = "🟡", "Good efficiency; moderate redundancy."
    elif eta_display >= 60.0: icon, msg = "🟠", "Moderate efficiency."
    else:                     icon, msg = "🔴", "Low efficiency; significant redundancy."
    st.markdown(f"{icon} **{msg}**  Codec uses **{gap:.4f} bits/sym** above the Shannon floor.")


def _render_sp_panel(img_metrics: ImageErrorMetrics, info: dict) -> None:
    psnr_val  = img_metrics.psnr
    pcss      = _psnr_css(psnr_val)
    plabel    = _psnr_label(psnr_val)
    ratio_pct = (1.0 - img_metrics.compression_ratio) * 100.0
    ch_label  = "Grayscale (1-ch)" if info["channels"] == 1 else "RGB (3-ch)"
    qb        = info["quantize_bits"]
    st.markdown(
        f'<div class="analytics-panel sp"><h4>📊 Pixel-Domain Distortion Metrics</h4>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Mode:** {ch_label} · **Bit-depth:** {qb}-bit ({2**qb} levels/channel) · "
        f"**Pixels compared:** {img_metrics.n_pixels:,}"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE", f"{img_metrics.mse:.4f}" if not img_metrics.is_lossless else "0.0000",
              delta="Lossless ✓" if img_metrics.is_lossless else f"{img_metrics.mse:.4f}",
              delta_color="off" if img_metrics.is_lossless else "inverse")
    c2.metric("PSNR  (dB)", img_metrics.psnr_str,
              delta=f"Quality: {plabel}", delta_color="off")
    c3.metric("Compression Ratio", f"{img_metrics.compression_ratio:.4f}×",
              delta=f"{ratio_pct:+.2f}% reduction" if ratio_pct > 0 else f"{-ratio_pct:.2f}% increase",
              delta_color="inverse" if ratio_pct > 0 else "normal")
    st.markdown(
        f'<span class="{pcss}">▶ PSNR Quality: {plabel}</span>'
        + (" — Lossless round-trip." if img_metrics.is_lossless else f" — PSNR = {img_metrics.psnr_str}."),
        unsafe_allow_html=True,
    )


def _render_audio_panel(
    sqnr_actual:      float,
    sqnr_theoretical: float | None,
    original:         np.ndarray,
    reconstructed:    np.ndarray,
    sample_rate:      int,
    algo_label:       str,
    payload_bytes:    int,
    original_bytes:   int,
) -> None:
    """Render the audio Signal Processing Analytics panel."""
    sqnr_str  = f"{sqnr_actual:.2f} dB" if sqnr_actual != float("inf") else "∞ (Lossless)"
    sqnr_lbl  = _sqnr_label(sqnr_actual)
    ratio_str, savings_str, css = _format_ratio(original_bytes, payload_bytes)

    st.markdown(
        f'<div class="analytics-panel audio"><h4>🎵 Audio Signal Quality — {algo_label}</h4>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Samples compared:** {min(len(original), len(reconstructed)):,} · "
        f"**Sample rate:** {sample_rate:,} Hz"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "SQNR (Signal-to-Quantization-Noise)",
        sqnr_str,
        delta=f"Quality: {sqnr_lbl}",
        delta_color="off",
        help="SQNR = 10·log₁₀(Σx²[n] / Σe²[n]). Higher is better.",
    )
    if sqnr_theoretical is not None:
        c2.metric(
            "Theoretical SQNR (6.02b + 1.76 dB)",
            f"{sqnr_theoretical:.2f} dB",
            delta=f"{sqnr_actual - sqnr_theoretical:+.2f} dB vs theory" if sqnr_actual != float("inf") else "N/A",
            delta_color="off",
            help="For uniform b-bit PCM: SQNR ≈ 6.02b + 1.76 dB (Bennet's formula).",
        )
    else:
        c2.metric("Theoretical SQNR", "N/A (DM / ADM)",
                  help="The 6.02b formula applies to uniform PCM only.")
    c3.metric(
        "Compression Ratio",
        ratio_str,
        delta=savings_str + " saved",
        delta_color="inverse",
    )

    # SQNR quality band
    if sqnr_actual == float("inf"):
        band_icon, band_msg = "🟢", "Lossless reconstruction."
    elif sqnr_actual >= 40:
        band_icon, band_msg = "🟢", "Excellent quality — indistinguishable from original."
    elif sqnr_actual >= 30:
        band_icon, band_msg = "🟡", "Good quality — minor quantization noise."
    elif sqnr_actual >= 20:
        band_icon, band_msg = "🟠", "Acceptable — noticeable distortion."
    else:
        band_icon, band_msg = "🔴", "Low quality — heavy quantization distortion."
    st.markdown(f"{band_icon} **{band_msg}**")


# ─────────────────────────────────────────────────────────────────────────────
# Application header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🗜️ CodecCore: Data Compression Engine")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_compress, tab_decompress, tab_theory = st.tabs([
    "🔵  Compress Data",
    "🟢  Decompress Data",
    "📖  Theory & Simulations",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Compress Data
# ══════════════════════════════════════════════════════════════════════════════
with tab_compress:
    st.subheader("Compress Data", divider="blue")

    uploaded_compress = st.file_uploader(
        "Choose a file to compress",
        accept_multiple_files=False,
        type=None,
        key="compress_uploader",
        help="Image files → RLE + Quantization.  Audio files (.wav, .mp3, .flac, .m4a, .ogg) → PCM/DM/ADM/DPCM.  All others → Huffman/LZ78/Chained.",
    )

    if uploaded_compress is not None:
        raw_compress: bytes = uploaded_compress.getvalue()
        sm         = _size_metrics(raw_compress)
        extension  = _extract_extension(uploaded_compress.name)
        base_name  = _extract_stem(uploaded_compress.name)
        is_img     = _is_image(extension)
        is_aud     = _is_audio(extension)

        compress_error:  str   = ""
        payload_bytes:   bytes = b""
        padding_bits:    int   = 0
        algorithm_id:    int   = ALGO_HUFFMAN
        file_type_id:    int   = FILE_TYPE_BINARY
        it_metrics:      InformationTheoryMetrics | None = None
        sp_metrics:      ImageErrorMetrics | None        = None
        img_info:        dict | None                     = None
        audio_hdr_kw:    dict                            = {}   # AudioHeader kwargs

        # ── Branch A — Audio pipeline ─────────────────────────────────────────
        if is_aud:
            st.markdown("#### 🎵 Audio File Detected — Waveform Coding Pipeline")
            st.markdown(
                '<div class="pipeline-box audio"><b>Pipeline:</b>&nbsp; '
                "WAV Load → Mono Mix → Downsample → Quantize / Predict → Bit-Pack"
                "</div>",
                unsafe_allow_html=True,
            )

            try:
                sample_rate, mono_signal = load_audio_to_array(raw_compress, extension)
                n_channels_src = 1  # Audio is automatically mixed to mono during ingestion
            except Exception as exc:
                st.error(f"Could not read Audio file: {exc}", icon="🚫")
                st.stop()

            original_length = len(mono_signal)
            st.success(
                f"✅  **{uploaded_compress.name}** — {sample_rate:,} Hz · "
                f"{n_channels_src}ch · {original_length:,} samples"
            )

            # ── Parameters ────────────────────────────────────────────────────
            st.markdown("#### ⚙️ Codec Settings")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                aud_algo_label = st.selectbox(
                    "Coding Algorithm",
                    options=_AUDIO_ALGORITHMS,
                    index=0,
                    key="audio_algo",
                )
            with pcol2:
                sf_options = {f"1× (No Downsample)": 1, "2× Downsample": 2, "4× Downsample": 4}
                sf_label   = st.selectbox(
                    "Sampling Factor",
                    options=list(sf_options.keys()),
                    index=0,
                    key="audio_sf",
                    help="Only applied for PCM. DM/ADM/DPCM encode every sample.",
                )
                sampling_factor = sf_options[sf_label]

            algorithm_id = ALGORITHM_IDS[aud_algo_label]
            is_dm_adm    = algorithm_id in (ALGO_DM, ALGO_ADM)
            is_dpcm      = algorithm_id == ALGO_DPCM
            is_pcm       = algorithm_id == ALGO_PCM

            qcol1, qcol2 = st.columns(2)
            with qcol1:
                quant_bits = st.slider(
                    "Quantization Bit-Depth",
                    min_value=1, max_value=8, value=8, step=1,
                    key="audio_bits",
                    help="Only applied for PCM and DPCM. DM/ADM use 1-bit (fixed).",
                    disabled=is_dm_adm,
                )
            with qcol2:
                delta_step = st.slider(
                    "Delta Step Size (Δ)",
                    min_value=0.001, max_value=0.200, value=0.050, step=0.001,
                    format="%.3f",
                    key="audio_delta",
                    help="Step size for DM and ADM.  Larger Δ reduces slope overload but increases granular noise.",
                    disabled=not is_dm_adm,
                )

            alpha_val: float = 0.9
            if is_dpcm:
                alpha_val = st.slider(
                    "Predictor Weight (α)",
                    min_value=0.50, max_value=0.99, value=0.90, step=0.01,
                    format="%.2f",
                    key="audio_alpha",
                    help="First-order predictor coefficient for DPCM.  α ≈ 0.9 is typical for speech.",
                )

            if is_pcm and quant_bits < 8:
                th_sqnr = compute_theoretical_sqnr(quant_bits)
                st.info(
                    f"ℹ️  **Theoretical SQNR** for {quant_bits}-bit PCM: "
                    f"**{th_sqnr:.2f} dB** (6.02 × {quant_bits} + 1.76 dB).",
                    icon="📐",
                )

            file_type_id = FILE_TYPE_AUDIO

            # ── Encode ────────────────────────────────────────────────────────
            with st.spinner(f"Running {aud_algo_label} encoder …"):
                try:
                    payload_bytes, padding_bits = encode_audio(
                        signal          = mono_signal,
                        algorithm_id    = algorithm_id,
                        sampling_factor = sampling_factor if is_pcm else 1,
                        bits            = quant_bits,
                        delta_step      = delta_step,
                        multiplier      = 1.5,
                        alpha           = alpha_val,
                    )
                    # SQNR: decode immediately for metrics
                    recon_signal = decode_audio(
                        payload         = payload_bytes,
                        padding_bits    = padding_bits,
                        algorithm_id    = algorithm_id,
                        original_length = original_length,
                        bits            = quant_bits,
                        delta_step      = delta_step,
                        multiplier      = 1.5,
                        alpha           = alpha_val,
                        sampling_factor = sampling_factor if is_pcm else 1,
                    )
                except Exception as exc:
                    compress_error = str(exc)

            # AudioHeader kwargs stored for archive assembly
            audio_hdr_kw = dict(
                algorithm_id    = algorithm_id,
                padding_bits    = padding_bits,
                extension       = extension,
                sampling_factor = sampling_factor if is_pcm else 1,
                bits            = quant_bits,
                delta_step      = delta_step if is_dm_adm else 0.0,
                alpha           = alpha_val if is_dpcm else 0.0,
                original_length = original_length,
                sample_rate     = sample_rate,
                n_channels      = 1,
            )

        # ── Branch B — Image pipeline ─────────────────────────────────────────
        elif is_img:
            st.markdown("#### 🖼️ Image File Detected — RLE + Quantization Pipeline")
            try:
                st.image(raw_compress, caption=uploaded_compress.name,
                         use_container_width=True)
            except Exception:
                st.warning("⚠️  Could not render image preview.")

            st.markdown(
                '<div class="pipeline-box img"><b>Pipeline:</b>&nbsp; '
                "Grayscale (optional) → Spatial Quantization (optional) → RLE Encoding"
                "</div>",
                unsafe_allow_html=True,
            )

            col_gray, col_quant = st.columns(2)
            with col_gray:
                apply_gray = st.checkbox("Grayscale Conversion (Lossy)", value=False,
                                         key="gray_check")
            with col_quant:
                quant_bits = st.slider("Bit-Depth per Channel", 1, 8, 8, 1,
                                       key="quant_slider")

            if quant_bits < 8 or apply_gray:
                parts = []
                if apply_gray:     parts.append("Grayscale")
                if quant_bits < 8: parts.append(f"{quant_bits}-bit Quantization ({2**quant_bits} levels)")
                st.info("⚠️  **Lossy mode:** " + " + ".join(parts) + ".", icon="⚠️")

            algorithm_id = ALGO_RLE_QUANT
            file_type_id = FILE_TYPE_PICTURE

            with st.spinner("Encoding image …"):
                try:
                    payload_bytes, padding_bits = image_engine.encode_image(
                        raw_compress, apply_gray, quant_bits
                    )
                    _recon = image_engine.decode_image(payload_bytes, padding_bits)
                    img_info = get_image_info(payload_bytes)
                    sp_metrics = analyze_image_error(
                        original_bytes=raw_compress,
                        reconstructed_bytes=_recon,
                        width=img_info["width"], height=img_info["height"],
                        is_grayscale=(img_info["channels"] == 1),
                        compressed_payload=payload_bytes,
                    )
                except Exception as exc:
                    compress_error = str(exc)

        # ── Branch C — General codec pipeline ─────────────────────────────────
        else:
            st.markdown("#### ⚙️ Algorithm Settings")
            algorithm_label = st.selectbox(
                "Compression Algorithm", options=_TEXT_ALGORITHMS,
                index=0, key="compress_algorithm",
            )
            algorithm_id = ALGORITHM_IDS[algorithm_label]
            file_type_id = FILE_TYPE_BINARY

            with st.spinner(f"Compressing with {algorithm_label} …"):
                try:
                    payload_bytes, padding_bits = codec_engine.compress(
                        raw_compress, algorithm_id
                    )
                    it_metrics = analyze_source(
                        data=raw_compress, compressed=payload_bytes,
                        padding_bits=padding_bits,
                    )
                except Exception as exc:
                    compress_error = str(exc)

        # ── Common: header, metrics, expander, download ───────────────────────
        if compress_error:
            st.error(f"Compression failed: {compress_error}", icon="🚫")
        else:
            # Build archive
            if is_aud:
                header_obj  = AudioHeader(**audio_hdr_kw)
            else:
                header_obj  = BinaryHeader(
                    file_type=file_type_id, algorithm_id=algorithm_id,
                    padding_bits=padding_bits,
                    extension=extension,
                )
            header_bytes: bytes = header_obj.serialize()
            final_file:   bytes = header_bytes + payload_bytes

            original_sz = sm["bytes"]
            payload_sz  = len(payload_bytes)
            archive_sz  = len(final_file)
            ratio_str, savings_str, css = _format_ratio(original_sz, payload_sz)

            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Original Size",    f"{original_sz:,} B")
            r2.metric("Compressed Size",  f"{payload_sz:,} B",
                      delta=f"{payload_sz - original_sz:+,} B", delta_color="inverse")
            r3.metric("Compression Ratio", ratio_str,
                      delta=savings_str + " saved", delta_color="inverse")

            st.success(
                f"✅  **{uploaded_compress.name}** → **{base_name}_compressed.abc** "
                f"| {original_sz:,} B → {payload_sz:,} B | Ratio: {ratio_str}",
                icon="✅",
            )

            # Progressive disclosure
            with st.expander("📊 View Advanced Mathematical Analytics", expanded=False):
                if it_metrics is not None:
                    st.subheader("📐 Information Theory Analytics")
                    _render_it_panel(it_metrics, ALGORITHM_LABELS[algorithm_id])

                if sp_metrics is not None and img_info is not None:
                    st.subheader("📡 Signal Processing Analytics")
                    _render_sp_panel(sp_metrics, img_info)

                if is_aud and "recon_signal" in dir():
                    st.subheader("🎵 Audio Signal Quality Analytics")
                    th = (compute_theoretical_sqnr(quant_bits)
                          if algorithm_id == ALGO_PCM else None)
                    _render_audio_panel(
                        sqnr_actual      = compute_sqnr(mono_signal, recon_signal),
                        sqnr_theoretical = th,
                        original         = mono_signal,
                        reconstructed    = recon_signal,
                        sample_rate      = sample_rate,
                        algo_label       = aud_algo_label,
                        payload_bytes    = payload_sz,
                        original_bytes   = original_sz,
                    )

                st.markdown("---")
                st.markdown("##### 🔩 Archive Internals")
                hd1, hd2, hd3, hd4 = st.columns(4)
                hd1.metric("Header",       f"{len(header_bytes)} B")
                hd2.metric("Payload",      f"{payload_sz:,} B")
                hd3.metric("Padding Bits", str(padding_bits))
                hd4.metric("Archive Total", f"{archive_sz:,} B")
                st.code(
                    "ARCH Header: " + " ".join(f"0x{b:02X}" for b in header_bytes),
                    language="text"
                )

            st.divider()
            download_name = f"{base_name}_compressed.abc"
            st.download_button(
                label     = f"⬇️  Download  {download_name}",
                data      = final_file,
                file_name = download_name,
                mime      = "application/octet-stream",
                key       = "compress_download",
            )
            st.caption(
                f"Self-describing ARCH archive — the {len(header_bytes)}-byte header "
                f"encodes algorithm, file type, extension"
                + (", and all audio encoding parameters." if is_aud else ", and bit-padding.")
            )

    else:
        st.info("📂 Upload a file to get started.", icon="📂")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Decompress Data
# ══════════════════════════════════════════════════════════════════════════════
with tab_decompress:
    st.subheader("Decompress Data", divider="green")

    uploaded_decompress = st.file_uploader(
        "Choose a .abc archive to decompress",
        accept_multiple_files=False, type=None, key="decompress_uploader",
    )

    if uploaded_decompress is not None:
        raw_decompress: bytes = uploaded_decompress.getvalue()
        sm_d = _size_metrics(raw_decompress)

        if len(raw_decompress) < HEADER_FIXED_SIZE:
            st.error(
                f"File too small ({len(raw_decompress)} B) to contain an ARCH header.",
                icon="🚫"
            )
            st.stop()

        # ── Peek at the file-type byte before dispatching ─────────────────────
        file_type_byte   = raw_decompress[4] if len(raw_decompress) > 4 else 0xFF
        is_audio_archive = (file_type_byte == FILE_TYPE_AUDIO)

        parsed_header = None
        audio_hdr     = None
        parse_error   = ""

        if is_audio_archive:
            try:
                audio_hdr = AudioHeader.deserialize_audio(raw_decompress)
                parsed_header = audio_hdr
            except ValueError as exc:
                parse_error = str(exc)
        else:
            try:
                parsed_header = BinaryHeader.deserialize(raw_decompress)
            except ValueError as exc:
                parse_error = str(exc)

        if parse_error or parsed_header is None:
            st.error(f"Invalid ARCH archive.  **Detail:** {parse_error}", icon="🚫")
            st.stop()

        algo_disp      = ALGORITHM_LABELS.get(parsed_header.algorithm_id, f"0x{parsed_header.algorithm_id:02X}")
        file_type_disp = FILE_TYPE_LABELS.get(parsed_header.file_type, "Unknown")
        ext_disp       = parsed_header.extension if parsed_header.extension else "(none)"
        payload_start  = parsed_header.total_size
        payload_sz_d   = len(raw_decompress) - payload_start
        logical_bits   = max(0, payload_sz_d * 8 - parsed_header.padding_bits)

        # ── Status board ──────────────────────────────────────────────────────
        audio_extra = ""
        if is_audio_archive and audio_hdr is not None:
            audio_extra = (
                f"<div><span class='label'>🎵 Sample Rate:</span>"
                f"<span class='value'>{audio_hdr.sample_rate:,} Hz</span></div>"
                f"<div><span class='label'>📏 Original Samples:</span>"
                f"<span class='value'>{audio_hdr.original_length:,}</span></div>"
                f"<div><span class='label'>🔢 Bit-Depth:</span>"
                f"<span class='value'>{audio_hdr.bits}</span></div>"
                f"<div><span class='label'>⬇️ Sampling Factor:</span>"
                f"<span class='value'>{audio_hdr.sampling_factor}×</span></div>"
                + (f"<div><span class='label'>Δ Step Size:</span>"
                   f"<span class='value'>{audio_hdr.delta_step:.4f}</span></div>"
                   if audio_hdr.algorithm_id in (ALGO_DM, ALGO_ADM) else "")
                + (f"<div><span class='label'>α Predictor:</span>"
                   f"<span class='value'>{audio_hdr.alpha:.4f}</span></div>"
                   if audio_hdr.algorithm_id == ALGO_DPCM else "")
            )

        st.markdown(
            f"""
            <div class="header-board">
                <div><span class="label">🔑 Magic Bytes:</span>
                    <span class="magic">ARCH</span>
                    <span class="sub">&nbsp;(0x41 0x52 0x43 0x48)</span></div>
                <div><span class="label">⚙️ Algorithm:</span>
                    <span class="algo">{algo_disp}
                        <span class="sub">(ID 0x{parsed_header.algorithm_id:02X})</span>
                    </span></div>
                <div><span class="label">📎 Original Extension:</span>
                    <span class="ext">.{ext_disp}</span></div>
                <div><span class="label">📦 Payload:</span>
                    <span class="value">{payload_sz_d:,} bytes
                        <span class="sub">→ {logical_bits:,} logical bits</span>
                    </span></div>
                <div><span class="label">🔲 Padding Bits:</span>
                    <span class="pad">{parsed_header.padding_bits}</span></div>
                {audio_extra}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Decode ─────────────────────────────────────────────────────────────
        raw_payload      = raw_decompress[payload_start:]
        restored_bytes:  bytes = b""
        decompress_error: str  = ""
        is_image_archive       = (parsed_header.algorithm_id == ALGO_RLE_QUANT)
        dec_it_metrics:  InformationTheoryMetrics | None = None
        dec_img_info:    dict | None                     = None
        dec_signal:      np.ndarray | None               = None
        dec_sample_rate: int = 44100

        if is_image_archive and len(raw_payload) >= 10:
            dec_img_info = get_image_info(raw_payload)

        with st.spinner(f"Decoding with {algo_disp} …"):
            try:
                if is_audio_archive and audio_hdr is not None:
                    dec_signal = decode_audio(
                        payload         = raw_payload,
                        padding_bits    = audio_hdr.padding_bits,
                        algorithm_id    = audio_hdr.algorithm_id,
                        original_length = audio_hdr.original_length,
                        bits            = audio_hdr.bits,
                        delta_step      = audio_hdr.delta_step,
                        multiplier      = 1.5,
                        alpha           = audio_hdr.alpha,
                        sampling_factor = audio_hdr.sampling_factor,
                    )
                    dec_sample_rate = audio_hdr.sample_rate
                    restored_bytes  = export_array_to_format(dec_signal, dec_sample_rate, ext_disp)

                elif is_image_archive:
                    restored_bytes = image_engine.decode_image(
                        raw_payload, parsed_header.padding_bits
                    )
                else:
                    restored_bytes = codec_engine.decompress(
                        raw_payload,
                        parsed_header.algorithm_id,
                        parsed_header.padding_bits,
                    )
                    dec_it_metrics = analyze_source(
                        data=restored_bytes, compressed=raw_payload,
                        padding_bits=parsed_header.padding_bits,
                    )
            except Exception as exc:
                decompress_error = str(exc)

        if decompress_error:
            st.error(f"Decompression failed: {decompress_error}", icon="🚫")
            st.stop()

        # ── Previews ───────────────────────────────────────────────────────────
        if is_audio_archive:
            st.markdown("##### 🎵 Reconstructed Audio")
            mime_type = f"audio/{ext_disp}" if ext_disp not in ['m4a', 'mp3'] else ("audio/mp4" if ext_disp == 'm4a' else "audio/mpeg")
            st.audio(restored_bytes, format=mime_type)

        if is_image_archive:
            try:
                st.image(restored_bytes, caption="Reconstructed image",
                         use_container_width=True)
            except Exception:
                pass

        # ── Summary ────────────────────────────────────────────────────────────
        sm_r = _size_metrics(restored_bytes)
        ratio_str_d, savings_str_d, css_d = _format_ratio(sm_r["bytes"], sm_d["bytes"])

        st.divider()
        dr1, dr2, dr3 = st.columns(3)
        dr1.metric("Archive Size",  f"{sm_d['bytes']:,} B")
        dr2.metric("Restored Size", f"{sm_r['bytes']:,} B",
                   delta=f"{sm_r['bytes'] - sm_d['bytes']:+,} B vs archive",
                   delta_color="off")
        dr3.metric("Archive / Restored", ratio_str_d)

        stem = _abc_stem(uploaded_decompress.name)
        ext  = parsed_header.extension
        if is_audio_archive:
            restored_filename = f"{stem}_restored.{ext_disp}"
        elif is_image_archive:
            restored_filename = f"{stem}_restored.png"
        elif ext:
            restored_filename = f"{stem}_restored.{ext}"
        else:
            restored_filename = f"{stem}_restored.bin"

        st.success(
            f"✅  Decoded via **{algo_disp}** — "
            f"**{uploaded_decompress.name}** → **{restored_filename}**",
            icon="✅",
        )

        # Progressive disclosure
        with st.expander("📊 View Advanced Mathematical Analytics", expanded=False):
            if dec_it_metrics is not None:
                st.subheader("📐 Information Theory Analytics")
                _render_it_panel(dec_it_metrics, algo_disp)

            if is_image_archive and dec_img_info is not None:
                st.subheader("📡 Image Archive Properties")
                st.info("MSE / PSNR require the original source image.  "
                        "Computed only during Compress.", icon="📡")
                qa1, qa2, qa3 = st.columns(3)
                qa1.metric("Dimensions",
                           f"{dec_img_info['height']} × {dec_img_info['width']} px")
                qa2.metric("Quantization",
                           f"{dec_img_info['quantize_bits']}-bit "
                           f"({2**dec_img_info['quantize_bits']} levels)")
                qa3.metric("Colour Space",
                           "Grayscale" if dec_img_info["channels"] == 1 else "RGB")

            if is_audio_archive and audio_hdr is not None:
                st.subheader("🎵 Audio Archive Properties")
                aa1, aa2, aa3, aa4 = st.columns(4)
                aa1.metric("Sample Rate", f"{audio_hdr.sample_rate:,} Hz")
                aa2.metric("Original Samples", f"{audio_hdr.original_length:,}")
                aa3.metric("Bit-Depth", f"{audio_hdr.bits}-bit")
                aa4.metric("Algorithm", algo_disp)
                st.info(
                    "SQNR requires the original source signal for comparison. "
                    "It is computed during the Compress step only.",
                    icon="🎵",
                )

            st.markdown("---")
            st.markdown("##### 🔩 Archive Internals")
            ai1, ai2, ai3, ai4 = st.columns(4)
            ai1.metric("Algorithm ID", f"0x{parsed_header.algorithm_id:02X}")
            ai2.metric("File Type",    file_type_disp)
            ai3.metric("Header Size",  f"{parsed_header.total_size} B")
            ai4.metric("Payload Bits", f"{logical_bits:,}")

        st.divider()
        st.download_button(
            label     = f"⬇️  Download  {restored_filename}",
            data      = restored_bytes,
            file_name = restored_filename,
            mime      = mime_type if is_audio_archive else "application/octet-stream",
            key       = "decompress_download",
        )
        st.caption(
            f"Restored via **{algo_disp}**. "
            + (f"Audio output (.{ext_disp}) — all encoding parameters recovered from the ARCH header."
               if is_audio_archive
               else "Lossless — run `md5sum` to verify byte-for-byte integrity."
               if not is_image_archive
               else "PNG output.  Lossy transforms applied at compression time are irreversible.")
        )

    else:
        st.info("📂 Upload a `.abc` archive to decompress.", icon="📂")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Theory & Simulations
# ══════════════════════════════════════════════════════════════════════════════
with tab_theory:
    st.subheader("Theory & Simulations", divider="gray")
    st.markdown(
        "A self-contained reference for the mathematical foundations of every "
        "algorithm in this engine."
    )

    # ── SECTION A: Information Theory ─────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="theory-section"><h3>📐 Section A — Information Theory</h3></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Shannon Entropy")
    st.markdown(
        "The **self-information** of a single event with probability $P$ is:"
    )
    st.latex(r"I(x_i) = -\log_2 P(x_i) \quad \text{[bits]}")
    st.markdown("**Shannon Entropy** $H(X)$ is the expected surprise over all symbols:")
    st.latex(r"H(X) = -\sum_{x_i \in \mathcal{X}} P(x_i) \log_2 P(x_i) \quad \text{[bits per symbol]}")
    st.markdown(
        '<div class="insight-box">💡 <b>Source Coding Theorem:</b>  '
        'H(X) is the absolute lower bound on the average bits per symbol for any lossless code.  '
        'No algorithm can compress below this limit.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Coding Efficiency")
    st.latex(r"L = \frac{\text{payload\_bits} - \text{padding\_bits}}{N} \qquad \eta = \frac{H(X)}{L} \times 100\%")

    # ── SECTION B: Algorithms ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="theory-section"><h3>⚙️ Section B — Compression Algorithms</h3></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Huffman Coding (Entropy Coder)")
    st.markdown("Maps frequent bytes to short codewords.  Satisfies:")
    st.latex(r"H(X) \leq L_{\text{Huffman}} < H(X) + 1 \quad \text{[bits/symbol]}")
    st.markdown("### LZ78 (Dictionary Coder)")
    st.latex(r"\text{token} = \bigl(\underbrace{i}_{\text{dict index}},\;\underbrace{c}_{\text{new byte}}\bigr)")
    st.markdown(
        "LZ78 encodes *phrases* rather than individual bytes.  On highly redundant "
        "sources it achieves $L < H(X)$, transcending the per-symbol Shannon bound."
    )
    st.markdown("### Chained Codec: LZ78 → Huffman")
    st.latex(r"\text{data} \xrightarrow{\text{LZ78}} \text{intermediate} \xrightarrow{\text{Huffman}} \text{payload}")
    st.markdown(
        '<div class="insight-box">💡 The same principle powers gzip (LZ77 + Huffman) and bzip2 (BWT + Huffman).</div>',
        unsafe_allow_html=True,
    )

    # ── SECTION C: Signal Processing (Image) ──────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="theory-section"><h3>📡 Section C — Image Signal Processing</h3></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Spatial Quantization")
    st.latex(r"\hat{p} = \left(\frac{p}{2^{8-q}}\right)\bigg\lfloor \cdot 2^{8-q}")
    st.markdown("### Mean Squared Error (MSE)")
    st.latex(
        r"\text{MSE} = \frac{1}{N} \sum_{i,j,c} \left[I_{\text{orig}}(i,j,c) - I_{\text{restored}}(i,j,c)\right]^2"
    )
    st.markdown("### Peak Signal-to-Noise Ratio (PSNR)")
    st.latex(r"\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{255^2}{\text{MSE}}\right) \quad \text{[dB]}")
    st.markdown(
        "As $\\text{MSE} \\to 0$, $\\text{PSNR} \\to \\infty$ (reported as **Infinity** for "
        "a lossless round-trip).  PSNR > 35 dB is generally considered visually lossless."
    )
    st.markdown("### ITU-R BT.601 Luma Conversion")
    st.latex(r"Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B")

    # ── SECTION D: Audio Waveform Coding ──────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="theory-section audio"><h3>🎵 Section D — Audio Waveform Coding Systems</h3></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### D.1  Uniform Pulse-Code Modulation (PCM)")
    st.markdown(
        """
Pulse-Code Modulation is the foundational digital audio standard.  A continuous
amplitude signal is sampled at rate $f_s$ (Nyquist theorem: $f_s \\geq 2 B$
where $B$ is the signal bandwidth) and each sample is quantised into one of
$2^b$ uniform levels.

**Uniform quantiser** (mid-rise, $b$ bits, amplitude range $[v_{\\min}, v_{\\max}]$):
        """
    )
    st.latex(
        r"\Delta_q = \frac{v_{\max} - v_{\min}}{2^b} \qquad "
        r"\hat{x}[n] = v_{\min} + \left\lfloor \frac{x[n] - v_{\min}}{\Delta_q} \right\rfloor \Delta_q + \frac{\Delta_q}{2}"
    )
    st.markdown(
        "For a full-scale sinusoidal input, Bennett's formula gives the theoretical "
        "**Signal-to-Quantisation-Noise Ratio (SQNR)**:"
    )
    st.latex(r"\text{SQNR} \approx 6.02\,b + 1.76 \quad \text{[dB]}")
    st.markdown(
        """
Every additional bit of quantisation adds approximately **6 dB** of dynamic range.
Common reference points:

| Bit-Depth | SQNR (theory) | Application |
|---|---|---|
| 4 bits | 25.8 dB | Telephony (G.726) |
| 8 bits | 49.9 dB | Standard telephony, AM radio |
| 16 bits | 98.1 dB | CD audio |
| 24 bits | 146.2 dB | Studio recording |

**Downsampling** by factor $M$ reduces the sample count (and thus the compressed
size) by $M$, at the cost of losing spectral content above $f_s / (2M)$.
        """
    )
    st.markdown(
        '<div class="insight-box audio">💡 <b>Key insight:</b>  PCM is an open-loop, '
        'sample-independent coder.  It does not exploit sample-to-sample correlation '
        '— each sample is encoded from scratch.  DPCM and DM exploit that correlation '
        'to compress further at the same bit-depth.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### D.2  Delta Modulation (DM)")
    st.markdown(
        """
Delta Modulation is a 1-bit differential coding scheme.  A **staircase
approximator** $\\sigma[n]$ tracks the input signal using a single bit per
sample:
        """
    )
    st.latex(
        r"\text{if } x[n] > \sigma[n-1]: \quad b[n] = 1, \quad \sigma[n] = \sigma[n-1] + \Delta"
    )
    st.latex(
        r"\text{else:} \qquad\qquad\qquad\quad b[n] = 0, \quad \sigma[n] = \sigma[n-1] - \Delta"
    )
    st.markdown("**Two fundamental noise regimes arise from the choice of $\\Delta$:**")
    st.markdown(
        """
**Slope-overload distortion** occurs when the signal changes faster than the
staircase can keep up.  The condition is:
        """
    )
    st.latex(
        r"\left|\frac{dx}{dt}\right|_{\max} > \Delta \cdot f_s"
    )
    st.markdown(
        """
When this condition is violated, the staircase falls perpetually behind the
signal — producing large systematic error.

**Granular (hunting) noise** occurs when the signal is near-stationary.
The staircase oscillates around the true value by ±Δ, producing a
sawtooth-like noise floor whose power is proportional to $\\Delta^2$:
        """
    )
    st.latex(
        r"\sigma_{\text{granular}}^2 \approx \frac{\Delta^2}{3}"
    )
    st.markdown(
        "The two conditions impose **conflicting requirements**: large $\\Delta$ "
        "prevents slope overload but worsens granular noise; small $\\Delta$ "
        "reduces granular noise but causes slope overload.  This dilemma motivates "
        "Adaptive DM."
    )

    st.markdown("### D.3  Adaptive Delta Modulation (ADM) — Jayant's Algorithm")
    st.markdown(
        """
Jayant (1974) resolves the DM dilemma by adapting $\\Delta[n]$ on a
sample-by-sample basis using the previous two output bits:
        """
    )
    st.latex(
        r"\Delta[n] = \Delta[n-1] \times M^{\,\text{sign}(2\,b[n]\,b[n-1] - 1)}"
    )
    st.markdown("Which simplifies to the two-state rule:")
    st.latex(
        r"\Delta[n] = \begin{cases}"
        r"\Delta[n-1] \times M & \text{if } b[n] = b[n-1] \quad \text{(slope overload detected)}\\"
        r"\Delta[n-1] / M & \text{if } b[n] \neq b[n-1] \quad \text{(granular noise detected)}"
        r"\end{cases}"
    )
    st.markdown(
        """
The **multiplier** $M > 1$ (typically 1.5 – 2.0) controls adaptation speed:

- **Large M**: Rapid adaptation to slope changes; but over-shoots create ringing artefacts.
- **Small M (near 1)**: Slow, smooth adaptation; residual slope-overload on transients.

$\\Delta[n]$ is clamped to $[\\Delta_{\\min},\\, \\Delta_{\\max}]$ to prevent
numerical explosion on sustained tones or vanishing on silence.  The step size
history is **implicit in the bitstream** — the decoder reconstructs $\\Delta[n]$
identically because it mirrors the same state machine, needing only $M$ and
$\\Delta[0]$ from the archive header.
        """
    )
    st.markdown(
        '<div class="insight-box audio">💡 <b>ADM vs DM:</b>  '
        'ADM effectively implements a variable time-constant low-pass filter on the '
        'quantisation step.  At the cost of a slightly more complex decoder it '
        'eliminates the fixed trade-off between slope-overload and granular noise '
        'that plagues standard DM.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### D.4  Differential PCM (DPCM) with First-Order Predictor")
    st.markdown(
        """
DPCM exploits the high sample-to-sample correlation of audio signals.  Rather
than encoding the raw sample $x[n]$, it encodes the **prediction residual**
$e[n]$ — the part of the signal that the predictor could *not* anticipate.

**First-order linear predictor** (auto-regressive AR(1) model):
        """
    )
    st.latex(r"\hat{x}[n] = \alpha \cdot x_{\text{rec}}[n-1]")
    st.markdown("**Residual** (the signal component the predictor did not capture):")
    st.latex(r"e[n] = x[n] - \hat{x}[n]")
    st.markdown(
        "The residual is quantised to $b$ bits and bit-packed.  The **closed-loop** "
        "structure (using the locally *reconstructed* $x_{\\text{rec}}[n]$, not the "
        "true $x[n]$, for the next prediction) is critical — it guarantees that the "
        "encoder and decoder maintain **identical internal state**, preventing "
        "long-term drift:"
    )
    st.latex(
        r"x_{\text{rec}}[n] = \hat{x}[n] + e_{\text{rec}}[n] = \alpha\,x_{\text{rec}}[n-1] + e_{\text{rec}}[n]"
    )
    st.markdown(
        """
**Why DPCM outperforms PCM at the same bit-depth:**

For a stationary AR(1) source with autocorrelation $r[1] = \\alpha \\sigma_x^2$,
the residual variance is:
        """
    )
    st.latex(
        r"\sigma_e^2 = (1 - \alpha^2)\,\sigma_x^2"
    )
    st.markdown(
        "When $\\alpha \\approx 0.9$ (typical for speech or music), "
        "$\\sigma_e^2 \\approx 0.19\\,\\sigma_x^2$ — the residual variance is "
        "only **19%** of the signal variance.  Quantising a smaller dynamic range "
        "with $b$ bits achieves approximately:"
    )
    st.latex(
        r"\Delta\,\text{SQNR}_{\text{DPCM}} \approx 10 \log_{10}\!\left(\frac{\sigma_x^2}{\sigma_e^2}\right) = 10 \log_{10}\!\left(\frac{1}{1 - \alpha^2}\right) \text{ dB}"
    )
    st.markdown(
        "For $\\alpha = 0.9$: $\\Delta\\,\\text{SQNR} \\approx 7.2$ dB improvement over raw PCM — "
        "roughly equivalent to adding **1.2 extra bits** of quantisation precision."
    )
    st.markdown(
        '<div class="insight-box audio">💡 <b>DPCM is the foundation of modern audio codecs.</b>  '
        'G.726 ADPCM (telephone speech), SBC (Bluetooth audio), and the prediction stages '
        'inside MP3 and AAC all use forms of DPCM or its higher-order generalisations.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(
        "All formulas implemented as production-grade, unit-tested, pure-Python code in "
        "`audio_engine.py`, `codec_engine.py`, and `image_engine.py`. "
        "No external compression libraries are used."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#6c757d;font-size:0.8rem;'>"
    "CodecCore: Data Compression Engine &nbsp;|&nbsp; "
    "Huffman · LZ78 · Chained · RLE Image · PCM · DM · ADM · DPCM &nbsp;|&nbsp; "
    "ARCH Binary Header Protocol"
    "</p>",
    unsafe_allow_html=True,
)
