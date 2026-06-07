"""
simulators.py — Live Micro-Simulators for CodecCore Phase 7.

Three interactive, step-by-step algorithm visualisers that run entirely with
Streamlit-native primitives (no Matplotlib or Plotly).  Each function owns an
isolated namespace inside ``st.session_state`` so that widgets in one simulator
never collide with those in another.

Simulator A : LZ78 Dictionary Builder
Simulator B : Spatial Quantization + Run-Length Encoding (Image)
Simulator C : Delta Modulation Waveform Tracker (Audio)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ss(key: str, default: Any) -> None:
    """Initialise a session-state key with *default* if it does not exist."""
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Simulator A — LZ78 Dictionary Builder
# ─────────────────────────────────────────────────────────────────────────────


def _lz78_step(text: str, pos: int, dictionary: dict) -> tuple[int, list, list, dict]:
    """Execute one LZ78 step starting at *pos*.

    Returns (new_pos, updated_dict_rows, updated_token_rows, updated_dict)
    where:
      dict_rows  — list of (index, pattern) for the full dictionary
      token_rows — list of (step, dict_ref, new_char) for the output stream
    """
    if pos >= len(text):
        return pos, [], [], dictionary

    current_str = ""
    while pos < len(text):
        current_str += text[pos]
        pos += 1
        if current_str not in dictionary:
            idx = len(dictionary) + 1
            dictionary[current_str] = idx
            break

    # Token: (dict_index_of_prefix, new_char)
    prefix = current_str[:-1]
    new_char = current_str[-1]
    dict_ref = dictionary.get(prefix, 0)

    return pos, dictionary, (dict_ref, new_char)


def render_lz78_simulator() -> None:
    """Render the interactive LZ78 Dictionary Builder simulator."""

    # ── Session-state bootstrap ───────────────────────────────────────────────
    _ss("lz78_pos", 0)
    _ss("lz78_dict", {})        # pattern → index
    _ss("lz78_tokens", [])      # list of (step, dict_ref, new_char)
    _ss("lz78_input", "TOBEORNOTTOBE")
    _ss("lz78_finished", False)

    # ── Controls ─────────────────────────────────────────────────────────────
    col_input, col_buttons = st.columns([3, 1])
    with col_input:
        new_text = st.text_input(
            "Input string",
            value=st.session_state.lz78_input,
            max_chars=64,
            key="lz78_text_input",
            placeholder="Enter any text string…",
        ).upper()

        # If the user changed the text, reset everything.
        if new_text != st.session_state.lz78_input:
            st.session_state.lz78_input = new_text
            st.session_state.lz78_pos = 0
            st.session_state.lz78_dict = {}
            st.session_state.lz78_tokens = []
            st.session_state.lz78_finished = False
            st.rerun()

    with col_buttons:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        step_clicked  = c1.button("▶ Step", key="lz78_step_btn",  use_container_width=True)
        reset_clicked = c2.button("↺ Reset", key="lz78_reset_btn", use_container_width=True)

    text = st.session_state.lz78_input

    # ── Reset ─────────────────────────────────────────────────────────────────
    if reset_clicked:
        st.session_state.lz78_pos = 0
        st.session_state.lz78_dict = {}
        st.session_state.lz78_tokens = []
        st.session_state.lz78_finished = False
        st.rerun()

    # ── Step forward ──────────────────────────────────────────────────────────
    if step_clicked and not st.session_state.lz78_finished and text:
        old_pos = st.session_state.lz78_pos
        if old_pos < len(text):
            new_pos, updated_dict, token = _lz78_step(
                text, old_pos, dict(st.session_state.lz78_dict)
            )
            st.session_state.lz78_dict = updated_dict
            st.session_state.lz78_tokens.append(token)
            st.session_state.lz78_pos = new_pos
            if new_pos >= len(text):
                st.session_state.lz78_finished = True
        st.rerun()

    # ── Progress bar ──────────────────────────────────────────────────────────
    pos      = st.session_state.lz78_pos
    finished = st.session_state.lz78_finished

    if text:
        progress = min(pos / max(len(text), 1), 1.0)
        st.progress(progress, text=f"Position {pos} / {len(text)}")
    else:
        st.warning("Please enter a non-empty string.", icon="⚠️")
        return

    # ── String highlight ──────────────────────────────────────────────────────
    st.markdown("#### 🔍 String Scanner")

    if text:
        # Determine what pattern is currently being built from pos onward
        # (only for display; we look back at where the last step started)
        last_token_patterns = [t[1] for t in st.session_state.lz78_tokens]
        consumed = sum(
            len(tok[1]) + (1 if tok else 0)
            for tok in st.session_state.lz78_tokens
        )
        # Simpler: reconstruct consumed chars from dict steps
        step_chars = 0
        for dict_ref, new_char in st.session_state.lz78_tokens:
            prefix_len = 0
            for pattern, idx in st.session_state.lz78_dict.items():
                if idx == dict_ref:
                    prefix_len = len(pattern)
                    break
            step_chars += prefix_len + 1

        html_chars = []
        for i, ch in enumerate(text):
            if i < pos - (len(text) - step_chars if finished else 0):
                # Already processed
                html_chars.append(
                    f"<span style='color:#4ade80;font-weight:700;font-size:1.3rem;"
                    f"font-family:monospace;'>{ch}</span>"
                )
            elif i == pos and not finished:
                # Current head
                html_chars.append(
                    f"<span style='background:#7c3aed;color:#fff;padding:2px 5px;"
                    f"border-radius:4px;font-weight:800;font-size:1.3rem;"
                    f"font-family:monospace;'>{ch}</span>"
                )
            else:
                html_chars.append(
                    f"<span style='color:#94a3b8;font-size:1.3rem;"
                    f"font-family:monospace;'>{ch}</span>"
                )

        st.markdown(
            "<div style='letter-spacing:4px;margin:0.5rem 0 1rem;'>"
            + "&nbsp;".join(html_chars)
            + "</div>",
            unsafe_allow_html=True,
        )

    if finished:
        st.success("✅ Full string parsed! All LZ78 phrases discovered.", icon="🎉")

    # ── Side-by-side tables ───────────────────────────────────────────────────
    col_dict, col_tokens = st.columns(2)

    with col_dict:
        st.markdown("**📖 Dictionary**")
        if st.session_state.lz78_dict:
            dict_df = pd.DataFrame(
                [
                    {"Index": idx, "Pattern": f'"{pattern}"'}
                    for pattern, idx in sorted(
                        st.session_state.lz78_dict.items(), key=lambda x: x[1]
                    )
                ]
            )
            st.dataframe(dict_df, hide_index=True, use_container_width=True)
        else:
            st.caption("Dictionary is empty — press ▶ Step to begin.")

    with col_tokens:
        st.markdown("**📤 Output Tokens**")
        if st.session_state.lz78_tokens:
            tok_df = pd.DataFrame(
                [
                    {
                        "Step": i + 1,
                        "Dict Ref": ref,
                        "New Char": f'"{ch}"',
                    }
                    for i, (ref, ch) in enumerate(st.session_state.lz78_tokens)
                ]
            )
            st.dataframe(tok_df, hide_index=True, use_container_width=True)
        else:
            st.caption("No tokens yet.")


# ─────────────────────────────────────────────────────────────────────────────
# Simulator B — Spatial Quantization + RLE (Image)
# ─────────────────────────────────────────────────────────────────────────────


def _quantize_array(arr: np.ndarray, bits: int) -> np.ndarray:
    """Apply uniform bit-shift quantisation to a uint8 array."""
    if bits >= 8:
        return arr.copy()
    shift = 8 - bits
    return ((arr >> shift) << shift).astype(np.uint8)


def _rle_encode(row: np.ndarray) -> list[tuple[int, int]]:
    """Return a flat list of (value, count) run-length encoding for one row."""
    if len(row) == 0:
        return []
    runs: list[tuple[int, int]] = []
    current_val = int(row[0])
    count = 1
    for v in row[1:]:
        v = int(v)
        if v == current_val:
            count += 1
        else:
            runs.append((current_val, count))
            current_val = v
            count = 1
    runs.append((current_val, count))
    return runs


def _array_to_pil_image(arr: np.ndarray, scale: int = 48):
    """Scale an 8×8 grayscale array to a visible PIL image."""
    try:
        from PIL import Image
        img = Image.fromarray(arr, mode="L")
        return img.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    except ImportError:
        return None


def render_quantization_simulator() -> None:
    """Render the Spatial Quantisation & RLE micro-simulator."""

    # Fixed 8×8 smooth gradient
    gradient = np.array(
        [[int(255 * (r * 8 + c) / 63) for c in range(8)] for r in range(8)],
        dtype=np.uint8,
    )

    bits = st.slider(
        "Quantization Bit-Depth (bits per pixel)",
        min_value=1, max_value=8, value=8, step=1,
        key="quant_sim_bits",
        help="8 bits = full quality (256 levels). Lower = fewer levels → longer RLE runs.",
    )

    quantized = _quantize_array(gradient, bits)
    n_levels  = 2 ** bits

    st.markdown(f"**Active levels:** `{n_levels}` &nbsp;|&nbsp; **Step size:** `{256 // n_levels}` grey values")

    col_orig, col_quant = st.columns(2)

    with col_orig:
        st.markdown("**Original (8-bit)**")
        pil_orig = _array_to_pil_image(gradient)
        if pil_orig:
            st.image(pil_orig, caption="Original 8-bit gradient", use_column_width=False)
        else:
            st.dataframe(
                pd.DataFrame(gradient),
                hide_index=True,
                use_container_width=True,
            )

    with col_quant:
        st.markdown(f"**Quantized ({bits}-bit)**")
        pil_quant = _array_to_pil_image(quantized)
        if pil_quant:
            st.image(pil_quant, caption=f"{bits}-bit quantized ({n_levels} levels)", use_column_width=False)
        else:
            st.dataframe(
                pd.DataFrame(quantized),
                hide_index=True,
                use_container_width=True,
            )

    # ── RLE table ─────────────────────────────────────────────────────────────
    st.markdown("#### 📊 Run-Length Encoding Output (row-by-row)")

    all_rle_rows = []
    total_tokens = 0
    for r in range(8):
        row_runs = _rle_encode(quantized[r])
        total_tokens += len(row_runs)
        for val, cnt in row_runs:
            all_rle_rows.append({"Row": r, "Value": val, "Run Length": cnt})

    rle_df = pd.DataFrame(all_rle_rows)
    st.dataframe(rle_df, hide_index=True, use_container_width=True, height=220)

    # ── Token count metric ────────────────────────────────────────────────────
    raw_tokens    = 8 * 8           # one token per pixel at 8-bit
    saving_pct    = (1 - total_tokens / raw_tokens) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Raw pixel count",  f"{raw_tokens}",    help="64 pixels — one value each.")
    m2.metric("RLE token count",  f"{total_tokens}",
              delta=f"{total_tokens - raw_tokens:+d} vs raw",
              delta_color="inverse")
    m3.metric("Token reduction",  f"{saving_pct:.1f}%",
              delta="↓ fewer tokens is better", delta_color="off")

    st.markdown(
        '<div class="insight-box">💡 <b>Key insight:</b> As bit-depth decreases, '
        'adjacent pixels merge into the same quantisation level, creating <em>longer runs</em> '
        'and dramatically fewer RLE tokens — this is the core of lossy image compression.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulator C — Delta Modulation Waveform Tracker
# ─────────────────────────────────────────────────────────────────────────────


def _run_delta_modulation(signal: np.ndarray, delta: float) -> np.ndarray:
    """Run a standard Delta Modulation tracking loop on *signal*.

    Returns the staircase approximator array (same length as *signal*).
    """
    staircase = np.zeros(len(signal), dtype=np.float32)
    approx    = 0.0
    for n, x in enumerate(signal):
        if x > approx:
            approx += delta
        else:
            approx -= delta
        staircase[n] = approx
    return staircase


def _run_dpcm(signal: np.ndarray, alpha: float, bits: int = 4) -> np.ndarray:
    """Run a closed-loop first-order DPCM encoder/decoder loop.

    Returns the reconstructed signal array.
    """
    n_levels  = 2 ** bits
    max_res   = 1.0
    step      = (2 * max_res) / n_levels

    reconstructed = np.zeros(len(signal), dtype=np.float32)
    x_rec_prev    = 0.0
    for n, x in enumerate(signal):
        x_hat = alpha * x_rec_prev
        residual = float(x) - x_hat
        # Uniform quantise residual
        level = int(np.clip(np.floor(residual / step), -(n_levels // 2), n_levels // 2 - 1))
        res_rec = level * step
        x_rec = x_hat + res_rec
        reconstructed[n] = x_rec
        x_rec_prev = x_rec

    return reconstructed


def render_waveform_simulator() -> None:
    """Render the Delta Modulation / DPCM waveform tracking micro-simulator."""

    # Fixed 50-sample sine wave
    n_samples  = 50
    t          = np.linspace(0, 2 * math.pi, n_samples)
    signal     = np.sin(t).astype(np.float32)

    col_sl1, col_sl2 = st.columns(2)
    with col_sl1:
        delta = st.slider(
            "Delta Step Size (Δ)",
            min_value=0.01, max_value=0.50, value=0.10, step=0.01,
            format="%.2f",
            key="waveform_sim_delta",
            help="Large Δ → slope overload eliminated but granular noise increases. "
                 "Small Δ → smooth but slope overload appears on steep parts.",
        )
    with col_sl2:
        alpha = st.slider(
            "DPCM Predictor Weight (α)",
            min_value=0.50, max_value=0.99, value=0.90, step=0.01,
            format="%.2f",
            key="waveform_sim_alpha",
            help="α ≈ 0.9 is optimal for speech-like signals with high inter-sample correlation.",
        )

    # Run coders
    dm_staircase  = _run_delta_modulation(signal, delta)
    dpcm_signal   = _run_dpcm(signal, alpha)

    # Build a tidy DataFrame for st.line_chart
    df = pd.DataFrame(
        {
            "Original Sine Wave":   signal,
            "DM Staircase":         dm_staircase,
            "DPCM Reconstruction":  dpcm_signal,
        },
        index=range(n_samples),
    )

    st.markdown("#### 📈 Live Waveform Tracker")
    st.line_chart(df, use_container_width=True, height=320)

    # ── Noise metrics ──────────────────────────────────────────────────────────
    def sqnr(orig: np.ndarray, recon: np.ndarray) -> float:
        noise = orig - recon
        s_pwr = float(np.sum(orig ** 2))
        n_pwr = float(np.sum(noise ** 2))
        if n_pwr < 1e-12:
            return float("inf")
        return 10.0 * math.log10(s_pwr / n_pwr)

    dm_sqnr   = sqnr(signal, dm_staircase)
    dpcm_sqnr = sqnr(signal, dpcm_signal)

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "DM SQNR",
        f"{dm_sqnr:.2f} dB" if dm_sqnr != float("inf") else "∞",
        help="Signal-to-Quantisation-Noise Ratio for the Delta Modulation staircase.",
    )
    m2.metric(
        "DPCM SQNR",
        f"{dpcm_sqnr:.2f} dB" if dpcm_sqnr != float("inf") else "∞",
        help="SQNR for the closed-loop DPCM reconstruction (4-bit residual quantiser).",
    )
    dm_err_max = float(np.max(np.abs(signal - dm_staircase)))
    m3.metric(
        "DM Max Abs Error",
        f"{dm_err_max:.4f}",
        delta="Slope overload visible if > Δ" if dm_err_max > delta else "Within Δ tolerance",
        delta_color="inverse" if dm_err_max > delta else "off",
    )

    # Slope-overload warning
    if dm_err_max > delta * 2:
        st.warning(
            f"⚠️  **Slope overload detected!**  The DM staircase is lagging behind the signal "
            f"by up to `{dm_err_max:.3f}` — more than 2× the step size Δ = `{delta:.2f}`.  "
            f"Increase Δ to eliminate overload.",
            icon="⚠️",
        )
    else:
        st.success(
            f"✅  The staircase is tracking within tolerance (max error `{dm_err_max:.3f}` ≤ 2Δ = `{2*delta:.2f}`).",
            icon="✅",
        )

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("🔢 View Raw Sample Data", expanded=False):
        df_display = df.copy()
        df_display.index.name = "Sample n"
        df_display = df_display.round(5)
        st.dataframe(df_display, use_container_width=True, height=260)

    st.markdown(
        '<div class="insight-box audio">💡 <b>Try it:</b> Drag Δ to <code>0.01</code> and watch '
        'the red staircase fall behind the steep rising edge of the sine wave — '
        'that is slope overload. Drag it to <code>0.40</code> and the staircase tracks '
        'the peak but granular noise appears on the flat trough.</div>',
        unsafe_allow_html=True,
    )
