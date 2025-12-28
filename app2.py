# app.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DN v2 Demo", layout="wide")

st.title("DN v2 Demo")

# ----------------------------
# Helpers
# ----------------------------
def parse_series(text: str, n_expected: int = 10):
    vals = [v for v in text.replace(",", " ").split() if v.strip() != ""]
    arr = [float(v) for v in vals]
    if len(arr) != n_expected:
        raise ValueError(f"Need exactly {n_expected} numbers, got {len(arr)}.")
    return np.array(arr, dtype=float)

def pct_change(arr: np.ndarray):
    out = np.zeros_like(arr, dtype=float)
    for i in range(1, len(arr)):
        prev = arr[i - 1]
        out[i] = 0.0 if prev == 0 else 100.0 * (arr[i] - prev) / prev
    return out

def lorentz_from_pct(pct: np.ndarray, K: float):
    # Core unchanged: T from %Δ normalized by K; E = 1 - T^2
    T = pct / K
    E = 1.0 - (T ** 2)
    vT = np.zeros_like(T)
    vE = np.zeros_like(E)
    vT[1:] = T[1:] - T[:-1]
    vE[1:] = E[1:] - E[:-1]
    return T, E, vT, vE

def robust_mad(x: np.ndarray):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad  # consistent MAD

def build_df(values: np.ndarray, T, E, vT, vE, status, note):
    df = pd.DataFrame({
        "minute": np.arange(1, len(values) + 1),
        "value": np.round(values, 4),
        "t": np.round(T, 6),
        "e": np.round(E, 6),
        "vt": np.round(vT, 6),
        "ve": np.round(vE, 6),
        "status": status,
        "note": note
    })
    return df

def download_csv(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

# ----------------------------
# HRV (Pattern-based; no absolute thresholds)
# ----------------------------
def classify_hrv_pattern(values: np.ndarray):
    pct = pct_change(values)
    T, E, vT, vE = lorentz_from_pct(pct, K=80.0)  # core unchanged, hidden from UI

    # Adaptive "event strength" scale from the series itself (not fixed thresholds)
    s = robust_mad(pct[1:])
    if s == 0:
        s = np.std(pct[1:]) if np.std(pct[1:]) > 0 else 1.0

    status = ["STABLE"] * len(values)
    note = [""] * len(values)

    # Detect local patterns using shape only
    for i in range(1, len(values)):
        # Basic step direction
        step = pct[i]

        # Candidate “impulse” relative to noise level
        is_big_neg = step < (-2.0 * s)
        is_big_pos = step > ( 2.0 * s)

        # V-shape: big drop then rebound next step with opposite sign and similar magnitude
        if i >= 2:
            a = pct[i - 1]
            b = pct[i]
            # rebound symmetry ratio (shape), not absolute number
            if (a < 0 and b > 0) or (a > 0 and b < 0):
                mag_a = abs(a)
                mag_b = abs(b)
                ratio = mag_b / mag_a if mag_a > 1e-9 else 0.0
                # “similar magnitude” + quick sign flip → transient pattern
                if 0.6 <= ratio <= 1.6 and (abs(a) > 1.5 * s):
                    status[i] = "TRANSIENT"
                    note[i] = "V-shape / rebound"
                    continue

        # Noise-spike: sharp up then sharp down (or reverse) within 2 steps, symmetric
        if i >= 3:
            a = pct[i - 2]
            b = pct[i - 1]
            c = pct[i]
            # spike core: + then - then settle or - then + then settle
            if (a > 0 and b < 0) or (a < 0 and b > 0):
                mag_a = abs(a)
                mag_b = abs(b)
                ratio = mag_b / mag_a if mag_a > 1e-9 else 0.0
                if 0.7 <= ratio <= 1.3 and (mag_a > 2.0 * s):
                    status[i] = "TRANSIENT"
                    note[i] = "noise-spike candidate"
                    continue

        # Drift-down: consecutive negatives with consistent vT/vE direction (shape), not absolute HRV
        if i >= 3:
            last3 = pct[i-2:i+1]
            if np.all(last3 < 0):
                # stability of direction: low sign flipping in vT/vE and median negative
                if np.median(last3) < (-0.8 * s):
                    status[i] = "STRESS_PATTERN"
                    note[i] = "drift-down"
                    continue

        # Single strong impulse (relative)
        if is_big_neg:
            status[i] = "STRESS_PATTERN"
            note[i] = "impulse drop"
        elif is_big_pos:
            status[i] = "TRANSIENT"
            note[i] = "impulse rise"

    return pct, T, E, vT, vE, status, note

# ----------------------------
# SpO2 (threshold logic kept)
# ----------------------------
def classify_spo2(values: np.ndarray):
    # Use absolute delta per minute (no %), normalized by K=5
    delta = np.zeros_like(values)
    delta[1:] = values[1:] - values[:-1]

    T = delta / 5.0
    E = 1.0 - (T ** 2)
    vT = np.zeros_like(T); vT[1:] = T[1:] - T[:-1]
    vE = np.zeros_like(E); vE[1:] = E[1:] - E[:-1]

    status = ["GREEN"] * len(values)
    note = [""] * len(values)

    for i in range(1, len(values)):
        t = abs(T[i])
        if t >= 0.6:
            status[i] = "RED"
        elif t >= 0.3:
            status[i] = "WARNING"

        # V-shape filter: drop then immediate recovery → INFO-like note (do not force RED)
        if i >= 2:
            a = T[i-1]
            b = T[i]
            if (a < 0 and b > 0) and (abs(a) >= 0.3) and (abs(b) >= 0.3):
                note[i] = "V-shape (possible artifact)"

    return delta, T, E, vT, vE, status, note

# ----------------------------
# RR (threshold logic kept)
# ----------------------------
def classify_rr(values: np.ndarray):
    pct = pct_change(values)
    T, E, vT, vE = lorentz_from_pct(pct, K=25.0)

    status = ["GREEN"] * len(values)
    note = [""] * len(values)

    for i in range(1, len(values)):
        t = abs(T[i])
        if t >= 1.0:
            status[i] = "RED"
        elif t >= 0.5:
            status[i] = "WARNING"

        if i >= 2:
            a = T[i-1]; b = T[i]
            if (a > 0 and b < 0) and (abs(a) >= 0.5) and (abs(b) >= 0.5):
                note[i] = "V-shape / transient breathing"

    return pct, T, E, vT, vE, status, note

# ----------------------------
# HR (threshold logic kept)
# ----------------------------
def classify_hr(values: np.ndarray):
    pct = pct_change(values)
    T, E, vT, vE = lorentz_from_pct(pct, K=15.0)

    status = ["GREEN"] * len(values)
    note = [""] * len(values)

    for i in range(1, len(values)):
        t = abs(T[i])
        if t >= 1.0:
            status[i] = "RED"
        elif t >= 0.5:
            status[i] = "WARNING"

        if i >= 2:
            a = T[i-1]; b = T[i]
            if (a > 0 and b < 0) and (abs(a) >= 0.5) and (abs(b) >= 0.5):
                note[i] = "V-shape / transient"

    return pct, T, E, vT, vE, status, note

# ----------------------------
# UI
# ----------------------------
tabs = st.tabs(["hrv", "spo2", "rr", "hr"])

with tabs[0]:
    txt = st.text_input("HRV (10 points)", value="48 47 46 45 28 27 26 26 25 25")
    if st.button("Compute", key="btn_hrv"):
        try:
            values = parse_series(txt, 10)
            _, T, E, vT, vE, status, note = classify_hrv_pattern(values)
            df = build_df(values, T, E, vT, vE, status, note)
            st.dataframe(df, use_container_width=True, hide_index=True)
            download_csv(df, "dn_v2_hrv.csv")
        except Exception as e:
            st.error(str(e))

with tabs[1]:
    txt = st.text_input("SpO₂ (10 points)", value="98 97 96 95 94 93 92 91 90 89")
    if st.button("Compute", key="btn_spo2"):
        try:
            values = parse_series(txt, 10)
            _, T, E, vT, vE, status, note = classify_spo2(values)
            df = build_df(values, T, E, vT, vE, status, note)
            st.dataframe(df, use_container_width=True, hide_index=True)
            download_csv(df, "dn_v2_spo2.csv")
        except Exception as e:
            st.error(str(e))

with tabs[2]:
    txt = st.text_input("RR (10 points)", value="18 19 20 22 24 26 28 30 32 34")
    if st.button("Compute", key="btn_rr"):
        try:
            values = parse_series(txt, 10)
            _, T, E, vT, vE, status, note = classify_rr(values)
            df = build_df(values, T, E, vT, vE, status, note)
            st.dataframe(df, use_container_width=True, hide_index=True)
            download_csv(df, "dn_v2_rr.csv")
        except Exception as e:
            st.error(str(e))

with tabs[3]:
    txt = st.text_input("HR (10 points)", value="75 76 110 77 78 82 88 95 103 112")
    if st.button("Compute", key="btn_hr"):
        try:
            values = parse_series(txt, 10)
            _, T, E, vT, vE, status, note = classify_hr(values)
            df = build_df(values, T, E, vT, vE, status, note)
            st.dataframe(df, use_container_width=True, hide_index=True)
            download_csv(df, "dn_v2_hr.csv")
        except Exception as e:
            st.error(str(e))
