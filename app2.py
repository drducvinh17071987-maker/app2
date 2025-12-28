# app2.py
# DN v2 Demo (Table-first) - 4 tabs: HRV / SpO2 / RR / HR
# Focus: clean table for SSRN v2 (status + note), robust (no crashes), no plots.

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Parsing & core math utils
# =========================
def parse_series(text: str):
    """
    Accept: '50 49 48', '50,49,48', '50;49;48'
    Return: list[float]
    """
    if text is None:
        return []
    s = text.replace(",", " ").replace(";", " ").strip()
    if not s:
        return []
    out = []
    for tok in s.split():
        try:
            out.append(float(tok))
        except:
            pass
    return out


def safe_pct_change(values: np.ndarray):
    """
    %Δ_i = 100 * (x_i - x_{i-1}) / x_{i-1}
    %Δ_0 = 0
    """
    x = values.astype(float)
    pct = np.zeros(len(x), dtype=float)
    for i in range(1, len(x)):
        prev = x[i - 1]
        if prev == 0 or np.isnan(prev) or np.isnan(x[i]):
            pct[i] = 0.0
        else:
            pct[i] = 100.0 * (x[i] - prev) / prev
    return pct


def lorentz_E(T: np.ndarray):
    # E = 1 - T^2 (no clipping by default; paper can show negatives if you want)
    return 1.0 - np.square(T)


def first_diff(arr: np.ndarray):
    d = np.zeros(len(arr), dtype=float)
    if len(arr) >= 2:
        d[1:] = arr[1:] - arr[:-1]
    return d


def detect_v_shape(pct: np.ndarray, idx: int,
                   d1_thr=-20.0, d2_thr=+15.0, total_thr=12.0):
    """
    V-shape: down then up quickly, and net change small.
    Uses two-step window: (idx-1 -> idx) as recovery step.
    Conditions (matching your chốt):
      d1_pct <= -20
      d2_pct >= +15
      |total_pct| <= 12, where total is from idx-2 -> idx
    """
    if idx < 2:
        return False
    d1 = pct[idx - 1]
    d2 = pct[idx]
    total = pct[idx - 1] + pct[idx]  # approx net over 2 steps in % domain
    return (d1 <= d1_thr) and (d2 >= d2_thr) and (abs(total) <= total_thr)


# =========================
# HRV (pattern-based, no absolute threshold labeling)
# =========================
def compute_hrv_table(values, K_hrv=80.0,
                      noise_step_thr=70.0, noise_abs_thr=60.0):
    """
    HRV DN_dynamic:
      pct_step = %ΔHRV
      T = pct_step / 80  (your rule)
      E = 1 - T^2
      vT, vE: first difference
    status/note:
      - status: BASE if nothing notable; PATTERN otherwise
      - note: pattern labels (NOISE_SPIKE, V_SHAPE_RECOVERY, DRIFT_DOWN, STEP_DROP, MILD_DOWN, STABLE)
      NO RED/WARNING by threshold in HRV tab.
    """
    x = np.array(values, dtype=float)
    n = len(x)
    if n == 0:
        return pd.DataFrame()

    pct = safe_pct_change(x)
    T = pct / K_hrv
    E = lorentz_E(T)
    vT = first_diff(T)
    vE = first_diff(E)

    note = [""] * n
    status = ["BASE"] * n

    for i in range(n):
        if i == 0:
            note[i] = "BASE"
            status[i] = "BASE"
            continue

        # Noise-brake (v1.5 idea): very fast step up/down or too large absolute jump
        if (abs(pct[i]) >= noise_step_thr) or (abs(x[i] - x[i - 1]) >= noise_abs_thr):
            note[i] = "INFO_NOISE_SPIKE"
            status[i] = "PATTERN"
            continue

        # V-shape recovery (your Core v1 logic)
        if detect_v_shape(pct, i, d1_thr=-20.0, d2_thr=+15.0, total_thr=12.0):
            note[i] = "INFO_V_SHAPE_RECOVERY"
            status[i] = "PATTERN"
            continue

        # Step-drop marker (still a pattern label, not "RED")
        if pct[i] <= -40.0:
            note[i] = "STEP_DROP"
            status[i] = "PATTERN"
            continue

        # Drift-down: last 3 steps mostly down (simple, baseline-free)
        if i >= 3:
            last3 = pct[i-2:i+1]
            if np.all(last3 <= -5.0):
                note[i] = "DRIFT_DOWN"
                status[i] = "PATTERN"
                continue

        # Mild down / stable
        if pct[i] <= -5.0:
            note[i] = "MILD_DOWN"
            status[i] = "PATTERN"
        elif pct[i] >= +5.0:
            note[i] = "MILD_UP"
            status[i] = "PATTERN"
        else:
            note[i] = "STABLE"
            status[i] = "BASE"

    df = pd.DataFrame({
        "minute": np.arange(1, n + 1),
        "value": x,
        "pct_step": np.round(pct, 4),
        "t": np.round(T, 6),
        "e": np.round(E, 6),
        "vT": np.round(vT, 6),
        "vE": np.round(vE, 6),
        "status": status,
        "note": note,
    })
    return df


# =========================
# Threshold-based channels (SpO2 / RR / HR)
# =========================
def compute_threshold_table(values, K, channel_name,
                            red_thr, warn_thr, mild_thr,
                            enable_vshape=True):
    """
    Generic threshold-mode:
      pct_step = %Δ
      T = pct_step / K
      E = 1 - T^2
      vT, vE: diff
    status:
      BASE if abs(T) < mild_thr/2 (quiet); PATTERN otherwise
    note:
      - If V-shape detected => INFO_V_SHAPE (overrides RED/WARN)
      - Else map abs(T) to GREEN/MILD/WARNING/RED
    """
    x = np.array(values, dtype=float)
    n = len(x)
    if n == 0:
        return pd.DataFrame()

    pct = safe_pct_change(x)
    T = pct / float(K)
    E = lorentz_E(T)
    vT = first_diff(T)
    vE = first_diff(E)

    status = ["BASE"] * n
    note = [""] * n

    for i in range(n):
        if i == 0:
            status[i] = "BASE"
            note[i] = "GREEN"
            continue

        # Structure label
        if abs(T[i]) >= (mild_thr / 2.0):
            status[i] = "PATTERN"
        else:
            status[i] = "BASE"

        # V-shape override (false-alarm filter)
        if enable_vshape and detect_v_shape(pct, i, d1_thr=-(warn_thr * K), d2_thr=+(warn_thr * K), total_thr=12.0):
            # Here we tie v-shape trigger to warn_thr*K in % space
            note[i] = "INFO_V_SHAPE"
            continue

        a = abs(T[i])
        if a >= red_thr:
            note[i] = "RED"
        elif a >= warn_thr:
            note[i] = "WARNING"
        elif a >= mild_thr:
            note[i] = "MILD"
        else:
            note[i] = "GREEN"

    df = pd.DataFrame({
        "minute": np.arange(1, n + 1),
        "value": x,
        "pct_step": np.round(pct, 4),
        "t": np.round(T, 6),
        "e": np.round(E, 6),
        "vT": np.round(vT, 6),
        "vE": np.round(vE, 6),
        "status": status,
        "note": note,
    })
    return df


def download_csv(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=filename, mime="text/csv")


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="DN v2 Demo - Table", layout="wide")
st.title("DN v2 Demo – Table-first (HRV / SpO2 / RR / HR)")
st.caption("Goal: clean computation + clean status/note for SSRN v2 table export.")


tabs = st.tabs(["HRV", "SpO2", "RR", "HR"])

# --- HRV ---
with tabs[0]:
    st.subheader("HRV (10 points)")
    default_hrv = "50 49 48 47 46 45 44 43 42 41"
    txt = st.text_input("HRV series", value=default_hrv, key="hrv_in")
    vals = parse_series(txt)

    col1, col2, col3 = st.columns(3)
    with col1:
        K_hrv = st.number_input("K_hrv (fixed)", value=80.0, step=1.0, key="K_hrv")
    with col2:
        noise_step = st.number_input("Noise step %Δ threshold", value=70.0, step=1.0, key="noise_step")
    with col3:
        noise_abs = st.number_input("Noise abs jump threshold", value=60.0, step=1.0, key="noise_abs")

    if st.button("Compute", key="btn_hrv"):
        df = compute_hrv_table(vals, K_hrv=K_hrv, noise_step_thr=noise_step, noise_abs_thr=noise_abs)
        st.dataframe(df, use_container_width=True)
        download_csv(df, "dn_hrv_table.csv")

# --- SpO2 ---
with tabs[1]:
    st.subheader("SpO2 (10 points)")
    default_spo2 = "98 97 96 95 94 93 92 91 90 89"
    txt = st.text_input("SpO2 series", value=default_spo2, key="spo2_in")
    vals = parse_series(txt)

    col1, col2 = st.columns(2)
    with col1:
        K_spo2 = st.number_input("K_spo2 (% per step)", value=10.0, step=1.0, key="K_spo2")
    with col2:
        st.write("Thresholds in |T| (your rule): RED≥0.6, WARNING≥0.3, MILD≥0.2 (V-shape => INFO)")

    if st.button("Compute", key="btn_spo2"):
        df = compute_threshold_table(
            vals, K=K_spo2, channel_name="SpO2",
            red_thr=0.6, warn_thr=0.3, mild_thr=0.2,
            enable_vshape=True
        )
        st.dataframe(df, use_container_width=True)
        download_csv(df, "dn_spo2_table.csv")

# --- RR ---
with tabs[2]:
    st.subheader("RR (10 points)")
    default_rr = "16 16 17 18 17 16 15 16 15 16"
    txt = st.text_input("RR series", value=default_rr, key="rr_in")
    vals = parse_series(txt)

    col1, col2 = st.columns(2)
    with col1:
        K_rr = st.number_input("K_rr (% per min)", value=25.0, step=1.0, key="K_rr")
    with col2:
        st.write("Your mapping: 25%/min => |T|=1 (RED); 12–15% => |T|≈0.5 (WARNING); 5–8% => |T|≈0.2–0.3 (MILD).")

    if st.button("Compute", key="btn_rr"):
        df = compute_threshold_table(
            vals, K=K_rr, channel_name="RR",
            red_thr=1.0, warn_thr=0.5, mild_thr=0.2,
            enable_vshape=True
        )
        st.dataframe(df, use_container_width=True)
        download_csv(df, "dn_rr_table.csv")

# --- HR ---
with tabs[3]:
    st.subheader("HR (10 points)")
    default_hr = "70 71 72 90 86 85 84 83 82 81"
    txt = st.text_input("HR series", value=default_hr, key="hr_in")
    vals = parse_series(txt)

    col1, col2 = st.columns(2)
    with col1:
        K_hr = st.number_input("K_hr (% per min)", value=25.0, step=1.0, key="K_hr")
    with col2:
        st.write("Same threshold logic as SpO2 by default: RED≥0.6, WARNING≥0.3, MILD≥0.2 (V-shape => INFO). You may tune K_hr later.")

    if st.button("Compute", key="btn_hr"):
        df = compute_threshold_table(
            vals, K=K_hr, channel_name="HR",
            red_thr=0.6, warn_thr=0.3, mild_thr=0.2,
            enable_vshape=True
        )
        st.dataframe(df, use_container_width=True)
        download_csv(df, "dn_hr_table.csv")
