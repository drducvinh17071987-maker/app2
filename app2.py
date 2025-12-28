import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DN v2 Demo", layout="wide")

# ----------------------------
# Helpers (keep paper-clean)
# ----------------------------
def parse_series(text: str, n_expected: int = 10):
    vals = [v for v in text.replace(",", " ").split() if v.strip() != ""]
    try:
        x = [float(v) for v in vals]
    except Exception:
        return None, "Input must be numbers separated by spaces."
    if len(x) != n_expected:
        return None, f"Please enter exactly {n_expected} values."
    return x, None

def compute_t_e_vt_ve(t: np.ndarray):
    # E from T (Lorentz form), then first-difference velocities
    e = 1.0 - np.square(t)
    vt = np.zeros_like(t)
    ve = np.zeros_like(t)
    vt[1:] = t[1:] - t[:-1]
    ve[1:] = e[1:] - e[:-1]
    return e, vt, ve

def detect_vshape(t: np.ndarray, d1_thr: float, d2_thr: float, total_abs_thr: float):
    """
    V-shape pattern on T:
    - drop then rebound: t[i] <= -d1_thr and t[i+1] >= d2_thr
    - and total change small: abs(t[i] + t[i+1]) <= total_abs_thr
    """
    flags = np.zeros_like(t, dtype=bool)
    for i in range(1, len(t) - 1):
        if (t[i] <= -d1_thr) and (t[i+1] >= d2_thr) and (abs(t[i] + t[i+1]) <= total_abs_thr):
            flags[i] = True
            flags[i+1] = True
    return flags

# ----------------------------
# HRV (dynamic only; no absolute threshold)
# ----------------------------
def dn_hrv(values: np.ndarray):
    # %Δ HRV step (baseline-free dynamic)
    pct = np.zeros_like(values, dtype=float)
    pct[1:] = 100.0 * (values[1:] - values[:-1]) / np.maximum(values[:-1], 1e-9)

    # Normalize to T (keep paper-clean: do NOT display formula in table headers)
    t = pct / 80.0
    e, vt, ve = compute_t_e_vt_ve(t)

    status = np.array(["GREEN"] * len(values), dtype=object)
    note = np.array([""] * len(values), dtype=object)

    # Step-drop RED (Core v1): any step <= -40%
    step_drop = pct <= -40.0
    status[step_drop] = "RED"
    note[step_drop] = "step-drop"

    # Noise spike / sudden jump INFO (v1.5 idea): step increase too fast
    noise = (pct >= 70.0) | ((values[1:] - values[:-1]) >= 60.0)
    noise = np.insert(noise, 0, False)
    status[(status != "RED") & noise] = "INFO"
    note[(status == "INFO") & noise] = "possible-noise"

    # V-shape recovery INFO (Core v1 style, expressed on T)
    vshape = detect_vshape(t, d1_thr=0.20, d2_thr=0.15, total_abs_thr=0.12)
    mask_v = (status != "RED") & vshape
    status[mask_v] = "INFO"
    note[mask_v] = "v-shape"

    # Drift-down WARNING (gentle, not ML): 3 consecutive negative steps (excluding RED/INFO)
    # This is optional but useful for paper; it does NOT use absolute HRV.
    for i in range(3, len(values)):
        if status[i] in ("RED", "INFO"):
            continue
        if pct[i] < 0 and pct[i-1] < 0 and pct[i-2] < 0:
            status[i] = "WARNING"
            note[i] = "drift-down"

    df = pd.DataFrame({
        "minute": np.arange(1, len(values) + 1),
        "value": values.astype(float),
        "t": np.round(t, 6),
        "e": np.round(e, 6),
        "vt": np.round(vt, 6),
        "ve": np.round(ve, 6),
        "status": status,
        "note": note
    })
    return df

# ----------------------------
# SpO2 (absolute Δ/k, threshold on |t|)
# ----------------------------
def dn_spo2(values: np.ndarray, k_abs: float = 5.0):
    d = np.zeros_like(values, dtype=float)
    d[1:] = values[1:] - values[:-1]
    t = d / k_abs
    e, vt, ve = compute_t_e_vt_ve(t)

    status = np.array(["GREEN"] * len(values), dtype=object)
    note = np.array([""] * len(values), dtype=object)

    at = np.abs(t)
    status[at >= 0.6] = "RED"
    note[at >= 0.6] = "threshold"
    mid = (at >= 0.3) & (at < 0.6)
    status[mid] = "WARNING"
    note[mid] = "threshold"

    # V-shape -> INFO (filter false alarm)
    vshape = detect_vshape(t, d1_thr=0.30, d2_thr=0.30, total_abs_thr=0.20)
    mask = (status != "RED") & vshape
    status[mask] = "INFO"
    note[mask] = "v-shape"

    df = pd.DataFrame({
        "minute": np.arange(1, len(values) + 1),
        "value": values.astype(float),
        "t": np.round(t, 6),
        "e": np.round(e, 6),
        "vt": np.round(vt, 6),
        "ve": np.round(ve, 6),
        "status": status,
        "note": note
    })
    return df

# ----------------------------
# RR (percent Δ / k_pct)
# ----------------------------
def dn_rr(values: np.ndarray, k_pct: float = 25.0):
    pct = np.zeros_like(values, dtype=float)
    pct[1:] = 100.0 * (values[1:] - values[:-1]) / np.maximum(values[:-1], 1e-9)
    t = pct / k_pct
    e, vt, ve = compute_t_e_vt_ve(t)

    status = np.array(["GREEN"] * len(values), dtype=object)
    note = np.array([""] * len(values), dtype=object)

    at = np.abs(t)
    status[at >= 1.0] = "RED"
    note[at >= 1.0] = "threshold"
    mid = (at >= 0.5) & (at < 1.0)
    status[mid] = "WARNING"
    note[mid] = "threshold"

    vshape = detect_vshape(t, d1_thr=0.50, d2_thr=0.50, total_abs_thr=0.30)
    mask = (status != "RED") & vshape
    status[mask] = "INFO"
    note[mask] = "v-shape"

    df = pd.DataFrame({
        "minute": np.arange(1, len(values) + 1),
        "value": values.astype(float),
        "t": np.round(t, 6),
        "e": np.round(e, 6),
        "vt": np.round(vt, 6),
        "ve": np.round(ve, 6),
        "status": status,
        "note": note
    })
    return df

# ----------------------------
# HR (percent Δ / k_pct)
# ----------------------------
def dn_hr(values: np.ndarray, k_pct: float = 15.0):
    pct = np.zeros_like(values, dtype=float)
    pct[1:] = 100.0 * (values[1:] - values[:-1]) / np.maximum(values[:-1], 1e-9)
    t = pct / k_pct
    e, vt, ve = compute_t_e_vt_ve(t)

    status = np.array(["GREEN"] * len(values), dtype=object)
    note = np.array([""] * len(values), dtype=object)

    at = np.abs(t)
    status[at >= 1.0] = "RED"
    note[at >= 1.0] = "threshold"
    mid = (at >= 0.5) & (at < 1.0)
    status[mid] = "WARNING"
    note[mid] = "threshold"

    vshape = detect_vshape(t, d1_thr=0.50, d2_thr=0.50, total_abs_thr=0.30)
    mask = (status != "RED") & vshape
    status[mask] = "INFO"
    note[mask] = "v-shape"

    df = pd.DataFrame({
        "minute": np.arange(1, len(values) + 1),
        "value": values.astype(float),
        "t": np.round(t, 6),
        "e": np.round(e, 6),
        "vt": np.round(vt, 6),
        "ve": np.round(ve, 6),
        "status": status,
        "note": note
    })
    return df

# ----------------------------
# UI
# ----------------------------
tabs = st.tabs(["hrv", "spo2", "rr", "hr"])

def render_tab(tab, label, default_text, compute_fn):
    with tab:
        text = st.text_input(f"{label} (10 points)", value=default_text)
        if st.button("Compute", key=f"btn_{label}"):
            arr, err = parse_series(text, 10)
            if err:
                st.error(err)
                return
            values = np.array(arr, dtype=float)
            df = compute_fn(values)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"dn_{label}_10points.csv",
                mime="text/csv",
                key=f"dl_{label}"
            )

render_tab(tabs[0], "HRV",  "50 49 48 47 46 45 44 43 42 41", dn_hrv)
render_tab(tabs[1], "SpO2", "98 97 96 95 94 93 92 91 90 89", lambda x: dn_spo2(x, k_abs=5.0))
render_tab(tabs[2], "RR",   "16 16 17 18 20 22 24 26 28 30", lambda x: dn_rr(x, k_pct=25.0))
render_tab(tabs[3], "HR",   "75 76 110 77 78 82 88 95 103 112", lambda x: dn_hr(x, k_pct=15.0))
