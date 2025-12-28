# app.py
# DN v2 Demo (4 systems, 10 points each) — table-only, minimal text, hidden formulas in headers
# Keeps the same core calculations + old thresholds; removes charts + extra notes/constant lines.

import streamlit as st
import numpy as np
import pandas as pd

# -----------------------------
# Helpers
# -----------------------------
def parse_series(txt: str, n: int = 10):
    vals = [float(x) for x in txt.strip().replace(",", " ").split() if x.strip() != ""]
    if len(vals) != n:
        raise ValueError(f"Need exactly {n} numbers.")
    return np.array(vals, dtype=float)

def safe_pct_change(x: np.ndarray):
    """%Δ step-by-step, first point = 0. Uses previous value as denominator."""
    pct = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        prev = x[i - 1]
        if prev == 0:
            pct[i] = 0.0
        else:
            pct[i] = 100.0 * (x[i] - prev) / prev
    return pct

def lorentz_from_T(T: np.ndarray):
    E = 1.0 - (T ** 2)
    return E

def velocity(arr: np.ndarray):
    v = np.zeros_like(arr, dtype=float)
    for i in range(1, len(arr)):
        v[i] = arr[i] - arr[i - 1]
    return v

# -----------------------------
# DN logic (keep old thresholds)
# -----------------------------
def dn_hrv_dynamic(hrv: np.ndarray, K_T: float = 80.0):
    """
    HRV: dynamic only, baseline-free.
    Old thresholds:
      - Step-drop RED: any step %Δ <= -40%
      - V-shape recovery INFO: d1 <= -20% and d2 >= +15% and |total| <= 12%
      - Otherwise: GREEN (v2 demo minimal)
    """
    pct = safe_pct_change(hrv)

    # T and E (internal; do not expose formulas in UI headers)
    T = pct / K_T
    E = lorentz_from_T(T)
    vT = velocity(T)
    vE = velocity(E)

    status = ["GREEN"] * len(hrv)
    note = [""] * len(hrv)

    # Step-drop RED (mark the minute where drop occurs)
    for i in range(1, len(pct)):
        if pct[i] <= -40.0:
            status[i] = "RED"
            note[i] = "step-drop"

    # V-shape recovery (mark middle point as INFO if pattern exists)
    for i in range(2, len(pct)):
        d1 = pct[i - 1]
        d2 = pct[i]
        total = 100.0 * (hrv[i] - hrv[i - 2]) / (hrv[i - 2] if hrv[i - 2] != 0 else 1.0)
        if (d1 <= -20.0) and (d2 >= 15.0) and (abs(total) <= 12.0):
            # only override if not already RED
            if status[i - 1] != "RED":
                status[i - 1] = "INFO"
                note[i - 1] = "v-shape"

    df = pd.DataFrame({
        "minute": np.arange(1, len(hrv) + 1),
        "t": T,
        "e": E,
        "vt": vT,
        "ve": vE,
        "status": status,
        "note": note
    })
    return df

def dn_spo2(spo2: np.ndarray, K_abs: float = 5.0):
    """
    SpO2: threshold on absolute change per step.
    Old thresholds (as you wrote):
      |T| >= 0.6 -> RED
      |T| >= 0.3 -> WARNING
      V-shape (drop then immediate recovery) -> INFO (filter false alarm)
    Here T = ΔSpO2 / K_abs (internal).
    """
    d = np.zeros_like(spo2, dtype=float)
    for i in range(1, len(spo2)):
        d[i] = spo2[i] - spo2[i - 1]  # absolute delta

    T = d / K_abs
    E = lorentz_from_T(T)
    vT = velocity(T)
    vE = velocity(E)

    status = ["GREEN"] * len(spo2)
    note = [""] * len(spo2)

    for i in range(1, len(T)):
        a = abs(T[i])
        if a >= 0.6:
            status[i] = "RED"
            note[i] = "threshold"
        elif a >= 0.3:
            status[i] = "WARNING"
            note[i] = "threshold"

    # V-shape: strong drop then immediate recovery (filter false alarm)
    for i in range(2, len(T)):
        if (T[i - 1] <= -0.6) and (T[i] >= 0.6):
            # override both points to INFO unless something else is clearly worse later
            status[i - 1] = "INFO"
            note[i - 1] = "v-shape"
            status[i] = "INFO"
            note[i] = "v-shape"

    df = pd.DataFrame({
        "minute": np.arange(1, len(spo2) + 1),
        "t": T,
        "e": E,
        "vt": vT,
        "ve": vE,
        "status": status,
        "note": note
    })
    return df

def dn_rr(rr: np.ndarray, K_pct: float = 25.0):
    """
    RR: threshold on %Δ per step.
    Mapping you wrote:
      RR +25%/min -> |T|=1 => RED
      RR +12~15%  -> |T|~0.5 => WARNING
      RR +5~8%    -> |T|~0.2~0.3 => light
    Here T = %ΔRR / K_pct (internal).
    """
    pct = safe_pct_change(rr)
    T = pct / K_pct
    E = lorentz_from_T(T)
    vT = velocity(T)
    vE = velocity(E)

    status = ["GREEN"] * len(rr)
    note = [""] * len(rr)

    for i in range(1, len(T)):
        a = abs(T[i])
        if a >= 1.0:
            status[i] = "RED"
            note[i] = "threshold"
        elif a >= 0.5:
            status[i] = "WARNING"
            note[i] = "threshold"
        elif a >= 0.2:
            status[i] = "INFO"
            note[i] = "light"

    df = pd.DataFrame({
        "minute": np.arange(1, len(rr) + 1),
        "t": T,
        "e": E,
        "vt": vT,
        "ve": vE,
        "status": status,
        "note": note
    })
    return df

def dn_hr(hr: np.ndarray, K_pct: float = 15.0):
    """
    HR: threshold on %Δ per step using K=15% (your current choice).
    Here T = %ΔHR / K_pct (internal).
    """
    pct = safe_pct_change(hr)
    T = pct / K_pct
    E = lorentz_from_T(T)
    vT = velocity(T)
    vE = velocity(E)

    status = ["GREEN"] * len(hr)
    note = [""] * len(hr)

    for i in range(1, len(T)):
        a = abs(T[i])
        if a >= 1.0:
            status[i] = "RED"
            note[i] = "threshold"
        elif a >= 0.5:
            status[i] = "WARNING"
            note[i] = "threshold"
        elif a >= 0.2:
            status[i] = "INFO"
            note[i] = "light"

    df = pd.DataFrame({
        "minute": np.arange(1, len(hr) + 1),
        "t": T,
        "e": E,
        "vt": vT,
        "ve": vE,
        "status": status,
        "note": note
    })
    return df

# -----------------------------
# Streamlit UI (minimal)
# -----------------------------
st.set_page_config(page_title="DN v2 Demo", layout="wide")
st.title("DN v2 Demo")

tab_hrv, tab_spo2, tab_rr, tab_hr = st.tabs(["hrv", "spo2", "rr", "hr"])

with tab_hrv:
    txt = st.text_input("HRV (10 points)", value="42 41 50 30 27 28 29 28 26 25")
    if st.button("Compute", key="btn_hrv"):
        try:
            x = parse_series(txt, 10)
            df = dn_hrv_dynamic(x, K_T=80.0)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(str(e))

with tab_spo2:
    txt = st.text_input("SpO2 (10 points)", value="98 97 96 95 94 93 94 95 95 95")
    if st.button("Compute", key="btn_spo2"):
        try:
            x = parse_series(txt, 10)
            df = dn_spo2(x, K_abs=5.0)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(str(e))

with tab_rr:
    txt = st.text_input("RR (10 points)", value="16 16 17 18 19 20 21 22 22 23")
    if st.button("Compute", key="btn_rr"):
        try:
            x = parse_series(txt, 10)
            df = dn_rr(x, K_pct=25.0)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(str(e))

with tab_hr:
    txt = st.text_input("HR (10 points)", value="75 76 110 77 78 82 88 95 103 112")
    if st.button("Compute", key="btn_hr"):
        try:
            x = parse_series(txt, 10)
            df = dn_hr(x, K_pct=15.0)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(str(e))
