# app2.py
# DN v2 Demo – Table (HRV / SpO2 / RR / HR) – clean for paper
# Core: pct_step -> T -> E=1-T^2 -> vT/vE ; scientific status/note

import math
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Helpers
# -----------------------------
def parse_series(text: str) -> List[float]:
    """
    Parse a whitespace/comma/semicolon separated list of numbers.
    """
    if text is None:
        return []
    cleaned = re.sub(r"[,\;\t]+", " ", text.strip())
    parts = [p for p in cleaned.split(" ") if p != ""]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except:
            pass
    return vals


def safe_pct_step(values: np.ndarray) -> np.ndarray:
    """
    pct_step[i] = 100*(x_i - x_{i-1})/x_{i-1}, with pct_step[0]=0.
    Protect divide-by-zero.
    """
    pct = np.zeros(len(values), dtype=float)
    for i in range(1, len(values)):
        prev = values[i - 1]
        if prev == 0:
            pct[i] = 0.0
        else:
            pct[i] = 100.0 * (values[i] - prev) / prev
    return pct


def lorentz_from_pct(pct_step: np.ndarray, k: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    T = pct_step / k  (IMPORTANT: pct_step is already in %, DO NOT divide by 100 again)
    E = 1 - T^2
    """
    T = pct_step / float(k)
    E = 1.0 - np.square(T)
    return T, E


def velocity(arr: np.ndarray) -> np.ndarray:
    """
    v[i] = arr[i] - arr[i-1], v[0]=0
    """
    v = np.zeros(len(arr), dtype=float)
    v[1:] = arr[1:] - arr[:-1]
    return v


def is_vshape(pct_step: np.ndarray, i: int,
              d1_thr: float, d2_thr: float, total_abs_thr: float) -> bool:
    """
    V-shape pattern over two steps: i is down-step, i+1 is recovery-step.
    Conditions:
      pct_step[i] <= d1_thr (negative)
      pct_step[i+1] >= d2_thr (positive)
      abs(pct_step[i] + pct_step[i+1]) <= total_abs_thr
    """
    if i < 1 or i + 1 >= len(pct_step):
        return False
    return (pct_step[i] <= d1_thr) and (pct_step[i + 1] >= d2_thr) and (abs(pct_step[i] + pct_step[i + 1]) <= total_abs_thr)


def note_by_absT(absT: float) -> str:
    """
    For threshold-based systems (SpO2/RR/HR):
      |T| >= 0.6 -> RED
      |T| >= 0.3 -> WARNING
      |T| >= 0.2 -> MILD
      else GREEN
    """
    if absT >= 0.6:
        return "RED"
    if absT >= 0.3:
        return "WARNING"
    if absT >= 0.2:
        return "MILD"
    return "GREEN"


# -----------------------------
# HRV logic (pattern-based, baseline-free)
# -----------------------------
def compute_hrv(values: List[float]) -> pd.DataFrame:
    """
    HRV: no absolute threshold.
    Use pattern shape:
      - noise-brake: step >= +70% OR abs(delta)>=60ms => INFO
      - V-shape: d1<=-20, d2>=+15, |total|<=12 => INFO
      - step-drop: any step<=-40% => RED
      - drift-down: last 3 steps all negative and total<=-15% => WARNING
      - else GREEN
    status:
      BASE for first row OR pct_step==0; else PATTERN
    """
    x = np.array(values, dtype=float)
    n = len(x)
    if n == 0:
        return pd.DataFrame()

    pct = safe_pct_step(x)

    # DN core (as you locked): K=80 for HRV dynamic
    T, E = lorentz_from_pct(pct, k=80.0)
    vT = velocity(T)
    vE = velocity(E)

    status = []
    note = [""] * n

    # status: BASE if first row or no change
    for i in range(n):
        if i == 0 or abs(pct[i]) < 1e-12:
            status.append("BASE")
        else:
            status.append("PATTERN")

    # pattern detection
    # 1) noise-brake (sensor noise / posture artifact)
    for i in range(1, n):
        delta = x[i] - x[i - 1]
        if (pct[i] >= 70.0) or (abs(delta) >= 60.0):
            note[i] = "INFO: possible sensor noise (noise-brake)"

    # 2) V-shape recovery override (apply to recovery point i+1)
    for i in range(1, n - 1):
        if is_vshape(pct, i, d1_thr=-20.0, d2_thr=15.0, total_abs_thr=12.0):
            # mark recovery point as INFO
            note[i + 1] = "INFO: V-shape recovery (transient drop)"

    # 3) step-drop RED (true abrupt drop)
    for i in range(1, n):
        if pct[i] <= -40.0 and note[i] == "":
            note[i] = "RED: step-drop (<= -40%)"

    # 4) drift-down WARNING (persistent decline)
    for i in range(3, n):
        last3 = pct[i-2:i+1]
        total3 = float(np.sum(last3))
        if np.all(last3 < 0) and total3 <= -15.0:
            if note[i] == "":
                note[i] = "WARNING: drift-down (3-step negative)"

    # 5) fill empty notes with GREEN (for paper consistency)
    for i in range(n):
        if note[i] == "":
            note[i] = "GREEN"

    df = pd.DataFrame({
        "minute": list(range(1, n + 1)),
        "value": x,
        "pct_step": pct,
        "t": T,
        "e": E,
        "vT": vT,
        "vE": vE,
        "status": status,
        "note": note
    })
    return df


# -----------------------------
# Threshold-based logic (SpO2 / RR / HR) + V-shape INFO override
# -----------------------------
def compute_threshold_system(values: List[float], k: float, label: str) -> pd.DataFrame:
    """
    Generic for SpO2/RR/HR:
      pct_step -> T=pct/k ; E=1-T^2 ; vT/vE
      note by |T| (RED/WARNING/MILD/GREEN)
      V-shape: (down then recover quickly) => INFO at recovery point
        use same V-shape rule: d1<=-20, d2>=+15, |total|<=12  (in pct domain)
      status: BASE for first row OR pct_step==0 else PATTERN
    """
    x = np.array(values, dtype=float)
    n = len(x)
    if n == 0:
        return pd.DataFrame()

    pct = safe_pct_step(x)
    T, E = lorentz_from_pct(pct, k=k)
    vT = velocity(T)
    vE = velocity(E)

    status = []
    note = [""] * n
    for i in range(n):
        if i == 0 or abs(pct[i]) < 1e-12:
            status.append("BASE")
        else:
            status.append("PATTERN")

    # Base threshold note
    for i in range(n):
        if i == 0:
            note[i] = "GREEN"
        else:
            note[i] = note_by_absT(abs(float(T[i])))

    # V-shape override (apply INFO to recovery point only)
    for i in range(1, n - 1):
        if is_vshape(pct, i, d1_thr=-20.0, d2_thr=15.0, total_abs_thr=12.0):
            note[i + 1] = "INFO: V-shape recovery (transient)"

    df = pd.DataFrame({
        "minute": list(range(1, n + 1)),
        "value": x,
        "pct_step": pct,
        "t": T,
        "e": E,
        "vT": vT,
        "vE": vE,
        "status": status,
        "note": note
    })
    return df


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="DN v2 Demo – Table", layout="wide")
st.title("DN v2 Demo – Table (HRV / SpO2 / RR / HR)")

tabs = st.tabs(["HRV", "SpO2", "RR", "HR"])

# Defaults (you can change the example strings anytime)
default_hrv = "50 49 48 33 32 31 44 43 42 41"
default_spo2 = "98 97 96 77 94 93 92 91 90 89"
default_rr = "16 16 17 33 17 16 15 16 15 16"
default_hr = "70 71 72 123 86 85 84 83 82 81"

# Internal constants (NOT shown in UI to keep paper clean)
K_HRV = 80.0
K_SPO2 = 5.0     # matches your screenshot: T ≈ (%ΔSpO2)/5
K_RR = 25.0      # you locked: K_rr=25 (%ΔRR per min)
K_HR = 25.0      # consistent with your screenshot: T ≈ (%ΔHR)/25


def render_tab(tab_name: str, default_text: str, compute_fn):
    with st.container():
        st.subheader(f"{tab_name} (10 points)")
        text = st.text_input(f"{tab_name} series", value=default_text, key=f"in_{tab_name}")
        col1, col2 = st.columns([1, 2])
        with col1:
            run = st.button("Compute", key=f"btn_{tab_name}")
        df = None
        if run:
            vals = parse_series(text)
            if len(vals) < 2:
                st.error("Please input at least 2 points.")
                return
            df = compute_fn(vals)

            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name=f"{tab_name.lower()}_dn_table.csv", mime="text/csv")


# HRV
with tabs[0]:
    render_tab("HRV", default_hrv, compute_hrv)

# SpO2
with tabs[1]:
    render_tab("SpO2", default_spo2, lambda v: compute_threshold_system(v, k=K_SPO2, label="SpO2"))

# RR
with tabs[2]:
    render_tab("RR", default_rr, lambda v: compute_threshold_system(v, k=K_RR, label="RR"))

# HR
with tabs[3]:
    render_tab("HR", default_hr, lambda v: compute_threshold_system(v, k=K_HR, label="HR"))
