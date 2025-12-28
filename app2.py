# app.py
import re
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Parsing
# -----------------------------
def parse_10_points(text: str):
    """
    Parse 10 numbers separated by spaces/commas/newlines.
    Returns list[float] length=10.
    """
    if text is None:
        raise ValueError("Empty input")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.strip())
    if len(nums) != 10:
        raise ValueError(f"Please input exactly 10 numbers (got {len(nums)}).")
    vals = [float(x) for x in nums]
    return vals

# -----------------------------
# Core math (shared)
# -----------------------------
def compute_core(values, k: float):
    """
    Core computation:
      pct[i] = 100*(v[i]-v[i-1]) / v[i-1]
      t[i]   = pct[i] / k
      e[i]   = 1 - t[i]^2
      vt[i]  = t[i] - t[i-1]
      ve[i]  = e[i] - e[i-1]

    Notes:
    - Row 0 uses pct=0, t=0, e=1, vt=0, ve=0
    - No rounding inside core (round only for display)
    """
    v = np.array(values, dtype=float)

    if np.any(v[:-1] == 0):
        raise ValueError("Zero previous value found -> percent-change undefined.")

    pct = np.zeros_like(v, dtype=float)
    pct[1:] = 100.0 * (v[1:] - v[:-1]) / v[:-1]

    t = np.zeros_like(v, dtype=float)
    t[1:] = pct[1:] / float(k)

    e = np.ones_like(v, dtype=float)
    e[1:] = 1.0 - np.square(t[1:])

    vt = np.zeros_like(v, dtype=float)
    ve = np.zeros_like(v, dtype=float)
    vt[1:] = t[1:] - t[:-1]
    ve[1:] = e[1:] - e[:-1]

    df = pd.DataFrame({
        "minute": np.arange(1, len(v) + 1, dtype=int),
        "value": v,
        "t": t,
        "e": e,
        "vt": vt,
        "ve": ve,
        "pct": pct,   # internal for rules; will drop from display
    })
    return df

# -----------------------------
# Pattern helpers
# -----------------------------
def is_v_shape(pct_prev, pct_now, drop_thr, rec_thr):
    """
    V-shape: one step drops <= -drop_thr then next step recovers >= +rec_thr
    """
    return (pct_prev <= -abs(drop_thr)) and (pct_now >= abs(rec_thr))

def detect_drift_sign(pct_series, n=3):
    """
    Simple drift detector for v2:
    if last n pct are all negative -> drift-down
    if last n pct are all positive -> drift-up
    else None
    """
    if len(pct_series) < n:
        return None
    tail = pct_series[-n:]
    if np.all(tail < 0):
        return "drift-down"
    if np.all(tail > 0):
        return "drift-up"
    return None

# -----------------------------
# Status/Note rules (v2)
# -----------------------------
def apply_rules_hrv(df: pd.DataFrame):
    """
    HRV v2:
    - NO absolute threshold; NO |t| severity labels.
    - status: BASE / PATTERN
    - note: step-drop / v-shape / drift / noise-spike (optional)
    Noise-brake (v1.5 idea): if pct jump >= +70% OR abs(value jump) >= 60 -> note "noise-spike (INFO)"
    Step-drop: pct <= -40% -> note "step-drop"
    V-shape: drop then recover -> note "v-shape (INFO)"
    Drift: last 3 pct negative -> note "drift-down"
    """
    out = df.copy()

    status = ["BASE"] * len(out)
    note = [""] * len(out)

    pct = out["pct"].to_numpy()
    val = out["value"].to_numpy()

    for i in range(1, len(out)):
        status[i] = "PATTERN"

        # noise-spike (INFO)
        abs_jump = abs(val[i] - val[i-1])
        if (pct[i] >= 70.0) or (abs_jump >= 60.0):
            note[i] = "noise-spike (INFO)"
            continue

        # step-drop
        if pct[i] <= -40.0:
            note[i] = "step-drop"
            continue

        # v-shape (lookback one step)
        if i >= 2 and is_v_shape(pct[i-1], pct[i], drop_thr=20.0, rec_thr=15.0):
            note[i] = "v-shape (INFO)"
            continue

        # drift (last 3 steps)
        drift = detect_drift_sign(pct[1:i+1], n=3)
        if drift is not None:
            note[i] = drift

    out["status"] = status
    out["note"] = note
    return out

def apply_rules_thresholded(df: pd.DataFrame, mild_thr=0.2, warn_thr=0.3, red_thr=0.6):
    """
    For SpO2 / RR / HR in v2:
    - status: BASE / PATTERN
    - note: MILD / WARNING / RED based on |t|
    - V-shape: if drop then recover -> INFO (filter false alarm) overrides severity at that step

    Severity is on |t| (dimensionless):
      |t| >= 0.6 -> RED
      |t| >= 0.3 -> WARNING
      |t| >= 0.2 -> MILD
      else       -> ""
    """
    out = df.copy()

    status = ["BASE"] * len(out)
    note = [""] * len(out)

    t = out["t"].to_numpy()
    pct = out["pct"].to_numpy()

    for i in range(1, len(out)):
        status[i] = "PATTERN"
        at = abs(t[i])

        # V-shape INFO: drop then recover quickly -> filter false alarm
        if i >= 2 and is_v_shape(pct[i-1], pct[i], drop_thr=20.0, rec_thr=15.0):
            note[i] = "v-shape (INFO)"
            continue

        if at >= red_thr:
            note[i] = "RED"
        elif at >= warn_thr:
            note[i] = "WARNING"
        elif at >= mild_thr:
            note[i] = "MILD"
        else:
            note[i] = ""

    out["status"] = status
    out["note"] = note
    return out

# -----------------------------
# Rendering
# -----------------------------
def render_tab(title, default_text, k, rule_fn):
    st.subheader(title)
    txt = st.text_input("", value=default_text, label_visibility="collapsed")
    if st.button("Compute", key=f"btn_{title}"):
        try:
            values = parse_10_points(txt)
            df = compute_core(values, k=k)
            df = rule_fn(df)

            # Drop pct from display (keep only requested columns)
            df_show = df[["minute", "value", "t", "e", "vt", "ve", "status", "note"]].copy()

            # Round ONLY for display (avoid ve==0 artifacts from rounding too early)
            df_show["value"] = df_show["value"].round(3)
            df_show["t"] = df_show["t"].round(4)
            df_show["e"] = df_show["e"].round(4)
            df_show["vt"] = df_show["vt"].round(4)
            df_show["ve"] = df_show["ve"].round(4)

            st.dataframe(df_show, use_container_width=True)

            csv = df_show.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name=f"{title.lower()}_dn_v2.csv", mime="text/csv")

        except Exception as e:
            st.error(str(e))

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="DN v2 Demo", layout="wide")

tabs = st.tabs(["hrv", "spo2", "rr", "hr"])

# Constants (hidden from UI text; used internally)
K_HRV = 80.0
K_SPO2 = 5.0
K_RR = 25.0
K_HR = 15.0

with tabs[0]:
    render_tab(
        title="HRV (10 points)",
        default_text="45 44 43 30 44 45 46 46 45 45",
        k=K_HRV,
        rule_fn=apply_rules_hrv
    )

with tabs[1]:
    render_tab(
        title="SpO2 (10 points)",
        default_text="98 97 96 95 94 93 92 91 90 89",
        k=K_SPO2,
        rule_fn=apply_rules_thresholded
    )

with tabs[2]:
    render_tab(
        title="RR (10 points)",
        default_text="16 16 17 18 34 17 16 15 16 18",
        k=K_RR,
        rule_fn=apply_rules_thresholded
    )

with tabs[3]:
    render_tab(
        title="HR (10 points)",
        default_text="75 76 110 77 78 82 88 95 103 112",
        k=K_HR,
        rule_fn=apply_rules_thresholded
    )
