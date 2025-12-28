import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Core parsing / computation
# =========================

def parse_series(text: str) -> List[float]:
    """
    Parse a user-entered numeric series.
    Accepts spaces, commas, semicolons, newlines.
    """
    if not text or not text.strip():
        return []
    parts = re.split(r"[,\s;]+", text.strip())
    vals = []
    for p in parts:
        if p == "":
            continue
        vals.append(float(p))
    return vals


def safe_pct_step(prev: float, cur: float) -> float:
    """
    pct_step = 100*(cur - prev)/prev
    - If prev == 0 -> return 0 (avoid division blow-up).
    """
    if prev == 0:
        return 0.0
    return 100.0 * (cur - prev) / prev


def compute_table(values: List[float], K: float) -> pd.DataFrame:
    """
    Compute the DN-dynamic table:
    pct_step -> T -> E -> vT/vE
    """
    n = len(values)
    if n == 0:
        return pd.DataFrame(columns=["minute", "value", "pct_step", "t", "e", "vT", "vE", "status", "note"])

    pct = [0.0]
    for i in range(1, n):
        pct.append(safe_pct_step(values[i - 1], values[i]))

    pct = np.array(pct, dtype=float)
    T = pct / float(K)
    E = 1.0 - (T ** 2)

    vT = np.zeros(n, dtype=float)
    vE = np.zeros(n, dtype=float)
    vT[1:] = T[1:] - T[:-1]
    vE[1:] = E[1:] - E[:-1]

    minute = np.arange(1, n + 1)
    status = np.array(["BASE" if i < 2 else "PATTERN" for i in range(n)], dtype=object)

    df = pd.DataFrame(
        {
            "minute": minute,
            "value": np.array(values, dtype=float),
            "pct_step": pct,
            "t": T,
            "e": E,
            "vT": vT,
            "vE": vE,
            "status": status,
            "note": np.array([""] * n, dtype=object),
        }
    )
    return df


# =========================
# Pattern labeling (NOTES)
# =========================

@dataclass
class VShapeRule:
    drop_pct: float            # pct_step <= -drop_pct
    rebound_pct: float         # next pct_step >= +rebound_pct
    net_abs_pct: float         # abs(net change over 2 steps) <= net_abs_pct
    # net change is from i-1 to i+1, normalized by value[i-1]


def apply_vshape_info(df: pd.DataFrame, rule: VShapeRule) -> np.ndarray:
    """
    Mark V-shape points as INFO when:
      pct_step[i] <= -drop_pct AND pct_step[i+1] >= rebound_pct AND
      abs(net_2step_pct) <= net_abs_pct
    Returns boolean mask length n (True means "this point is part of V-shape").
    """
    n = len(df)
    mark = np.zeros(n, dtype=bool)
    if n < 3:
        return mark

    vals = df["value"].values
    pct = df["pct_step"].values

    for i in range(1, n - 1):
        if pct[i] <= -abs(rule.drop_pct) and pct[i + 1] >= abs(rule.rebound_pct):
            prev = vals[i - 1]
            if prev != 0:
                net_2step = 100.0 * (vals[i + 1] - prev) / prev
            else:
                net_2step = 0.0
            if abs(net_2step) <= abs(rule.net_abs_pct):
                mark[i] = True
                mark[i + 1] = True
    return mark


def label_hrv_notes(df: pd.DataFrame) -> pd.DataFrame:
    """
    HRV: no absolute threshold. Use shape only:
    - V-shape recovery => INFO
    - Noise spike (very large up-step) => INFO
    - Sustained drift-down => WARNING/RED
    - Otherwise => GREEN
    """
    n = len(df)
    notes = np.array(["GREEN"] * n, dtype=object)

    # BASE rows: keep GREEN (but you still get computed rows)
    # (you can change BASE note to "" if you prefer, but you said "note trống là sao" => keep non-empty)
    # V-shape (your spirit): drop then rebound, net small
    vmask = apply_vshape_info(df, VShapeRule(drop_pct=20.0, rebound_pct=15.0, net_abs_pct=12.0))
    notes[vmask] = "INFO"  # recovery / transient

    pct = df["pct_step"].values

    # Noise-brake: too-fast rise (sensor spike) -> INFO (and override GREEN)
    # (matches your v1.5 idea but kept minimal and deterministic)
    noise = (pct >= 70.0) | (np.abs(df["value"].diff().fillna(0.0).values) >= 60.0)
    notes[noise] = "INFO"

    # Drift-down detection: 3 consecutive negative steps
    # Severity based on cumulative drop over the last 4 points
    vals = df["value"].values
    for i in range(3, n):
        if (pct[i] < 0) and (pct[i - 1] < 0) and (pct[i - 2] < 0):
            base = vals[i - 3]
            if base != 0:
                total = 100.0 * (vals[i] - base) / base
            else:
                total = 0.0
            if total <= -30.0:
                notes[i] = "RED"
            elif total <= -15.0:
                notes[i] = "WARNING"
            else:
                notes[i] = "MILD"

    # Do not overwrite INFO (INFO has priority for false-alarm / recovery patterns)
    # Priority: INFO > RED > WARNING > MILD > GREEN
    # We'll enforce by re-applying INFO at the end:
    notes[vmask | noise] = "INFO"

    df["note"] = notes
    return df


def label_threshold_notes(df: pd.DataFrame, *,
                          red_T: float,
                          warn_T: float,
                          mild_T: float,
                          vshape_rule: VShapeRule) -> pd.DataFrame:
    """
    Systems with thresholds (SpO2/RR/HR):
    - Use |T| thresholds to set GREEN/MILD/WARNING/RED
    - If V-shape (drop then rebound, net small) => INFO overrides severity (false alarm filter)
    """
    n = len(df)
    notes = np.array(["GREEN"] * n, dtype=object)

    T = df["t"].values
    aT = np.abs(T)

    notes[aT >= mild_T] = "MILD"
    notes[aT >= warn_T] = "WARNING"
    notes[aT >= red_T] = "RED"

    # V-shape override => INFO
    vmask = apply_vshape_info(df, vshape_rule)
    notes[vmask] = "INFO"

    df["note"] = notes
    return df


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="DN v2 Demo (4 tabs)", layout="wide")

st.title("DN v2 Demo — 4 tabs (HRV / SpO2 / RR / HR)")
st.caption("Blank input by default. Paste a series, press Compute, export CSV.")

tab_hrv, tab_spo2, tab_rr, tab_hr = st.tabs(["HRV", "SpO2", "RR", "HR"])


def render_tab(tab, title: str, K: float, mode: str):
    with tab:
        st.subheader(f"{title} (dynamic, 10+ points recommended)")
        series_text = st.text_input(f"{title} series", value="", placeholder="e.g. 48 47 46 45 28 27 26 26 25 25", key=f"{mode}_series")

        if st.button("Compute", key=f"{mode}_btn"):
            values = parse_series(series_text)

            if len(values) < 2:
                st.warning("Please input at least 2 numbers.")
                return

            df = compute_table(values, K=K)

            # Apply notes
            if mode == "hrv":
                df = label_hrv_notes(df)

            elif mode == "spo2":
                # SpO2: use |T| thresholds and V-shape INFO
                # K=5 makes 3% step -> |T|=0.6 (RED) as your spirit example.
                df = label_threshold_notes(
                    df,
                    red_T=0.6,
                    warn_T=0.3,
                    mild_T=0.2,
                    vshape_rule=VShapeRule(drop_pct=3.0, rebound_pct=2.0, net_abs_pct=1.0),
                )

            elif mode == "rr":
                # RR mapping you wrote (per minute):
                # 25% -> |T|=1 (RED), 12-15% -> |T|~0.5 (WARNING), 5-8% -> |T|~0.2-0.3 (MILD)
                df = label_threshold_notes(
                    df,
                    red_T=1.0,
                    warn_T=0.5,
                    mild_T=0.2,
                    vshape_rule=VShapeRule(drop_pct=25.0, rebound_pct=12.0, net_abs_pct=5.0),
                )

            elif mode == "hr":
                # HR: treat as % dynamics (same K=25 as RR for demo consistency),
                # but keep thresholded notes + V-shape INFO so “spike then recover” won't stay RED.
                df = label_threshold_notes(
                    df,
                    red_T=1.0,
                    warn_T=0.5,
                    mild_T=0.2,
                    vshape_rule=VShapeRule(drop_pct=25.0, rebound_pct=12.0, net_abs_pct=8.0),
                )

            # Display
            show_cols = ["minute", "value", "pct_step", "t", "e", "vT", "vE", "status", "note"]
            st.dataframe(df[show_cols], use_container_width=True)

            csv = df[show_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"{mode}_dn_v2.csv",
                mime="text/csv",
                key=f"{mode}_csv",
            )

        # Minimal scientific note (short and clean)
        with st.expander("Scientific note (what is computed)"):
            st.markdown(
                f"""
- **pct_step** = 100×(xᵢ − xᵢ₋₁) / xᵢ₋₁, first point = 0  
- **T** = pct_step / **K** (here K={K:g})  
- **E** = 1 − T²  
- **vT** = Tᵢ − Tᵢ₋₁, **vE** = Eᵢ − Eᵢ₋₁  
- **status**: BASE for first 2 rows, PATTERN afterwards  
- **note**:  
  - HRV: shape-based (drift / V-shape / noise)  
  - SpO2/RR/HR: |T|-threshold + V-shape ⇒ INFO (false-alarm filter)
"""
            )


# Constants (fixed, no extra sliders/inputs)
render_tab(tab_hrv, "HRV", K=80.0, mode="hrv")
render_tab(tab_spo2, "SpO2", K=5.0, mode="spo2")
render_tab(tab_rr, "RR", K=25.0, mode="rr")
render_tab(tab_hr, "HR", K=25.0, mode="hr")
