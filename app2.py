# app2.py
# DN v2 Demo (4 systems, 10 points each) — SSRN-ready table output
# Tabs: hrv / spo2 / rr / hr
# Output columns only: minute, value, t, e, vt, ve, status, note
# No formula strings shown in UI.

import re
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------
# Helpers
# ---------------------------
def _parse_10_numbers(text: str):
    nums = re.findall(r"-?\d+(?:\.\d+)?", (text or "").strip())
    if len(nums) != 10:
        raise ValueError("Please input exactly 10 numbers (space-separated).")
    return [float(x) for x in nums]


def _safe_pct_change(values):
    """%Δ between consecutive points. First point = 0.
    pct[i] = 100*(v[i]-v[i-1])/v[i-1]
    """
    pct = [0.0]
    for i in range(1, len(values)):
        prev = values[i - 1]
        cur = values[i]
        if prev == 0:
            pct.append(0.0)
        else:
            pct.append(100.0 * (cur - prev) / prev)
    return pct


def _safe_abs_delta(values):
    """Δ between consecutive points. First point = 0."""
    d = [0.0]
    for i in range(1, len(values)):
        d.append(values[i] - values[i - 1])
    return d


def _compute_core_from_series(values, mode: str):
    """
    mode:
      - "hrv": dynamic (%Δ-based), k=80
      - "rr":  dynamic (%Δ-based), k=25
      - "hr":  dynamic (%Δ-based), k=15
      - "spo2": delta-based (Δ-based), k=5
    returns: df with minute,value,t,e,vt,ve + internal arrays
    """
    n = len(values)
    minute = list(range(1, n + 1))

    if mode == "spo2":
        # SpO2: use absolute delta per step (Δ), then normalized by k
        k = 5.0
        delta = _safe_abs_delta(values)
        t = [d / k for d in delta]
    elif mode == "rr":
        k = 25.0
        pct = _safe_pct_change(values)
        t = [p / k for p in pct]
    elif mode == "hr":
        k = 15.0
        pct = _safe_pct_change(values)
        t = [p / k for p in pct]
    else:  # hrv
        k = 80.0
        pct = _safe_pct_change(values)
        t = [p / k for p in pct]

    # Lorentz-style energy proxy (kept internal; only numbers shown)
    e = [1.0 - (ti ** 2) for ti in t]

    # velocities
    vt = [0.0] + [t[i] - t[i - 1] for i in range(1, n)]
    ve = [0.0] + [e[i] - e[i - 1] for i in range(1, n)]

    df = pd.DataFrame(
        {
            "minute": minute,
            "value": values,
            "t": t,
            "e": e,
            "vt": vt,
            "ve": ve,
        }
    )

    # keep internals for pattern/notes
    internals = {
        "t": np.array(t, dtype=float),
        "e": np.array(e, dtype=float),
        "vt": np.array(vt, dtype=float),
        "ve": np.array(ve, dtype=float),
        "values": np.array(values, dtype=float),
        "mode": mode,
    }
    if mode in ("hrv", "rr", "hr"):
        internals["pct"] = np.array(_safe_pct_change(values), dtype=float)
        internals["delta"] = np.array(_safe_abs_delta(values), dtype=float)
    else:
        internals["delta"] = np.array(_safe_abs_delta(values), dtype=float)

    return df, internals


# ---------------------------
# v2 labeling (STATUS + NOTE)
# ---------------------------
def _status_base_or_pattern(ti, vti, vei, base_eps=1e-6):
    # Strict: if truly zero-ish dynamics, call BASE; else PATTERN
    if abs(ti) <= base_eps and abs(vti) <= base_eps and abs(vei) <= base_eps:
        return "BASE"
    return "PATTERN"


def _detect_vshape_generic(t_arr, i, warn_thr):
    """
    Generic 2-step V-shape:
      - step i is "bad" (|t| >= warn_thr)
      - step i+1 rebounds opposite sign (t_i * t_{i+1} < 0)
      - net cancels well (|t_i + t_{i+1}| <= 0.25*max(|t_i|,|t_{i+1}|)
    """
    if i < 1 or i + 1 >= len(t_arr):
        return False
    a = t_arr[i]
    b = t_arr[i + 1]
    if abs(a) < warn_thr:
        return False
    if a * b >= 0:
        return False
    denom = max(abs(a), abs(b), 1e-9)
    if abs(a + b) <= 0.25 * denom:
        return True
    return False


def _label_hrv_v2(df, internals):
    """
    HRV v2:
      - NO absolute thresholds, NO RED/WARNING labels
      - NOTE only: drift-down / drift-up / step-drop / v-shape / noise-spike
      - STATUS: BASE vs PATTERN (pure dynamics)
    """
    pct = internals["pct"]
    delta = internals["delta"]
    t = internals["t"]
    vt = internals["vt"]
    ve = internals["ve"]
    n = len(t)

    status = []
    note = [""] * n

    # --- noise brake (baseline-free): huge positive jump likely sensor artifact
    # (v1.5 idea) step increase >= 70% OR abs(delta) >= 60 ms
    for i in range(n):
        status.append(_status_base_or_pattern(t[i], vt[i], ve[i], base_eps=1e-9))

    for i in range(1, n):
        if (pct[i] >= 70.0) or (abs(delta[i]) >= 60.0):
            note[i] = "noise-spike"

    # --- step-drop: any single step drop <= -40%
    for i in range(1, n):
        if pct[i] <= -40.0 and note[i] == "":
            note[i] = "step-drop"

    # --- V-shape: two-step drop then recover (baseline-free)
    # d1 <= -20%, d2 >= +15%, and |(d1+d2)| <= 12%
    for i in range(1, n - 1):
        d1 = pct[i]
        d2 = pct[i + 1]
        if (d1 <= -20.0) and (d2 >= 15.0) and (abs(d1 + d2) <= 12.0):
            if note[i] == "":
                note[i] = "v-shape"
            if note[i + 1] == "":
                note[i + 1] = "v-shape"

    # --- drift: 3 consecutive negative (or positive) small-to-moderate steps
    # (avoid marking when there is already step-drop/noise/v-shape)
    def is_clean(i):
        return note[i] == ""

    for i in range(3, n):
        window = pct[i - 2 : i + 1]  # 3 steps
        if all(is_clean(j) for j in range(i - 2, i + 1)):
            # drift-down
            if np.all(window < 0) and (np.median(np.abs(window)) <= 15.0):
                for j in range(i - 2, i + 1):
                    if note[j] == "":
                        note[j] = "drift-down"
            # drift-up
            if np.all(window > 0) and (np.median(np.abs(window)) <= 15.0):
                for j in range(i - 2, i + 1):
                    if note[j] == "":
                        note[j] = "drift-up"

    out = df.copy()
    out["status"] = status
    out["note"] = note
    return out


def _severity_from_t_abs(t_abs, mild_thr, warn_thr, red_thr):
    if t_abs >= red_thr:
        return "RED"
    if t_abs >= warn_thr:
        return "WARNING"
    if t_abs >= mild_thr:
        return "MILD"
    return "GREEN"


def _label_thresholded_v2(df, internals, mode: str):
    """
    SpO2/RR/HR v2:
      - NOTE = GREEN/MILD/WARNING/RED or INFO (when V-shape cancels quickly)
      - STATUS = BASE vs PATTERN (pure dynamics)
      - V-shape override: if severe step but rebounds immediately -> INFO
    """
    t = internals["t"]
    vt = internals["vt"]
    ve = internals["ve"]
    n = len(t)

    if mode == "spo2":
        # |T| thresholds (given by you): >=0.6 RED, >=0.3 WARNING
        mild_thr, warn_thr, red_thr = 0.15, 0.30, 0.60
        vshape_warn = 0.30
    elif mode == "rr":
        # RR: k=25(%Δ/min). Use generic thresholds
        mild_thr, warn_thr, red_thr = 0.20, 0.50, 1.00
        vshape_warn = 0.50
    else:  # hr
        # HR: k=15(%Δ). Use same severity bands
        mild_thr, warn_thr, red_thr = 0.20, 0.50, 1.00
        vshape_warn = 0.50

    status = []
    note = [""] * n

    for i in range(n):
        status.append(_status_base_or_pattern(t[i], vt[i], ve[i], base_eps=1e-9))

    # severity note per step
    for i in range(n):
        note[i] = _severity_from_t_abs(abs(t[i]), mild_thr, warn_thr, red_thr)

    # V-shape override (filter likely transient artifact / short rebound)
    # If step i is WARNING/RED (>= vshape_warn), and immediate opposite rebound cancels well -> INFO
    for i in range(1, n - 1):
        if _detect_vshape_generic(t, i, warn_thr=vshape_warn):
            note[i] = "INFO"
            note[i + 1] = "INFO"

    out = df.copy()
    out["status"] = status
    out["note"] = note
    return out


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="DN v2 Demo", layout="wide")
st.title("DN v2 Demo")

tabs = st.tabs(["hrv", "spo2", "rr", "hr"])


def render_tab(tab, label, default_text, mode):
    with tab:
        st.subheader(label)
        txt = st.text_input("", value=default_text, key=f"inp_{mode}")
        if st.button("Compute", key=f"btn_{mode}"):
            try:
                values = _parse_10_numbers(txt)
                df, internals = _compute_core_from_series(values, mode=mode)

                # label status/note (v2)
                if mode == "hrv":
                    out = _label_hrv_v2(df, internals)
                else:
                    out = _label_thresholded_v2(df, internals, mode=mode)

                # show only SSRN-ready columns
                out_show = out[["minute", "value", "t", "e", "vt", "ve", "status", "note"]].copy()

                # formatting for nicer paper table
                for c in ["t", "e", "vt", "ve"]:
                    out_show[c] = out_show[c].astype(float).round(4)

                st.dataframe(out_show, use_container_width=True)

                csv = out_show.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"dn_v2_{mode}_10points.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(str(e))


# Defaults (you can replace anytime)
render_tab(
    tabs[0],
    "HRV (10 points)",
    "48 47 46 45 28 27 26 26 25 25",
    "hrv",
)
render_tab(
    tabs[1],
    "SpO2 (10 points)",
    "98 97 96 95 94 93 92 91 90 89",
    "spo2",
)
render_tab(
    tabs[2],
    "RR (10 points)",
    "16 16 17 18 43 17 16 15 16 18",
    "rr",
)
render_tab(
    tabs[3],
    "HR (10 points)",
    "75 76 110 77 78 82 88 95 103 112",
    "hr",
)
