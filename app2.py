import streamlit as st
import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------
def parse_10_numbers(text: str) -> list[float]:
    parts = [p for p in text.replace(",", " ").split() if p.strip() != ""]
    vals = [float(p) for p in parts]
    if len(vals) != 10:
        raise ValueError("Need exactly 10 numbers.")
    return vals


def lorentz_from_t(t: np.ndarray) -> np.ndarray:
    # core math, do NOT display formula in UI
    return 1.0 - (t ** 2)


def compute_table(values: list[float], mode: str) -> pd.DataFrame:
    """
    mode in {"HRV","SPO2","RR","HR"}.
    Returns dataframe with columns:
    minute, value, t, e, vT, vE, status, note
    """
    v = np.array(values, dtype=float)
    n = len(v)

    # minute index 1..10
    minute = np.arange(1, n + 1, dtype=int)

    # step change
    if mode == "SPO2":
        # absolute delta in percentage-points
        delta = np.zeros(n, dtype=float)
        delta[1:] = v[1:] - v[:-1]
        # normalize with k=5 (hidden)
        t = delta / 5.0
    elif mode == "HRV":
        pct = np.zeros(n, dtype=float)
        pct[1:] = 100.0 * (v[1:] - v[:-1]) / np.where(v[:-1] == 0, np.nan, v[:-1])
        pct = np.nan_to_num(pct, nan=0.0, posinf=0.0, neginf=0.0)
        # normalize with K=80 (hidden)
        t = pct / 80.0
    elif mode == "RR":
        pct = np.zeros(n, dtype=float)
        pct[1:] = 100.0 * (v[1:] - v[:-1]) / np.where(v[:-1] == 0, np.nan, v[:-1])
        pct = np.nan_to_num(pct, nan=0.0, posinf=0.0, neginf=0.0)
        # normalize with K=25 (hidden)
        t = pct / 25.0
    elif mode == "HR":
        pct = np.zeros(n, dtype=float)
        pct[1:] = 100.0 * (v[1:] - v[:-1]) / np.where(v[:-1] == 0, np.nan, v[:-1])
        pct = np.nan_to_num(pct, nan=0.0, posinf=0.0, neginf=0.0)
        # normalize with K=15 (hidden)
        t = pct / 15.0
    else:
        raise ValueError("Unknown mode.")

    e = lorentz_from_t(t)

    vT = np.zeros(n, dtype=float)
    vE = np.zeros(n, dtype=float)
    vT[1:] = t[1:] - t[:-1]
    vE[1:] = e[1:] - e[:-1]

    status = np.array(["GREEN"] * n, dtype=object)
    note = np.array([""] * n, dtype=object)

    # -------------------------
    # Status logic (paper-clean)
    # -------------------------
    if mode in {"SPO2", "RR", "HR"}:
        # threshold on |t| (hidden), applied to dynamic only
        abs_t = np.abs(t)

        status[abs_t >= 0.6] = "RED"
        status[(abs_t >= 0.3) & (abs_t < 0.6)] = "WARNING"

        # add short reason only when not GREEN
        for i in range(n):
            if status[i] == "RED":
                note[i] = "threshold-hit"
            elif status[i] == "WARNING":
                note[i] = "threshold-hit"

        # V-shape filter (drop then immediate recovery)
        # If (i) is a drop (t negative strong) and (i+1) rebounds positive strong -> mark both as INFO
        for i in range(1, n - 1):
            if (t[i] <= -0.3) and (t[i + 1] >= 0.3):
                status[i] = "INFO"
                status[i + 1] = "INFO"
                note[i] = "V-shape"
                note[i + 1] = "V-shape"

    elif mode == "HRV":
        # HRV: no absolute HRV threshold; only dynamic patterns

        # Step-drop rule (dynamic, not absolute): pct <= -40% -> RED
        # (This is still pattern-based, not baseline-based.)
        pct = np.zeros(n, dtype=float)
        pct[1:] = 100.0 * (v[1:] - v[:-1]) / np.where(v[:-1] == 0, np.nan, v[:-1])
        pct = np.nan_to_num(pct, nan=0.0, posinf=0.0, neginf=0.0)

        for i in range(1, n):
            if pct[i] <= -40.0:
                status[i] = "RED"
                note[i] = "step-drop"

        # Noise-brake: very fast positive spike -> INFO (possible sensor noise)
        # Fix: everything scalar/array-safe
        delta_ms = np.zeros(n, dtype=float)
        delta_ms[1:] = v[1:] - v[:-1]

        noise_mask = (pct >= 70.0) | (np.abs(delta_ms) >= 60.0)
        # ignore minute 1
        noise_mask[0] = False

        for i in range(n):
            if noise_mask[i]:
                status[i] = "INFO"
                note[i] = "noise-spike"

        # V-shape recovery (dynamic): drop then rebound, downgrade to INFO if it looks like transient artifact/recovery
        # Criteria: pct[i] <= -20 and pct[i+1] >= +15 and |(v[i+1]-v[i-1])/v[i-1]| <= 12%
        for i in range(1, n - 1):
            total_pct = 0.0
            if v[i - 1] != 0:
                total_pct = 100.0 * (v[i + 1] - v[i - 1]) / v[i - 1]
            if (pct[i] <= -20.0) and (pct[i + 1] >= 15.0) and (abs(total_pct) <= 12.0):
                status[i] = "INFO"
                status[i + 1] = "INFO"
                note[i] = "V-shape"
                note[i + 1] = "V-shape"

        # Drift-down hint (paper-friendly): small persistent negatives -> keep GREEN but note "drift"
        # Only annotate, do not escalate.
        for i in range(2, n):
            if (pct[i] < 0) and (pct[i - 1] < 0) and status[i] == "GREEN":
                note[i] = "drift-down"

        # Minute 1 is always baseline row
        status[0] = "GREEN"
        if note[0] == "":
            note[0] = ""

    df = pd.DataFrame(
        {
            "minute": minute,
            "value": v,
            "t": np.round(t, 4),
            "e": np.round(e, 4),
            "vT": np.round(vT, 4),
            "vE": np.round(vE, 4),
            "status": status,
            "note": note,
        }
    )
    return df


def render_tab(tab_name: str, default_text: str, mode: str):
    st.subheader(f"{tab_name} (10 points)")
    txt = st.text_input("", value=default_text, key=f"inp_{mode}")
    if st.button("Compute", key=f"btn_{mode}"):
        try:
            values = parse_10_numbers(txt)
            df = compute_table(values, mode=mode)

            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"DN_v2_{mode}.csv",
                mime="text/csv",
                key=f"dl_{mode}",
            )
        except Exception as ex:
            st.error(str(ex))


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="DN v2 Demo", layout="wide")

tabs = st.tabs(["hrv", "spo2", "rr", "hr"])

with tabs[0]:
    render_tab("HRV", "50 49 48 47 46 45 44 43 42 41", "HRV")

with tabs[1]:
    render_tab("SpO2", "98 97 96 95 94 93 92 91 90 89", "SPO2")

with tabs[2]:
    render_tab("RR", "16 17 18 19 21 23 25 28 31 34", "RR")

with tabs[3]:
    render_tab("HR", "75 76 110 77 78 82 88 95 103 112", "HR")
