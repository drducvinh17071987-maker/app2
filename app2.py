import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DN v2 Demo", layout="wide")


# -----------------------------
# Parsing
# -----------------------------
def parse_10_numbers(text: str):
    parts = [p for p in text.replace(",", " ").split() if p.strip() != ""]
    vals = [float(p) for p in parts]
    if len(vals) != 10:
        raise ValueError("Need exactly 10 numbers.")
    return vals


# -----------------------------
# Core math (hidden from UI text)
# -----------------------------
def compute_series(values, mode: str):
    """
    Returns dataframe columns:
    minute, value, t, e, vt, ve, status(BASE/PATTERN), note
    """

    v = np.array(values, dtype=float)
    n = len(v)

    # Step signal per mode
    if mode == "spo2":
        # absolute delta for SpO2 (% points)
        step = np.zeros(n, dtype=float)
        step[1:] = v[1:] - v[:-1]
        # Normalize with K=5 (per your image)
        t = np.zeros(n, dtype=float)
        t[1:] = step[1:] / 5.0
    else:
        # percent delta for HRV/RR/HR
        step = np.zeros(n, dtype=float)
        step[1:] = 100.0 * (v[1:] - v[:-1]) / np.where(v[:-1] == 0, np.nan, v[:-1])

        if mode == "hrv":
            k = 80.0
        elif mode == "rr":
            k = 25.0
        elif mode == "hr":
            k = 15.0
        else:
            k = 80.0

        t = np.zeros(n, dtype=float)
        t[1:] = step[1:] / k

    # Lorentz-like energy
    e = np.zeros(n, dtype=float)
    e[:] = 1.0 - (t ** 2)

    # Derivatives
    vt = np.zeros(n, dtype=float)
    ve = np.zeros(n, dtype=float)
    vt[1:] = t[1:] - t[:-1]
    ve[1:] = e[1:] - e[:-1]

    # Notes + status
    note = [""] * n
    status = ["BASE"] * n

    if mode == "hrv":
        # --- HRV: NO absolute threshold, only patterns ---
        # step-drop (your earlier core idea: pct step <= -40%)
        for i in range(1, n):
            if step[i] <= -40.0:
                note[i] = "step-drop"
                status[i] = "PATTERN"

        # noise-spike brake (v1.5 idea): fast jump (>=70% OR >=60 ms)
        for i in range(1, n):
            abs_ms = abs(v[i] - v[i - 1])
            if (step[i] >= 70.0) or (abs_ms >= 60.0):
                # override to noise note
                note[i] = "noise-spike"
                status[i] = "PATTERN"

        # V-shape recovery (core v1 logic adapted to 10 points)
        # condition on two consecutive step-% (d1 then d2)
        for i in range(2, n):
            d1 = step[i - 1]
            d2 = step[i]
            total = d1 + d2
            if (d1 <= -20.0) and (d2 >= 15.0) and (abs(total) <= 12.0):
                # mark the recovery point (i) as v-shape
                note[i] = "v-shape"
                status[i] = "PATTERN"

        # drift-down / drift-up (simple monotone trend flag; still "pattern", not ICU alert)
        # 4 consecutive negatives (or positives) beyond tiny noise
        for i in range(4,_toggle := 4, ):
            pass  # just to avoid lint in some editors

        for i in range(4, n):
            window = step[i - 3:i + 1]  # 4 steps (includes current)
            if np.all(window < -1.0):
                # don't overwrite a stronger note like step-drop/noise/v-shape
                if note[i] == "":
                    note[i] = "drift-down"
                    status[i] = "PATTERN"
            if np.all(window > 1.0):
                if note[i] == "":
                    note[i] = "drift-up"
                    status[i] = "PATTERN"

    else:
        # --- SpO2 / RR / HR: keep |T| thresholds like your image ---
        # SpO2: |T|>=0.6 RED; |T|>=0.3 WARNING; V-shape -> INFO
        # RR:   |T|>=1.0 RED; |T|>=0.5 WARNING; |T|>=0.2 mild (optional)
        # HR:   |T|>=1.0 RED; |T|>=0.6 WARNING; |T|>=0.3 mild (optional)

        if mode == "spo2":
            red_thr = 0.6
            warn_thr = 0.3

            for i in range(1, n):
                if abs(t[i]) >= red_thr:
                    note[i] = "RED"
                    status[i] = "PATTERN"
                elif abs(t[i]) >= warn_thr:
                    note[i] = "WARNING"
                    status[i] = "PATTERN"

            # V-shape (drop then immediate recovery) -> INFO (filter false alert)
            for i in range(2, n):
                if (t[i - 1] <= -red_thr) and (t[i] >= red_thr) and (abs(t[i - 1] + t[i]) <= 0.2):
                    note[i] = "INFO (v-shape)"
                    status[i] = "PATTERN"

        elif mode == "rr":
            red_thr = 1.0
            warn_thr = 0.5
            mild_thr = 0.2  # aligns with your "5–8% => |T| ~0.2–0.3" note

            for i in range(1, n):
                if abs(t[i]) >= red_thr:
                    note[i] = "RED"
                    status[i] = "PATTERN"
                elif abs(t[i]) >= warn_thr:
                    note[i] = "WARNING"
                    status[i] = "PATTERN"
                elif abs(t[i]) >= mild_thr:
                    note[i] = "MILD"
                    status[i] = "PATTERN"

            # Optional V-shape for RR as noise/position artifact filter
            for i in range(2, n):
                if (abs(t[i - 1]) >= warn_thr) and (abs(t[i]) >= warn_thr) and (abs(t[i - 1] + t[i]) <= 0.2):
                    note[i] = "INFO (v-shape)"
                    status[i] = "PATTERN"

        elif mode == "hr":
            red_thr = 1.0
            warn_thr = 0.6
            mild_thr = 0.3

            for i in range(1, n):
                if abs(t[i]) >= red_thr:
                    note[i] = "RED"
                    status[i] = "PATTERN"
                elif abs(t[i]) >= warn_thr:
                    note[i] = "WARNING"
                    status[i] = "PATTERN"
                elif abs(t[i]) >= mild_thr:
                    note[i] = "MILD"
                    status[i] = "PATTERN"

            for i in range(2, n):
                if (abs(t[i - 1]) >= warn_thr) and (abs(t[i]) >= warn_thr) and (abs(t[i - 1] + t[i]) <= 0.2):
                    note[i] = "INFO (v-shape)"
                    status[i] = "PATTERN"

    df = pd.DataFrame({
        "minute": np.arange(1, n + 1),
        "value": v,
        "t": np.round(t, 4),
        "e": np.round(e, 4),
        "vt": np.round(vt, 4),
        "ve": np.round(ve, 4),
        "status": status,
        "note": note
    })
    return df


def render_tab(tab_name: str, default_text: str, mode: str):
    text = st.text_input("", value=default_text, key=f"in_{mode}")
    col1, col2 = st.columns([1, 4])
    with col1:
        run = st.button("Compute", key=f"btn_{mode}")
    with col2:
        st.write("")  # keep clean

    if run:
        try:
            values = parse_10_numbers(text)
            df = compute_series(values, mode=mode)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"dn_v2_{mode}.csv",
                mime="text/csv",
                key=f"dl_{mode}"
            )
        except Exception as ex:
            st.error(str(ex))


tabs = st.tabs(["hrv", "spo2", "rr", "hr"])

with tabs[0]:
    st.subheader("HRV (10 points)")
    render_tab("HRV", "50 49 48 47 46 45 44 43 42 41", mode="hrv")

with tabs[1]:
    st.subheader("SpO2 (10 points)")
    render_tab("SpO2", "98 97 96 95 94 93 92 91 90 89", mode="spo2")

with tabs[2]:
    st.subheader("RR (10 points)")
    render_tab("RR", "16 16 17 18 20 22 21 20 19 18", mode="rr")

with tabs[3]:
    st.subheader("HR (10 points)")
    render_tab("HR", "75 76 110 77 78 82 88 95 103 112", mode="hr")
