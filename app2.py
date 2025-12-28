import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="DN v2 Demo (4 Systems, 10 points)", layout="wide")

# -------------------------
# Helpers
# -------------------------
def parse_series(text: str, n_expected: int = 10):
    """Parse space/comma/newline separated numbers. Returns list[float] length n_expected or raises ValueError."""
    if text is None:
        raise ValueError("Empty input.")
    cleaned = text.replace(",", " ").replace(";", " ").replace("\n", " ").strip()
    parts = [p for p in cleaned.split(" ") if p.strip() != ""]
    vals = []
    for p in parts:
        vals.append(float(p))
    if len(vals) != n_expected:
        raise ValueError(f"Need exactly {n_expected} numbers, got {len(vals)}.")
    return vals

def pct_change(x_prev, x_cur):
    if x_prev == 0:
        return np.nan
    return 100.0 * (x_cur - x_prev) / x_prev

def dn_table_from_pct(pct_series, k):
    """
    pct_series: list length N, where pct_series[i] is %Δ at minute i (minute 1 = 0 or NaN)
    T = pct/k
    E = 1 - T^2
    vT = T[i] - T[i-1]
    vE = E[i] - E[i-1]
    """
    T = np.array(pct_series, dtype=float) / float(k)
    E = 1.0 - (T ** 2)
    vT = np.full_like(T, np.nan, dtype=float)
    vE = np.full_like(E, np.nan, dtype=float)
    for i in range(1, len(T)):
        vT[i] = T[i] - T[i - 1]
        vE[i] = E[i] - E[i - 1]
    return T, E, vT, vE

def status_from_absT(absT):
    """Generic thresholds used for RR/HR: INFO>=0.2, WARNING>=0.5, RED>=1.0"""
    if np.isnan(absT):
        return "—"
    if absT >= 1.0:
        return "RED"
    if absT >= 0.5:
        return "WARNING"
    if absT >= 0.2:
        return "INFO"
    return "GREEN"

def status_from_absT_spo2(absT):
    """SpO2 thresholds: WARNING>=0.3, RED>=0.6"""
    if np.isnan(absT):
        return "—"
    if absT >= 0.6:
        return "RED"
    if absT >= 0.3:
        return "WARNING"
    return "GREEN"

def plot_with_status(minutes, values, statuses, title):
    # Map status -> marker symbol/outline. Avoid manual colors? User asked "xanh vàng đỏ rõ nét".
    # Plotly default colors aren't red/yellow/green, so we must set to be clear.
    # We'll set clear RGB colors.
    color_map = {
        "GREEN": "#2ecc71",
        "INFO": "#3498db",
        "WARNING": "#f1c40f",
        "RED": "#e74c3c",
        "—": "#95a5a6",
    }
    colors = [color_map.get(s, "#95a5a6") for s in statuses]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=minutes,
            y=values,
            mode="lines+markers",
            marker=dict(size=10, color=colors, line=dict(width=1, color="#2c3e50")),
            line=dict(width=2),
            hovertemplate="Minute %{x}<br>Value %{y}<br>Status %{text}<extra></extra>",
            text=statuses,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Minute",
        yaxis_title="Value",
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
    )
    return fig

# -------------------------
# Core computations per system
# -------------------------
def compute_hrv(series):
    """
    HRV DN-dynamic (no absolute threshold):
    pct[i] = %ΔHRV between i-1 and i
    T = pct/80
    E = 1 - T^2
    Basic notes:
      - step_drop_red: pct <= -40
      - V-shape: drop <= -20 then next >= +15 and |total over 2 steps| <= 12
      - noise_brake (optional idea): pct >= +70 (or >= +60ms jump) -> INFO possible noise
    """
    N = len(series)
    pct = [np.nan] * N
    pct[0] = 0.0
    for i in range(1, N):
        pct[i] = pct_change(series[i - 1], series[i])

    T, E, vT, vE = dn_table_from_pct(pct, k=80.0)

    note = [""] * N
    status = ["—"] * N
    status[0] = "—"

    for i in range(1, N):
        p = pct[i]
        if np.isnan(p):
            status[i] = "—"
            continue

        # Step-drop flag (strong)
        if p <= -40.0:
            status[i] = "RED"
            note[i] = "step-drop (≤ -40%)"
        elif p <= -20.0:
            status[i] = "WARNING"
            note[i] = "moderate drop"
        elif p >= 70.0:
            status[i] = "INFO"
            note[i] = "possible noise (fast spike)"
        else:
            status[i] = "GREEN"

    # V-shape recovery tag (overrides to INFO if pattern matches)
    for i in range(1, N - 1):
        d1 = pct[i]
        d2 = pct[i + 1]
        if np.isnan(d1) or np.isnan(d2):
            continue
        total = d1 + d2
        if (d1 <= -20.0) and (d2 >= 15.0) and (abs(total) <= 12.0):
            # mark both points for readability
            status[i] = "INFO"
            status[i + 1] = "INFO"
            note[i] = "V-shape (drop→recover)"
            note[i + 1] = "V-shape (recover)"

    df = pd.DataFrame({
        "Minute": list(range(1, N + 1)),
        "HRV": series,
        "%Δ (raw)": np.round(pct, 2),
        "T (= %Δ/80)": np.round(T, 4),
        "E (= 1-T²)": np.round(E, 4),
        "vT": np.round(vT, 4),
        "vE": np.round(vE, 4),
        "DN_status": status,
        "DN_note": note
    })
    return df, status

def compute_spo2(series):
    """
    SpO2: use step-delta per minute with k=5:
      delta = SpO2[i] - SpO2[i-1]
      T = delta/5
      E = 1 - T^2 (optional but we keep for consistent table)
      Status by |T|: RED>=0.6, WARNING>=0.3 else GREEN
      V-shape filter: (T<=-0.6 and next>=+0.6) -> INFO
    """
    N = len(series)
    delta = [np.nan] * N
    delta[0] = 0.0
    for i in range(1, N):
        delta[i] = series[i] - series[i - 1]

    T = np.array(delta, dtype=float) / 5.0
    E = 1.0 - (T ** 2)
    vT = np.full_like(T, np.nan, dtype=float)
    vE = np.full_like(E, np.nan, dtype=float)
    for i in range(1, N):
        vT[i] = T[i] - T[i - 1]
        vE[i] = E[i] - E[i - 1]

    status = ["—"] * N
    note = [""] * N
    status[0] = "—"
    for i in range(1, N):
        absT = abs(T[i])
        status[i] = status_from_absT_spo2(absT)
        if status[i] == "RED":
            note[i] = "|T|≥0.6"
        elif status[i] == "WARNING":
            note[i] = "|T|≥0.3"

    # V-shape (drop then immediate recover)
    for i in range(1, N - 1):
        if (T[i] <= -0.6) and (T[i + 1] >= 0.6):
            status[i] = "INFO"
            status[i + 1] = "INFO"
            note[i] = "V-shape (possible artifact)"
            note[i + 1] = "V-shape (recover)"

    df = pd.DataFrame({
        "Minute": list(range(1, N + 1)),
        "SpO2": series,
        "Δ (raw)": delta,
        "T (= Δ/5)": np.round(T, 4),
        "E (= 1-T²)": np.round(E, 4),
        "vT": np.round(vT, 4),
        "vE": np.round(vE, 4),
        "DN_status": status,
        "DN_note": note
    })
    return df, status

def compute_rr(series):
    """
    RR: %Δ per minute, k=25:
      T = %Δ/25 ; E = 1 - T^2
      Status by |T| (generic): INFO>=0.2, WARNING>=0.5, RED>=1.0
      V-shape optional tag: big rise then fall quickly -> INFO
    """
    N = len(series)
    pct = [np.nan] * N
    pct[0] = 0.0
    for i in range(1, N):
        pct[i] = pct_change(series[i - 1], series[i])

    T, E, vT, vE = dn_table_from_pct(pct, k=25.0)

    status = ["—"] * N
    note = [""] * N
    status[0] = "—"
    for i in range(1, N):
        status[i] = status_from_absT(abs(T[i]))
        if status[i] in ["INFO", "WARNING", "RED"]:
            note[i] = f"|T|={abs(T[i]):.2f}"

    # simple V-shape filter for RR: up then down strongly (often motion/cough/short artifact)
    for i in range(1, N - 1):
        if (T[i] >= 0.8) and (T[i + 1] <= -0.8):
            status[i] = "INFO"
            status[i + 1] = "INFO"
            note[i] = "V-shape (transient)"
            note[i + 1] = "V-shape (recover)"

    df = pd.DataFrame({
        "Minute": list(range(1, N + 1)),
        "RR": series,
        "%Δ (raw)": np.round(pct, 2),
        "T (= %Δ/25)": np.round(T, 4),
        "E (= 1-T²)": np.round(E, 4),
        "vT": np.round(vT, 4),
        "vE": np.round(vE, 4),
        "DN_status": status,
        "DN_note": note
    })
    return df, status

def compute_hr(series):
    """
    HR: %Δ per minute, k=15:
      T = %Δ/15 ; E = 1 - T^2
      Status by |T|: INFO>=0.2, WARNING>=0.5, RED>=1.0
    """
    N = len(series)
    pct = [np.nan] * N
    pct[0] = 0.0
    for i in range(1, N):
        pct[i] = pct_change(series[i - 1], series[i])

    T, E, vT, vE = dn_table_from_pct(pct, k=15.0)

    status = ["—"] * N
    note = [""] * N
    status[0] = "—"
    for i in range(1, N):
        status[i] = status_from_absT(abs(T[i]))
        if status[i] in ["INFO", "WARNING", "RED"]:
            note[i] = f"|T|={abs(T[i]):.2f}"

    # optional V-shape filter for HR: spike then back (motion artifact)
    for i in range(1, N - 1):
        if (T[i] >= 1.0) and (T[i + 1] <= -0.7):
            status[i] = "INFO"
            status[i + 1] = "INFO"
            note[i] = "V-shape (possible artifact)"
            note[i + 1] = "recover"

    df = pd.DataFrame({
        "Minute": list(range(1, N + 1)),
        "HR": series,
        "%Δ (raw)": np.round(pct, 2),
        "T (= %Δ/15)": np.round(T, 4),
        "E (= 1-T²)": np.round(E, 4),
        "vT": np.round(vT, 4),
        "vE": np.round(vE, 4),
        "DN_status": status,
        "DN_note": note
    })
    return df, status

# -------------------------
# UI
# -------------------------
st.title("DN v2 Demo – 4 Systems (10 points each)")
st.caption("Each tab: input 10 values (1-minute spacing) → click button → table (%Δ vs DN: T, E, vT, vE) + colored chart.")

left, right = st.columns([1, 1])
with left:
    st.markdown("**Input format**: 10 numbers separated by spaces. Example: `42 41 40 26 39 38 37 36 35 34`")
with right:
    st.markdown("**Constants**: HRV k=80 (dynamic), SpO₂ k=5 (Δ), RR k=25 (%Δ), HR k=15 (%Δ).")

tabs = st.tabs(["HRV (dynamic, no absolute threshold)", "SpO₂ (k=5, |T| thresholds)", "RR (k=25, |T| thresholds)", "HR (k=15, |T| thresholds)"])

# ---- HRV tab
with tabs[0]:
    c1, c2 = st.columns([1, 1])
    with c1:
        hrv_text = st.text_input("HRV (10 points)", value="42 41 40 26 39 38 37 36 35 34", key="hrv_in")
        run = st.button("Compute HRV", key="hrv_btn")
    with c2:
        st.markdown("**Notes**: DN-dynamic only (step-drop, V-shape, noise-spike). No absolute HRV threshold.")

    if run:
        try:
            series = parse_series(hrv_text, 10)
            df, status = compute_hrv(series)
            st.dataframe(df, use_container_width=True)
            fig = plot_with_status(df["Minute"], df["HRV"], df["DN_status"], "HRV – raw series with DN status (per transition into minute)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(str(e))

# ---- SpO2 tab
with tabs[1]:
    c1, c2 = st.columns([1, 1])
    with c1:
        spo2_text = st.text_input("SpO₂ (10 points)", value="98 97 96 95 93 92 93 92 91 92", key="spo2_in")
        run = st.button("Compute SpO₂", key="spo2_btn")
    with c2:
        st.markdown("**Thresholds**: |T|≥0.6 RED, |T|≥0.3 WARNING, V-shape → INFO.  T = Δ/5.")

    if run:
        try:
            series = parse_series(spo2_text, 10)
            df, status = compute_spo2(series)
            st.dataframe(df, use_container_width=True)
            fig = plot_with_status(df["Minute"], df["SpO2"], df["DN_status"], "SpO₂ – raw series with DN status (per transition into minute)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(str(e))

# ---- RR tab
with tabs[2]:
    c1, c2 = st.columns([1, 1])
    with c1:
        rr_text = st.text_input("RR (10 points)", value="16 16 17 17 18 18 19 19 21 21", key="rr_in")
        run = st.button("Compute RR", key="rr_btn")
    with c2:
        st.markdown("**Thresholds**: |T|≥1 RED, |T|≥0.5 WARNING, |T|≥0.2 INFO.  T = %Δ/25.")

    if run:
        try:
            series = parse_series(rr_text, 10)
            df, status = compute_rr(series)
            st.dataframe(df, use_container_width=True)
            fig = plot_with_status(df["Minute"], df["RR"], df["DN_status"], "RR – raw series with DN status (per transition into minute)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(str(e))

# ---- HR tab
with tabs[3]:
    c1, c2 = st.columns([1, 1])
    with c1:
        hr_text = st.text_input("HR (10 points)", value="82 83 84 118 86 87 88 89 90 91", key="hr_in")
        run = st.button("Compute HR", key="hr_btn")
    with c2:
        st.markdown("**Thresholds**: |T|≥1 RED, |T|≥0.5 WARNING, |T|≥0.2 INFO.  T = %Δ/15.")

    if run:
        try:
            series = parse_series(hr_text, 10)
            df, status = compute_hr(series)
            st.dataframe(df, use_container_width=True)
            fig = plot_with_status(df["Minute"], df["HR"], df["DN_status"], "HR – raw series with DN status (per transition into minute)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(str(e))

st.markdown("---")
st.markdown("Tip: For SSRN v2, use the **table output** as Appendix (10-point sequences) and keep main text on **transition tables (5 transitions)**.")
