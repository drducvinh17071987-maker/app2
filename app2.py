import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="HRV Sentinel Demo", layout="wide")
st.title("HRV Sentinel demo")

# ===== Colors (bold) =====
C_GREEN = "#00C853"
C_YELLOW = "#FFD600"
C_RED = "#D50000"
C_INFO = "#90A4AE"
C_BG = "#11111108"
C_BAR_BG = "#E0E0E0"

# ===== DN thresholds (DROP side only) =====
DN_GREEN = 0.95
DN_RED = 0.85

# ===== Noise-brake (positive spikes) =====
NOISE_PCT = 70.0
NOISE_MS = 60.0

if "result" not in st.session_state:
    st.session_state["result"] = None

left, right = st.columns([1, 1], gap="large")

with left:
    c1, c2 = st.columns(2)
    with c1:
        hrv_prev = st.number_input("HRV (t-1) ms", value=20.0, step=1.0)
    with c2:
        hrv_curr = st.number_input("HRV (t) ms", value=30.0, step=1.0)

    do_calc = st.button("CALCULATE", type="primary")

    if do_calc:
        # ===== Core compute (UNCHANGED) =====
        pct_hrv = 0.0 if hrv_prev == 0 else (hrv_curr - hrv_prev) / hrv_prev * 100.0  # %HRV (= %TT)
        TT = pct_hrv / 80.0
        DN = 1.0 - (TT ** 2)
        DN = max(0.0, min(1.0, DN))  # clamp display only
        delta_ms = hrv_curr - hrv_prev

        # ===== State logic (SIGN-AWARE) =====
        if pct_hrv >= 0:
            # Recovery unless spike/noise
            if (pct_hrv >= NOISE_PCT) or (delta_ms >= NOISE_MS):
                state, msg, s_color = "INFO", "Possible spike / sensor noise", C_INFO
            else:
                state, msg, s_color = "GREEN", "Recovery / rebound", C_GREEN
        else:
            # DROP side uses DN thresholds (KEEP)
            if DN < DN_RED:
                state, msg, s_color = "RED", "Reserve collapsing – trigger recommended", C_RED
            elif DN < DN_GREEN:
                state, msg, s_color = "YELLOW", "Load increasing", C_YELLOW
            else:
                state, msg, s_color = "GREEN", "Stable", C_GREEN

        # ===== REC for rise side (display only) =====
        if pct_hrv >= 0:
            if state == "INFO":
                rec = 0.0
            else:
                rec = min(1.0, max(0.0, pct_hrv / NOISE_PCT))  # 0..1
        else:
            rec = None

        st.session_state["result"] = dict(
            hrv_prev=hrv_prev,
            hrv_curr=hrv_curr,
            pct_hrv=pct_hrv,
            TT=TT,
            DN=DN,
            delta_ms=delta_ms,
            state=state,
            msg=msg,
            s_color=s_color,
            rec=rec
        )

    res = st.session_state["result"]
    if res is None:
        st.info("Nhập 2 giá trị HRV rồi bấm **CALCULATE**.")
        st.stop()

    pct_hrv = res["pct_hrv"]
    DN = res["DN"]
    delta_ms = res["delta_ms"]
    state = res["state"]
    msg = res["msg"]
    s_color = res["s_color"]
    rec = res["rec"]

    # ===== Left panel metrics (keep simple) =====
    st.markdown(
        f"""
        <div style="display:flex; gap:18px; align-items:flex-end; margin-top:10px;">
          <div style="flex:1; padding:14px; border-radius:12px; background:{C_BG};">
            <div style="font-size:12px; opacity:0.7;">%HRV</div>
            <div style="font-size:34px; font-weight:800;">{pct_hrv:+.1f}%</div>
            <div style="font-size:12px; opacity:0.65;">ΔHRV = {delta_ms:+.1f} ms</div>
          </div>
          <div style="flex:1; padding:14px; border-radius:12px; background:{C_BG};">
            <div style="font-size:12px; opacity:0.7;">DN (core)</div>
            <div style="font-size:34px; font-weight:800;">{DN:.3f}</div>
            <div style="font-size:12px; opacity:0.65;">(core value)</div>
          </div>
          <div style="flex:1; padding:14px; border-radius:12px; background:{s_color}; color:#111;">
            <div style="font-size:12px; font-weight:800; letter-spacing:0.5px;">STATE</div>
            <div style="font-size:34px; font-weight:900;">{state}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if state == "RED":
        st.error(msg)
    elif state == "YELLOW":
        st.warning(msg)
    elif state == "INFO":
        st.info(msg)
    else:
        st.success(msg)

    st.caption("Single HRV signal. Time-dynamic processing. No ML. No absolute HRV threshold shown.")

with right:
    # Read again for charts
    res = st.session_state["result"]
    hrv_prev = res["hrv_prev"]
    hrv_curr = res["hrv_curr"]
    pct_hrv = res["pct_hrv"]
    DN = res["DN"]
    state = res["state"]
    rec = res["rec"]

    # 1) HRV raw
    st.subheader("1) HRV raw")
    df_hrv = pd.DataFrame({"t": ["t-1", "t"], "HRV": [hrv_prev, hrv_curr]})
    chart_hrv = (
        alt.Chart(df_hrv)
        .mark_line(point=True)
        .encode(
            x=alt.X("t:N", title="Time"),
            y=alt.Y("HRV:Q", title="HRV (ms)", scale=alt.Scale(zero=False))
        )
        .properties(height=210)
    )
    st.altair_chart(chart_hrv, use_container_width=True)

    # 2) %HRV diverging bar
    st.subheader("2) %HRV (=%TT linear velocity)")
    df_pct = pd.DataFrame({"label": ["%HRV"], "value": [pct_hrv]})
    if pct_hrv >= 0:
        v_color = C_INFO if state == "INFO" else C_GREEN
    else:
        v_color = C_RED

    bar = alt.Chart(df_pct).mark_bar(color=v_color, cornerRadius=6).encode(
        y=alt.Y("label:N", title=""),
        x=alt.X("value:Q", title="% change",
                scale=alt.Scale(domain=[-100, 100]),
                axis=alt.Axis(format=".0f"))
    )
    zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#333", strokeWidth=2).encode(x="x:Q")
    text = alt.Chart(df_pct).mark_text(align="left", dx=8, color="#111", fontSize=16, fontWeight="bold").encode(
        y="label:N", x="value:Q", text=alt.Text("value:Q", format="+.1f")
    )
    st.altair_chart((bar + zero_line + text).properties(height=110), use_container_width=True)

    # 3) Unified sentinel bar (0..2)
    st.subheader("3) Sentinel bar (0–2)")

    # Map to one bar:
    # - DROP: use DN in [0,1]
    # - RISE: use 1 + REC in [1,2]
    if pct_hrv >= 0:
        rec_val = 0.0 if rec is None else rec
        gauge_value = 1.0 + rec_val
    else:
        gauge_value = DN

    # Color by STATE
    if state == "INFO":
        gauge_color = C_INFO
    elif state == "GREEN":
        gauge_color = C_GREEN
    elif state == "YELLOW":
        gauge_color = C_YELLOW
    else:
        gauge_color = C_RED

    bg = alt.Chart(pd.DataFrame({"x0":[0], "x1":[2], "label":["bar"]})).mark_bar(
        color=C_BAR_BG, cornerRadius=8
    ).encode(
        y=alt.Y("label:N", title=""),
        x=alt.X("x0:Q", scale=alt.Scale(domain=[0,2]), title=""),
        x2="x1:Q"
    )

    fg = alt.Chart(pd.DataFrame({"x0":[0], "x1":[gauge_value], "label":["bar"]})).mark_bar(
        color=gauge_color, cornerRadius=8
    ).encode(y="label:N", x="x0:Q", x2="x1:Q")

    # Midline at 1.0
    mid = alt.Chart(pd.DataFrame({"x":[1.0]})).mark_rule(color="#111", strokeWidth=2).encode(x="x:Q")

    # DN thresholds shown on left side only (still useful reference)
    t85 = alt.Chart(pd.DataFrame({"x":[DN_RED]})).mark_rule(color="#444", strokeDash=[5,5], strokeWidth=2).encode(x="x:Q")
    t95 = alt.Chart(pd.DataFrame({"x":[DN_GREEN]})).mark_rule(color="#444", strokeDash=[5,5], strokeWidth=2).encode(x="x:Q")

    txt = alt.Chart(pd.DataFrame({"x":[gauge_value], "label":["bar"], "t":[f"{gauge_value:.3f}"]})).mark_text(
        align="left", dx=8, color="#111", fontSize=16, fontWeight="bold"
    ).encode(y="label:N", x="x:Q", text="t:N")

    st.altair_chart((bg + fg + mid + t85 + t95 + txt).properties(height=110), use_container_width=True)

    st.caption("0–1: drop reserve (DN). 1–2: recovery confidence (REC). Center=1.0 neutral.")
