import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="HRV Sentinel Demo", layout="wide")
st.title("HRV Sentinel demo")

# ==== Strong colors ====
C_GREEN = "#00C853"   # vivid green
C_YELLOW = "#FFD600"  # vivid yellow
C_RED = "#D50000"     # vivid red
C_GRAY = "#90A4AE"    # info / noise
C_BG = "#11111108"
C_BAR_BG = "#E0E0E0"

# ==== Noise-brake thresholds (tweakable if you want) ====
NOISE_PCT = 70.0      # %HRV spike threshold
NOISE_MS = 60.0       # absolute jump threshold (ms)

# ==== Keep last results ====
if "result" not in st.session_state:
    st.session_state["result"] = None

# ==== 2-column layout ====
left, right = st.columns([1, 1], gap="large")

with left:
    # Inputs
    c1, c2 = st.columns(2)
    with c1:
        hrv_prev = st.number_input("HRV (t-1) ms", value=25.0, step=1.0)
    with c2:
        hrv_curr = st.number_input("HRV (t) ms", value=20.0, step=1.0)

    do_calc = st.button("CALCULATE", type="primary")

    if do_calc:
        # ===== Compute (UNCHANGED) =====
        pct_hrv = 0.0 if hrv_prev == 0 else (hrv_curr - hrv_prev) / hrv_prev * 100.0  # %HRV (= %TT)
        TT = pct_hrv / 80.0
        DN = 1.0 - (TT ** 2)
        DN = max(0.0, min(1.0, DN))  # clamp for display only

        delta_ms = hrv_curr - hrv_prev

        # ===== State logic (SIGN-AWARE) =====
        # Positive side: treat as recovery unless spike/noise
        if pct_hrv >= 0:
            if (pct_hrv >= NOISE_PCT) or (delta_ms >= NOISE_MS):
                state = "INFO"
                msg = "Possible spike / sensor noise"
                s_color = C_GRAY
            else:
                state = "GREEN"
                msg = "Recovery / rebound"
                s_color = C_GREEN
        else:
            # Negative side: KEEP DN THRESHOLDS EXACTLY THE SAME
            if DN < 0.85:
                state = "RED"
                msg = "Reserve collapsing – trigger recommended"
                s_color = C_RED
            elif DN < 0.95:
                state = "YELLOW"
                msg = "Load increasing"
                s_color = C_YELLOW
            else:
                state = "GREEN"
                msg = "Stable"
                s_color = C_GREEN

        st.session_state["result"] = {
            "pct_hrv": pct_hrv,
            "DN": DN,
            "state": state,
            "msg": msg,
            "s_color": s_color,
            "delta_ms": delta_ms
        }

    res = st.session_state["result"]
    if res is None:
        st.info("Nhập 2 giá trị HRV rồi bấm **CALCULATE**.")
        st.stop()

    pct_hrv = res["pct_hrv"]
    DN = res["DN"]
    state = res["state"]
    msg = res["msg"]
    s_color = res["s_color"]
    delta_ms = res["delta_ms"]

    # Big bold results
    st.markdown(
        f"""
        <div style="display:flex; gap:18px; align-items:flex-end; margin-top:10px;">
          <div style="flex:1; padding:14px; border-radius:12px; background:{C_BG};">
            <div style="font-size:12px; opacity:0.7;">%HRV</div>
            <div style="font-size:34px; font-weight:800;">{pct_hrv:+.1f}%</div>
            <div style="font-size:12px; opacity:0.65;">ΔHRV = {delta_ms:+.1f} ms</div>
          </div>
          <div style="flex:1; padding:14px; border-radius:12px; background:{C_BG};">
            <div style="font-size:12px; opacity:0.7;">DN</div>
            <div style="font-size:34px; font-weight:800;">{DN:.3f}</div>
          </div>
          <div style="flex:1; padding:14px; border-radius:12px; background:{s_color}; color:#111;">
            <div style="font-size:12px; font-weight:800; letter-spacing:0.5px;">STATE</div>
            <div style="font-size:34px; font-weight:900;">{state}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Alert box
    if state == "RED":
        st.error(msg)
    elif state == "YELLOW":
        st.warning(msg)
    elif state == "INFO":
        st.info(msg)
    else:
        st.success(msg)

    st.caption(
        "Single physiological signal (HRV). Time-dynamic processing. No ML. "
        "No absolute HRV threshold shown."
    )

with right:
    res = st.session_state["result"]
    pct_hrv = res["pct_hrv"]
    DN = res["DN"]
    state = res["state"]

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

    # 2) %HRV (=%TT) diverging bar centered at 0
    st.subheader("2) %HRV (=%TT linear velocity)")
    df_pct = pd.DataFrame({"label": ["%HRV"], "value": [pct_hrv]})

    if pct_hrv >= 0:
        v_color = C_GREEN if state != "INFO" else C_GRAY
    else:
        v_color = C_RED

    bar = (
        alt.Chart(df_pct)
        .mark_bar(color=v_color, cornerRadius=6)
        .encode(
            y=alt.Y("label:N", title=""),
            x=alt.X(
                "value:Q",
                title="% change",
                scale=alt.Scale(domain=[-100, 100]),
                axis=alt.Axis(format=".0f")
            )
        )
    )
    zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#333", strokeWidth=2).encode(x="x:Q")
    text = (
        alt.Chart(df_pct)
        .mark_text(align="left", dx=8, color="#111", fontSize=16, fontWeight="bold")
        .encode(y="label:N", x="value:Q", text=alt.Text("value:Q", format="+.1f"))
    )
    st.altair_chart((bar + zero_line + text).properties(height=110), use_container_width=True)

    # 3) DN gauge-like bar (0→1)
    st.subheader("3) DN (state)")

    # Color DN bar by STATE (so INFO shows gray, recovery shows green)
    if state == "INFO":
        dn_color = C_GRAY
    elif state == "GREEN":
        dn_color = C_GREEN
    elif state == "YELLOW":
        dn_color = C_YELLOW
    else:
        dn_color = C_RED

    bg = alt.Chart(pd.DataFrame({"x0":[0], "x1":[1], "label":["DN"]})).mark_bar(color=C_BAR_BG, cornerRadius=8).encode(
        y=alt.Y("label:N", title=""),
        x=alt.X("x0:Q", scale=alt.Scale(domain=[0,1]), title=""),
        x2="x1:Q"
    )
    fg = alt.Chart(pd.DataFrame({"x0":[0], "x1":[DN], "label":["DN"]})).mark_bar(color=dn_color, cornerRadius=8).encode(
        y="label:N",
        x="x0:Q",
        x2="x1:Q"
    )

    # Keep the reference markers (unchanged)
    t85 = alt.Chart(pd.DataFrame({"x":[0.85]})).mark_rule(color="#444", strokeDash=[5,5], strokeWidth=2).encode(x="x:Q")
    t95 = alt.Chart(pd.DataFrame({"x":[0.95]})).mark_rule(color="#444", strokeDash=[5,5], strokeWidth=2).encode(x="x:Q")

    dn_text = alt.Chart(pd.DataFrame({"x":[DN], "label":["DN"], "txt":[f"{DN:.3f}"]})).mark_text(
        align="left", dx=8, color="#111", fontSize=16, fontWeight="bold"
    ).encode(y="label:N", x="x:Q", text="txt:N")

    st.altair_chart((bg + fg + t85 + t95 + dn_text).properties(height=110), use_container_width=True)
