import streamlit as st
import pandas as pd
import altair as alt

# =========================
# PAGE
# =========================
st.set_page_config(page_title="HRV DN Sentinel Demo", layout="wide")

# =========================
# DESIGN CONSTANTS
# =========================
K = 80.0  # unified constant: TT = %HRV / 80, and noise threshold = 80%

C_GREEN = "#00C853"
C_YELLOW = "#FFD600"
C_RED = "#D50000"
C_INFO = "#90A4AE"
C_BAR_BG = "#E0E0E0"
C_CARD_BG = "#11111108"

DN_GREEN = 0.95
DN_RED = 0.85

# =========================
# HELPERS
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def compute_state(pct_hrv: float, delta_ms: float, dn_core: float):
    """
    State logic:
    - Rise side: GREEN unless spike/noise (INFO). Noise uses K=80% (unified).
    - Drop side: uses DN thresholds (0.95, 0.85).
    """
    if pct_hrv >= K:
        return "INFO", "Possible spike / sensor noise", C_INFO
    if pct_hrv > 0:
        return "GREEN", "Recovery / rebound", C_GREEN
    if pct_hrv < 0:
        if dn_core < DN_RED:
            return "RED", "Reserve collapsing – trigger recommended", C_RED
        if dn_core < DN_GREEN:
            return "YELLOW", "Load increasing", C_YELLOW
        return "GREEN", "Stable", C_GREEN
    return "GREEN", "Stable", C_GREEN

# =========================
# TITLE
# =========================
st.title("HRV Sentinel demo")

# =========================
# INPUTS (TOP, 2 COLUMNS)
# =========================
in1, in2 = st.columns([1, 1], gap="large")
with in1:
    hrv_prev = st.number_input("HRV (t-1) ms", value=20.0, step=1.0)
with in2:
    hrv_curr = st.number_input("HRV (t) ms", value=22.0, step=1.0)

do_calc = st.button("CALCULATE", type="primary")

if "res" not in st.session_state:
    st.session_state["res"] = None

if do_calc:
    # =========================
    # CORE COMPUTE
    # =========================
    delta_ms = hrv_curr - hrv_prev
    pct_hrv = 0.0 if hrv_prev == 0 else 100.0 * delta_ms / hrv_prev  # %HRV

    # TT signed, unified constant
    TT_signed = pct_hrv / K
    TT_abs = abs(TT_signed)

    # DN core (drop-side), clamp for display
    DN_core = 1.0 - (TT_abs ** 2)
    DN_core = clamp(DN_core, 0.0, 1.0)

    # State
    state, msg, s_color = compute_state(pct_hrv, delta_ms, DN_core)

    # =========================
    # DN SENTINEL (0–2) - ONE NUMBER (matches bar)
    # =========================
    # Neutral: 1.0 when %HRV == 0 or INFO
    # Rise: 1 + TT_pos where TT_pos = clamp(TT_signed, 0..1)
    # Drop: DN_core (0..1)
    if state == "INFO" or pct_hrv == 0:
        DN_sentinel = 1.0
    elif pct_hrv > 0:
        TT_pos = clamp(TT_signed, 0.0, 1.0)
        DN_sentinel = 1.0 + TT_pos
    else:
        DN_sentinel = DN_core

    st.session_state["res"] = {
        "hrv_prev": hrv_prev,
        "hrv_curr": hrv_curr,
        "delta_ms": delta_ms,
        "pct_hrv": pct_hrv,
        "TT_signed": TT_signed,
        "DN_core": DN_core,
        "DN_sentinel": DN_sentinel,
        "state": state,
        "msg": msg,
        "s_color": s_color
    }

res = st.session_state["res"]
if res is None:
    st.info("Nhập 2 giá trị HRV rồi bấm **CALCULATE**.")
    st.stop()

# unpack
hrv_prev = res["hrv_prev"]
hrv_curr = res["hrv_curr"]
delta_ms = res["delta_ms"]
pct_hrv = res["pct_hrv"]
DN_sentinel = res["DN_sentinel"]
DN_core = res["DN_core"]
state = res["state"]
msg = res["msg"]
s_color = res["s_color"]

# =========================
# MAIN LAYOUT: 2 COLUMNS (LEFT: cards, RIGHT: charts)
# =========================
left, right = st.columns([1, 1], gap="large")

with left:
    # --- cards row (3)
    a, b, c = st.columns([1, 1, 1], gap="medium")

    with a:
        st.markdown(
            f"""
            <div style="padding:14px;border-radius:12px;background:{C_CARD_BG};">
              <div style="font-size:12px;opacity:0.7;">%HRV</div>
              <div style="font-size:34px;font-weight:800;">{pct_hrv:+.1f}%</div>
              <div style="font-size:12px;opacity:0.65;">ΔHRV = {delta_ms:+.1f} ms</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b:
        st.markdown(
            f"""
            <div style="padding:14px;border-radius:12px;background:{C_CARD_BG};">
              <div style="font-size:12px;opacity:0.7;">DN Sentinel (0–2)</div>
              <div style="font-size:34px;font-weight:800;">{DN_sentinel:.3f}</div>
              <div style="font-size:12px;opacity:0.65;">(same value as the bar)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c:
        st.markdown(
            f"""
            <div style="padding:14px;border-radius:12px;background:{s_color};color:#111;">
              <div style="font-size:12px;font-weight:900;letter-spacing:0.5px;">STATE</div>
              <div style="font-size:34px;font-weight:900;">{state}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # message
    if state == "RED":
        st.error(msg)
    elif state == "YELLOW":
        st.warning(msg)
    elif state == "INFO":
        st.info(msg)
    else:
        st.success(msg)

    st.caption("Single HRV signal · Time-dynamic processing · No ML · No absolute HRV threshold shown.")

with right:
    # =========================
    # 1) HRV RAW (FIXED ORDER)
    # =========================
    st.subheader("1) HRV raw")
    df_raw = pd.DataFrame({"Time": ["t-1", "t"], "HRV": [hrv_prev, hrv_curr]})

    chart_raw = (
        alt.Chart(df_raw)
        .mark_line(point=True)
        .encode(
            x=alt.X("Time:N", sort=["t-1", "t"], title="Time"),
            y=alt.Y("HRV:Q", title="HRV (ms)", scale=alt.Scale(zero=False))
        )
        .properties(height=210)
    )
    st.altair_chart(chart_raw, use_container_width=True)

    # =========================
    # 2) %HRV diverging bar
    # =========================
    st.subheader("2) %HRV (= %TT linear velocity)")
    df_pct = pd.DataFrame({"label": ["%HRV"], "value": [pct_hrv]})

    # color: rise=GREEN, drop=RED, info=INFO
    if state == "INFO":
        v_color = C_INFO
    else:
        v_color = C_GREEN if pct_hrv >= 0 else C_RED

    bar = alt.Chart(df_pct).mark_bar(color=v_color, cornerRadius=6).encode(
        y=alt.Y("label:N", title=""),
        x=alt.X("value:Q", title="% change",
                scale=alt.Scale(domain=[-100, 100]),
                axis=alt.Axis(format=".0f"))
    )
    zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#333", strokeWidth=2).encode(x="x:Q")
    txt = alt.Chart(df_pct).mark_text(
        align="left", dx=8, color="#111", fontSize=16, fontWeight="bold"
    ).encode(
        y="label:N", x="value:Q", text=alt.Text("value:Q", format="+.1f")
    )
    st.altair_chart((bar + zero_line + txt).properties(height=110), use_container_width=True)

    # =========================
    # 3) DN Sentinel bar (0–2) + labels + thresholds
    # =========================
    st.subheader("3) DN Sentinel (0–2)")

    # color by STATE
    if state == "INFO":
        g_color = C_INFO
    elif state == "GREEN":
        g_color = C_GREEN
    elif state == "YELLOW":
        g_color = C_YELLOW
    else:
        g_color = C_RED

    # background bar 0..2
    bg = alt.Chart(pd.DataFrame({"x0": [0], "x1": [2], "y": ["bar"]})).mark_bar(
        color=C_BAR_BG, cornerRadius=8
    ).encode(
        x=alt.X("x0:Q", scale=alt.Scale(domain=[0, 2]), title=""),
        x2="x1:Q",
        y=alt.Y("y:N", title="")
    )

    # foreground to DN_sentinel
    fg = alt.Chart(pd.DataFrame({"x0": [0], "x1": [DN_sentinel], "y": ["bar"]})).mark_bar(
        color=g_color, cornerRadius=8
    ).encode(x="x0:Q", x2="x1:Q", y="y:N")

    # midline at 1.0
    mid = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(color="#111", strokeWidth=2).encode(x="x:Q")

    # DN thresholds shown on left side for reference (0.85 & 0.95)
    t85 = alt.Chart(pd.DataFrame({"x": [DN_RED]})).mark_rule(
        color="#444", strokeDash=[5, 5], strokeWidth=2
    ).encode(x="x:Q")
    t95 = alt.Chart(pd.DataFrame({"x": [DN_GREEN]})).mark_rule(
        color="#444", strokeDash=[5, 5], strokeWidth=2
    ).encode(x="x:Q")

    # value label
    val = alt.Chart(pd.DataFrame({"x": [DN_sentinel], "y": ["bar"], "t": [f"{DN_sentinel:.3f}"]})).mark_text(
        align="left", dx=8, color="#111", fontSize=16, fontWeight="bold"
    ).encode(x="x:Q", y="y:N", text="t:N")

    # labels: DROP | NEUTRAL | RECOVERY
    labels_df = pd.DataFrame({
        "x": [0.15, 1.0, 1.85],
        "y": ["bar", "bar", "bar"],
        "t": ["DROP", "NEUTRAL", "RECOVERY"]
    })
    lbl = alt.Chart(labels_df).mark_text(
        dy=-22, color="#333", fontSize=11, fontWeight="bold"
    ).encode(x="x:Q", y="y:N", text="t:N")

    chart_dn = alt.layer(bg, fg, mid, t85, t95, val, lbl).properties(height=120)
    st.altair_chart(chart_dn, use_container_width=True)

    st.caption("0–1: reserve contraction (DN core) · 1.0: neutral · 1–2: recovery (1 + TT⁺), TT = %HRV/80.")
