import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="HRV Sentinel Demo", layout="centered")

st.title("HRV Sentinel Demo")

# -------------------------
# Inputs
# -------------------------
col1, col2 = st.columns(2)
with col1:
    hrv_prev = st.number_input("HRV (t-1) ms", value=50.0, step=1.0)
with col2:
    hrv_curr = st.number_input("HRV (t) ms", value=45.0, step=1.0)

# Button
do_calc = st.button("CALCULATE", type="primary")

# Keep last results
if "result" not in st.session_state:
    st.session_state["result"] = None

if do_calc:
    # -------- compute --------
    pct_hrv = 0.0 if hrv_prev == 0 else (hrv_curr - hrv_prev) / hrv_prev * 100.0  # %HRV (= %TT)
    TT = pct_hrv / 80.0
    DN = 1.0 - (TT ** 2)
    DN = max(0.0, min(1.0, DN))  # clamp for display

    # alert (simple, you can tune thresholds)
    if DN < 0.85:
        state = "RED"
        msg = "Reserve collapsing – trigger recommended"
    elif DN < 0.95:
        state = "YELLOW"
        msg = "Load increasing"
    else:
        state = "GREEN"
        msg = "Stable"

    st.session_state["result"] = {
        "pct_hrv": pct_hrv,
        "DN": DN,
        "state": state,
        "msg": msg
    }

# -------------------------
# Render results (only after button)
# -------------------------
res = st.session_state["result"]
if res is None:
    st.info("Nhập 2 giá trị HRV rồi bấm **CALCULATE**.")
    st.stop()

pct_hrv = res["pct_hrv"]
DN = res["DN"]
state = res["state"]
msg = res["msg"]

# Metrics row
m1, m2, m3 = st.columns(3)
m1.metric("%HRV", f"{pct_hrv:+.1f}%")
m2.metric("DN", f"{DN:.3f}")
m3.metric("STATE", state)

# Alert box
if state == "RED":
    st.error(msg)
elif state == "YELLOW":
    st.warning(msg)
else:
    st.success(msg)

st.divider()

# -------------------------
# 1) HRV raw chart (2 points, easy to read)
# -------------------------
st.subheader("1) HRV raw")
df_hrv = pd.DataFrame({
    "t": ["t-1", "t"],
    "HRV": [hrv_prev, hrv_curr]
})
chart_hrv = (
    alt.Chart(df_hrv)
    .mark_line(point=True)
    .encode(
        x=alt.X("t:N", title="Time"),
        y=alt.Y("HRV:Q", title="HRV (ms)", scale=alt.Scale(zero=False))
    )
    .properties(height=220)
)
st.altair_chart(chart_hrv, use_container_width=True)

# -------------------------
# 2) %HRV (=%TT) diverging bar centered at 0
# -------------------------
st.subheader("2) %HRV (=%TT linear velocity)")
df_pct = pd.DataFrame({"label": ["%HRV"], "value": [pct_hrv]})

color = "#2ecc71" if pct_hrv >= 0 else "#e74c3c"  # green / red

bar = (
    alt.Chart(df_pct)
    .mark_bar(color=color)
    .encode(
        y=alt.Y("label:N", title=""),
        x=alt.X(
            "value:Q",
            title="% change",
            scale=alt.Scale(domain=[-100, 100]),  # fixed so it never looks “giant”
            axis=alt.Axis(format=".0f")
        )
    )
)

zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#666").encode(x="x:Q")

text = (
    alt.Chart(df_pct)
    .mark_text(align="left", dx=6, color="#111")
    .encode(
        y=alt.Y("label:N"),
        x=alt.X("value:Q"),
        text=alt.Text("value:Q", format="+.1f")
    )
)

chart_pct = (bar + zero_line + text).properties(height=120)
st.altair_chart(chart_pct, use_container_width=True)

# -------------------------
# 3) DN gauge-like bar (0 → 1)
# -------------------------
st.subheader("3) DN (state)")
df_dn = pd.DataFrame({"DN": [DN]})

# background bar 0..1
bg = alt.Chart(pd.DataFrame({"x0": [0], "x1": [1], "label": ["DN"]})).mark_bar(color="#eaeaea").encode(
    y=alt.Y("label:N", title=""),
    x=alt.X("x0:Q", scale=alt.Scale(domain=[0, 1]), title=""),
    x2="x1:Q"
)

fg_color = "#2ecc71" if DN >= 0.95 else ("#f1c40f" if DN >= 0.85 else "#e74c3c")
fg = alt.Chart(pd.DataFrame({"x0": [0], "x1": [DN], "label": ["DN"]})).mark_bar(color=fg_color).encode(
    y="label:N",
    x="x0:Q",
    x2="x1:Q"
)

t85 = alt.Chart(pd.DataFrame({"x": [0.85]})).mark_rule(color="#999", strokeDash=[4,4]).encode(x="x:Q")
t95 = alt.Chart(pd.DataFrame({"x": [0.95]})).mark_rule(color="#999", strokeDash=[4,4]).encode(x="x:Q")

dn_text = alt.Chart(pd.DataFrame({"x": [DN], "label": ["DN"], "txt": [f"{DN:.3f}"]})).mark_text(
    align="left", dx=6, color="#111"
).encode(
    y="label:N",
    x="x:Q",
    text="txt:N"
)

chart_dn = (bg + fg + t85 + t95 + dn_text).properties(height=120)
st.altair_chart(chart_dn, use_container_width=True)

st.caption("Single physiological signal (HRV). Time-dynamic processing. No ML. No absolute HRV threshold shown.")
