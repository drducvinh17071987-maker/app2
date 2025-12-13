import streamlit as st
import pandas as pd

st.set_page_config(page_title="HRV Sentinel Demo", layout="centered")

st.title("HRV Sentinel (Single-signal, time-dynamic)")

# ===== Input =====
col1, col2 = st.columns(2)
with col1:
    hrv_prev = st.number_input("HRV (t-1) ms", value=50.0, step=1.0)
with col2:
    hrv_curr = st.number_input("HRV (t) ms", value=45.0, step=1.0)

# ===== Compute =====
# %HRV = %TT (linear velocity)
pct_tt = 0.0 if hrv_prev == 0 else (hrv_curr - hrv_prev) / hrv_prev * 100.0

# Normalize velocity (constant, not personalized)
TT = pct_tt / 80.0

# DN (internal kernel value)
DN = 1.0 - (TT ** 2)
DN = max(0.0, min(1.0, DN))  # clamp for display safety

# ===== Data for charts =====
df_hrv = pd.DataFrame(
    {"HRV (ms)": [hrv_prev, hrv_curr]},
    index=["t-1", "t"]
)

df_pct = pd.DataFrame(
    {"%TT (linear velocity)": [pct_tt]},
    index=["window"]
)

df_dn = pd.DataFrame(
    {"DN (reserve state)": [DN]},
    index=["window"]
)

# ===== Charts (Streamlit native, no matplotlib) =====
st.subheader("1) HRV raw")
st.line_chart(df_hrv)

st.subheader("2) %TT (linear velocity)")
st.bar_chart(df_pct)

st.subheader("3) DN (state)")
st.bar_chart(df_dn)

# ===== Output state (signal only) =====
st.subheader("State (signal)")
if DN < 0.85:
    st.error("Reserve collapsing – trigger recommended")
elif DN < 0.95:
    st.warning("Load increasing")
else:
    st.success("Stable")

st.caption(
    "Single physiological signal (HRV). Time-dynamic processing. "
    "No thresholds on HRV. No ML."
)
