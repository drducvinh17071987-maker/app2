import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

def compute_dn(series, k, drop_thr=0.3, recover_thr=0.2):
    x = np.array(series, dtype=float)
    n = len(x)

    pct = np.zeros(n)
    T = np.zeros(n)
    E = np.zeros(n)
    vT = np.zeros(n)
    vE = np.zeros(n)
    note = ["OK"] * n

    for i in range(1, n):
        pct[i] = 100 * (x[i] - x[i-1]) / x[i-1]
        T[i] = pct[i] / k
        E[i] = 1 - T[i]**2
        vT[i] = T[i] - T[i-1]
        vE[i] = E[i] - E[i-1]

    for i in range(1, n):
        # DROP
        if abs(T[i]) >= drop_thr:
            note[i] = "DROP"

        # V-SHAPE (lọc báo giả)
        if i >= 1:
            if abs(T[i-1]) >= drop_thr:
                if np.sign(T[i]) != np.sign(T[i-1]) and abs(T[i]) <= recover_thr:
                    note[i] = "V-SHAPE"

        # DRIFT
        if i >= 3:
            last = T[i-3:i+1]
            if np.all(np.sign(last) == np.sign(last[0])) and np.all(np.abs(last) < drop_thr):
                note[i] = "DRIFT"

    df = pd.DataFrame({
        "idx": range(1, n+1),
        "raw": x,
        "pct_step": pct,
        "T": T,
        "E": E,
        "vT": vT,
        "vE": vE,
        "note": note
    })
    return df


def tab_ui(title, k):
    st.subheader(title)
    raw = st.text_input("Nhập 10 giá trị thô (cách nhau bằng dấu cách):", "")
    if st.button("Compute", key=title):
        vals = [float(v) for v in raw.strip().split()]
        if len(vals) != 10:
            st.error("Cần đúng 10 giá trị")
            return
        df = compute_dn(vals, k)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name=f"{title.replace(' ', '_')}.csv"
        )


tab1, tab2, tab3, tab4 = st.tabs(["HRV", "HR", "SpO₂", "RR"])

with tab1:
    tab_ui("HRV (dynamic)", k=80)

with tab2:
    tab_ui("HR (dynamic)", k=15)

with tab3:
    tab_ui("SpO₂ (dynamic)", k=5)

with tab4:
    tab_ui("RR (dynamic)", k=25)
