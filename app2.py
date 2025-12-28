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

    # NOTE logic: DROP / V-SHAPE / DRIFT / OK
    for i in range(1, n):
        # DROP
        if abs(T[i]) >= drop_thr:
            note[i] = "DROP"

        # V-SHAPE: rơi mạnh rồi hồi ngay (lọc báo giả)
        if abs(T[i-1]) >= drop_thr:
            if np.sign(T[i]) != np.sign(T[i-1]) and abs(T[i]) <= recover_thr:
                note[i] = "V-SHAPE"

        # DRIFT: nhỏ nhưng lì (>=3 bước cùng dấu, dưới drop_thr)
        if i >= 3:
            last = T[i-3:i+1]
            if np.all(np.sign(last) == np.sign(last[0])) and np.all(np.abs(last) < drop_thr):
                note[i] = "DRIFT"

    df = pd.DataFrame({
        "idx": np.arange(1, n+1),
        "raw": x,
        "pct_step": pct,
        "T": T,
        "E": E,
        "vT": vT,
        "vE": vE,
        "note": note
    })
    return df

def parse_10_values(raw_text: str):
    raw_text = raw_text.strip()
    if not raw_text:
        return None, "Ô input đang trống."
    try:
        vals = [float(v) for v in raw_text.split()]
    except Exception:
        return None, "Sai định dạng. Nhập 10 số cách nhau bằng dấu cách."
    if len(vals) != 10:
        return None, f"Cần đúng 10 giá trị, bạn đang nhập {len(vals)}."
    # tránh chia 0 ở bước % (x[i-1])
    if any(v == 0 for v in vals[:-1]):
        return None, "Có giá trị 0 ở vị trí trước đó → không tính %step được (chia 0)."
    return vals, None

def tab_ui(tab_id: str, title: str, k: float):
    st.subheader(title)

    raw = st.text_input(
        "Nhập 10 giá trị thô (cách nhau bằng dấu cách):",
        value="",
        key=f"{tab_id}_input"
    )

    if st.button("Compute", key=f"{tab_id}_btn"):
        vals, err = parse_10_values(raw)
        if err:
            st.error(err)
            return

        df = compute_dn(vals, k=k)

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{tab_id}.csv",
            mime="text/csv",
            key=f"{tab_id}_dl"
        )

st.title("DN v2 Demo — 4 tabs (HRV / HR / SpO₂ / RR) — dynamic-only notes")

tab1, tab2, tab3, tab4 = st.tabs(["HRV", "HR", "SpO₂", "RR"])

with tab1:
    tab_ui("hrv", "HRV (dynamic)", k=80)

with tab2:
    tab_ui("hr", "HR (dynamic)", k=15)

with tab3:
    tab_ui("spo2", "SpO₂ (dynamic)", k=5)

with tab4:
    tab_ui("rr", "RR (dynamic)", k=25)
