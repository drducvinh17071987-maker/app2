# app.py  (Streamlit)
# DN Dynamic 4-tabs: HRV / HR / SpO2 / RR
# - Input: 10 raw values (space/comma/newline separated)
# - Output columns: raw, pct_step, T, E, vT, vE, note
# - NOTE only shows: OK / STEP_MILD / STEP_MID / STEP_BIG / V-SHAPE_DROP / V-SHAPE_RECOVER / DRIFT
# - No GREEN/RED, no STATUS.
#
# Chốt K:
#   HRV: 80
#   HR : 15  (theo bạn chốt)
#   RR : 25
#   SpO2: 5  (chỉ để chuẩn hoá %step -> T; ngưỡng note dựa trên |T|)

import re
import numpy as np
import pandas as pd
import streamlit as st


# ---------- Core parsing ----------
def parse_series(text: str, n_expected: int = 10):
    if text is None:
        return None, "Empty input."
    s = text.strip()
    if not s:
        return None, "Empty input."
    # split by comma/space/newline/semicolon
    parts = re.split(r"[\s,;]+", s)
    vals = []
    for p in parts:
        if p == "":
            continue
        try:
            vals.append(float(p))
        except:
            return None, f"Cannot parse value: {p}"
    if len(vals) != n_expected:
        return None, f"Need exactly {n_expected} values, got {len(vals)}."
    # avoid zeros in denominator where needed
    if any(v == 0 for v in vals):
        # allow raw zeros, but pct_step will break when prev is 0
        # we will guard later
        pass
    return vals, None


# ---------- DN Dynamic core ----------
def compute_dn_dynamic(series, k: float,
                       big_thr=0.6, mid_thr=0.3, mild_thr=0.2,
                       drift_len=3):
    x = np.array(series, dtype=float)
    n = len(x)

    pct = np.zeros(n, dtype=float)
    T = np.zeros(n, dtype=float)
    E = np.ones(n, dtype=float)
    vT = np.zeros(n, dtype=float)
    vE = np.zeros(n, dtype=float)

    note = ["OK"] * n

    # step computations
    for i in range(1, n):
        if x[i-1] == 0:
            pct[i] = np.nan
            T[i] = np.nan
            E[i] = np.nan
            vT[i] = np.nan
            vE[i] = np.nan
            note[i] = "OK"
            continue

        pct[i] = 100.0 * (x[i] - x[i-1]) / x[i-1]
        T[i] = pct[i] / k
        E[i] = 1.0 - (T[i] ** 2)

        vT[i] = T[i] - T[i-1]
        vE[i] = E[i] - E[i-1]

    # 1) base step label
    for i in range(1, n):
        if np.isnan(T[i]):
            continue
        a = abs(T[i])
        if a >= big_thr:
            note[i] = "STEP_BIG"
        elif a >= mid_thr:
            note[i] = "STEP_MID"
        elif a >= mild_thr:
            note[i] = "STEP_MILD"
        else:
            note[i] = "OK"

    # 2) V-shape override: BIG step then immediate opposite step (>= MID)
    for i in range(1, n - 1):
        if np.isnan(T[i]) or np.isnan(T[i + 1]):
            continue
        if abs(T[i]) >= big_thr and np.sign(T[i]) != 0:
            if abs(T[i + 1]) >= mid_thr and np.sign(T[i + 1]) == -np.sign(T[i]):
                note[i] = "V-SHAPE_DROP"
                note[i + 1] = "V-SHAPE_RECOVER"

    # 3) DRIFT: drift_len consecutive small steps (< MID) same direction
    # mark only the last point of the drift window
    if drift_len >= 3:
        for i in range(drift_len, n):
            seg = T[i - (drift_len - 1): i + 1]
            if np.any(np.isnan(seg)):
                continue
            if np.all(np.abs(seg) < mid_thr):
                sgn = np.sign(seg)
                if np.all(sgn == sgn[0]) and sgn[0] != 0:
                    # do not overwrite V-shape labels
                    if not note[i].startswith("V-SHAPE"):
                        note[i] = "DRIFT"

    df = pd.DataFrame({
        "idx": np.arange(1, n + 1),
        "raw": x,
        "pct_step": pct,
        "T": T,
        "E": E,
        "vT": vT,
        "vE": vE,
        "note": note
    })
    return df


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------- UI ----------
st.set_page_config(page_title="DN Dynamic (4 tabs)", layout="wide")

st.title("DN Dynamic — 4 tabs (10 points)")
st.caption("Nhập đúng 10 giá trị thô, cách nhau bởi dấu cách / dấu phẩy / xuống dòng. Bấm Compute để ra bảng động học (pct_step, T, E, vT, vE) và NOTE (V-shape/Step/Drift).")

tabs = st.tabs(["HRV (k=80)", "HR (k=15)", "SpO2 (k=5)", "RR (k=25)"])

TAB_CONFIG = {
    "HRV (k=80)": {"k": 80.0},
    "HR (k=15)": {"k": 15.0},
    "SpO2 (k=5)": {"k": 5.0},
    "RR (k=25)": {"k": 25.0},
}

def render_tab(tab_label: str):
    k = TAB_CONFIG[tab_label]["k"]

    # Unique widget keys per tab to avoid StreamlitDuplicateElementId
    raw = st.text_input(
        "Nhập 10 giá trị thô:",
        value="",
        key=f"input_{tab_label}",
        placeholder="ví dụ: 45 44 43 42 41 40 41 42 43 44"
    )

    colA, colB = st.columns([1, 3])
    with colA:
        do = st.button("Compute", key=f"btn_{tab_label}")
    with colB:
        st.markdown(
            f"- **k = {k:g}**  •  NOTE thresholds on |T|: mild≥0.2, mid≥0.3, big≥0.6  •  V-shape = big step + immediate opposite step"
        )

    if do:
        series, err = parse_series(raw, n_expected=10)
        if err:
            st.error(err)
            return

        df = compute_dn_dynamic(series, k=k)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download CSV",
            data=df_to_csv_bytes(df),
            file_name=f"dn_dynamic_{tab_label.split()[0].lower()}.csv",
            mime="text/csv",
            key=f"dl_{tab_label}"
        )

with tabs[0]:
    render_tab("HRV (k=80)")
with tabs[1]:
    render_tab("HR (k=15)")
with tabs[2]:
    render_tab("SpO2 (k=5)")
with tabs[3]:
    render_tab("RR (k=25)")
