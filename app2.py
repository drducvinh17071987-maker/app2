import streamlit as st
import pandas as pd
import numpy as np

# =========================
# Utilities (robust parsing)
# =========================
def parse_series(text: str, expected_len: int = 10):
    """
    Accepts numbers separated by spaces/commas/newlines.
    Returns a list[float] of length expected_len or raises ValueError with a clean message.
    """
    if text is None:
        raise ValueError("Empty input.")
    s = text.replace(",", " ").replace("\n", " ").strip()
    parts = [p for p in s.split(" ") if p.strip() != ""]
    if len(parts) != expected_len:
        raise ValueError(f"Please enter exactly {expected_len} values (you entered {len(parts)}).")
    try:
        vals = [float(p) for p in parts]
    except:
        raise ValueError("Invalid number found. Use only digits, dot, minus, spaces/commas.")
    return vals

def compute_core(values: list[float], k: float):
    """
    Core engine (kept consistent across systems):
    pct[i] = 100*(x[i]-x[i-1])/x[i-1], pct[0]=0
    t[i]   = pct[i]/k
    e[i]   = 1 - t[i]^2
    vT[i]  = t[i] - t[i-1], vT[0]=0
    vE[i]  = e[i] - e[i-1], vE[0]=0
    """
    x = np.array(values, dtype=float)
    pct = np.zeros_like(x)
    pct[1:] = 100.0 * (x[1:] - x[:-1]) / np.where(x[:-1] == 0, np.nan, x[:-1])
    pct = np.nan_to_num(pct, nan=0.0, posinf=0.0, neginf=0.0)

    t = pct / float(k)
    e = 1.0 - (t ** 2)

    vT = np.zeros_like(t)
    vE = np.zeros_like(e)
    vT[1:] = t[1:] - t[:-1]
    vE[1:] = e[1:] - e[:-1]

    return pct, t, e, vT, vE

def make_df(values, t, e, vT, vE, status, note):
    n = len(values)
    df = pd.DataFrame({
        "minute": np.arange(1, n + 1),
        "value": values,
        "t": t,
        "e": e,
        "vt": vT,
        "ve": vE,
        "status": status,
        "note": note
    })
    # format (avoid hiding small non-zero values)
    df["t"] = df["t"].map(lambda z: float(f"{z:.4f}"))
    df["e"] = df["e"].map(lambda z: float(f"{z:.4f}"))
    df["vt"] = df["vt"].map(lambda z: float(f"{z:.4f}"))
    df["ve"] = df["ve"].map(lambda z: float(f"{z:.4f}"))
    return df

# =========================
# HRV v2 (pattern-only)
# =========================
def hrv_status_note(values, pct, t, e, vT, vE):
    """
    HRV: NO thresholds for RED/WARNING.
    Only: status = BASE or PATTERN
    note must never be empty: flat / micro-fluctuation / step-drop / drift / v-shape / noise-spike
    """
    n = len(values)
    status = ["BASE"] * n
    note = ["flat"] * n

    # helper: consecutive drift detection on pct sign
    def is_drift_down(i):
        if i < 3: 
            return False
        return (pct[i-2] < 0) and (pct[i-1] < 0) and (pct[i] < 0)

    def is_drift_up(i):
        if i < 3:
            return False
        return (pct[i-2] > 0) and (pct[i-1] > 0) and (pct[i] > 0)

    # V-shape (3-point): drop then recover, net small
    def is_v_shape(i):
        if i < 2:
            return False
        d1 = pct[i-1]     # step into middle
        d2 = pct[i]       # step out
        total = 100.0 * (values[i] - values[i-2]) / (values[i-2] if values[i-2] != 0 else 1.0)
        # Your stored v-shape rule for HRV:
        return (d1 <= -20.0) and (d2 >= 15.0) and (abs(total) <= 12.0)

    # noise-brake v1.5 idea: very fast spike up
    def is_noise_spike(i):
        if i < 1:
            return False
        delta = values[i] - values[i-1]
        return (pct[i] >= 70.0) or (delta >= 60.0)

    for i in range(n):
        if i == 0:
            status[i] = "BASE"
            note[i] = "flat"
            continue

        if abs(pct[i]) < 1e-9:
            status[i] = "BASE"
            note[i] = "flat"
            continue

        status[i] = "PATTERN"
        # default
        note[i] = "micro-fluctuation"

        # priority rules (more specific overrides generic)
        if is_noise_spike(i):
            note[i] = "noise-spike"
        if pct[i] <= -40.0:
            note[i] = "step-drop"
        if is_v_shape(i):
            note[i] = "v-shape"
        if is_drift_down(i):
            note[i] = "drift-down"
        if is_drift_up(i):
            note[i] = "drift-up"

    return status, note

# =========================
# Thresholded systems: SpO2 / RR / HR
# =========================
def threshold_status_note(values, pct, t, e, vT, vE, thr_warn=0.3, thr_red=0.6):
    """
    status: BASE / PATTERN
    note: GREEN / MILD / WARNING / RED / INFO (v-shape)
    V-shape should override RED/WARNING if it's an immediate drop+recovery pattern (filter false alarms).
    """
    n = len(values)
    status = ["BASE"] * n
    note = ["GREEN"] * n

    def is_v_shape(i):
        if i < 2:
            return False
        # drop then recovery next step
        d1 = t[i-1]
        d2 = t[i]
        # immediate sign flip + meaningful magnitude
        return (abs(d1) >= thr_warn) and (abs(d2) >= thr_warn) and (np.sign(d1) != np.sign(d2))

    for i in range(n):
        if i == 0:
            status[i] = "BASE"
            note[i] = "GREEN"
            continue

        if abs(t[i]) < 1e-9:
            status[i] = "BASE"
            note[i] = "GREEN"
            continue

        status[i] = "PATTERN"

        # V-shape filter has priority for the "recovery step" (i)
        if is_v_shape(i):
            note[i] = "INFO"
            continue

        a = abs(t[i])
        if a >= thr_red:
            note[i] = "RED"
        elif a >= thr_warn:
            note[i] = "WARNING"
        elif a >= 0.2:
            note[i] = "MILD"
        else:
            note[i] = "GREEN"

    return status, note

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="DN v2 Demo", layout="wide")

tabs = st.tabs(["hrv", "spo2", "rr", "hr"])

DEFAULT = {
    "hrv": "50 49 48 33 32 31 44 43 42 41",
    "spo2": "98 97 96 66 94 93 92 91 90 89",
    "rr": "16 16 17 18 43 17 16 15 16 18",
    "hr": "70 71 72 90 88 86 85 84 83 82"
}

def render_tab(key, k_value, mode):
    st.subheader(f"{key.upper()} (10 points)")
    txt = st.text_input("", value=DEFAULT[key], key=f"input_{key}")
    if st.button("Compute", key=f"btn_{key}"):
        try:
            values = parse_series(txt, 10)
            pct, t, e, vT, vE = compute_core(values, k=k_value)

            if mode == "hrv":
                status, note = hrv_status_note(values, pct, t, e, vT, vE)
            else:
                # thresholded: note shows GREEN/MILD/WARNING/RED; v-shape => INFO
                status, note = threshold_status_note(values, pct, t, e, vT, vE, thr_warn=0.3, thr_red=0.6)

            df = make_df(values, t, e, vT, vE, status, note)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, file_name=f"{key}_dn_v2.csv", mime="text/csv")

        except ValueError as ve:
            st.error(str(ve))
        except Exception as ex:
            st.error("Unexpected error. Please check input format and retry.")

with tabs[0]:
    # HRV: k=80 dynamic
    render_tab("hrv", k_value=80, mode="hrv")

with tabs[1]:
    # SpO2: k=5 (as you already used)
    render_tab("spo2", k_value=5, mode="thresholded")

with tabs[2]:
    # RR: k=25
    render_tab("rr", k_value=25, mode="thresholded")

with tabs[3]:
    # HR: k=15
    render_tab("hr", k_value=15, mode="thresholded")
