import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# App metadata
# ---------------------------------------------------------
APP_NAME = "App 2 – %HRV (Three Profiles)"
APP_VERSION = "1.1.0"

st.set_page_config(
    page_title=APP_NAME,
    layout="wide"
)

st.title(APP_NAME)
st.caption(f"Version: {APP_VERSION}")

st.write(
    "This app computes step-by-step percentage changes (%ΔHRV) for up to "
    "three individuals (A: high HRV, B: medium HRV, C: low HRV). "
    "It shows how normalization by percentage helps, but still cannot make "
    "different people fully overlap in one common geometry."
)

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def parse_hrv_input(text: str):
    """
    Parse an HRV string like '80, 75, 70' into a list[float].

    - Strips whitespace
    - Ignores empty items
    - Raises ValueError if something is not numeric
    """
    if not text:
        return []
    items = [x.strip() for x in text.split(",")]
    values = []
    for x in items:
        if x == "":
            continue
        try:
            values.append(float(x))
        except ValueError:
            raise ValueError(f"Invalid value: '{x}' (not a number).")
    return values


def compute_pct_hrv(hrv_list):
    """
    Compute step-by-step %ΔHRV.

    %ΔHRV[i] = 100 * (HRV[i] - HRV[i-1]) / HRV[i-1]
    First point is set to 0.0 (%).
    """
    n = len(hrv_list)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    pct = [0.0]  # first point = 0%
    for i in range(1, n):
        prev = hrv_list[i - 1]
        curr = hrv_list[i]
        if prev == 0:
            pct.append(0.0)
        else:
            pct_change = 100.0 * (curr - prev) / prev
            pct.append(pct_change)
    return pct


def make_raw_dict(a, b, c):
    """
    Bundle 3 raw HRV lists into a dict for plotting.
    Only include series that actually have data.
    """
    data = {}
    if a:
        data["A (high HRV)"] = a
    if b:
        data["B (medium HRV)"] = b
    if c:
        data["C (low HRV)"] = c
    return data


def make_pct_dict(a_pct, b_pct, c_pct):
    """
    Bundle 3 %HRV lists into a dict for plotting.
    """
    data = {}
    if a_pct:
        data["A (%HRV)"] = a_pct
    if b_pct:
        data["B (%HRV)"] = b_pct
    if c_pct:
        data["C (%HRV)"] = c_pct
    return data


# ---------------------------------------------------------
# Layout: 2 columns (input | plots)
# ---------------------------------------------------------

col_input, col_plot = st.columns([1, 2])

# ---------------- LEFT COLUMN: INPUT + BUTTON ----------------

with col_input:
    st.subheader("Input HRV values for three individuals")

    default_a = "80, 78, 76, 75, 77, 79, 80, 78, 76, 77"
    default_b = "60, 58, 56, 55, 57, 59, 60, 58, 56, 57"
    default_c = "40, 38, 36, 35, 37, 39, 40, 38, 36, 37"

    hrv_a_text = st.text_area(
        "Profile A – high HRV:",
        value=default_a,
        height=80,
        help="Example: 80, 78, 76, 75, 77..."
    )
    hrv_b_text = st.text_area(
        "Profile B – medium HRV:",
        value=default_b,
        height=80,
        help="Example: 60, 58, 56, 55, 57..."
    )
    hrv_c_text = st.text_area(
        "Profile C – low HRV:",
        value=default_c,
        height=80,
        help="Example: 40, 38, 36, 35, 37..."
    )

    st.caption(
        "Values must be separated by **commas**. "
        "You can paste them directly from Word / Excel."
    )

    calc_button = st.button("Compute & plot %HRV")


# ---------------- RIGHT COLUMN: PLOTS ----------------

with col_plot:
    st.subheader("Raw HRV and %ΔHRV (three profiles)")

    if calc_button:
        try:
            hrv_a = parse_hrv_input(hrv_a_text)
            hrv_b = parse_hrv_input(hrv_b_text)
            hrv_c = parse_hrv_input(hrv_c_text)

            if not (hrv_a or hrv_b or hrv_c):
                st.warning("No HRV data found. Please enter at least one profile.")
            else:
                # Raw HRV dataframe
                raw_dict = make_raw_dict(hrv_a, hrv_b, hrv_c)
                df_raw = pd.DataFrame(raw_dict)
                df_raw.index = range(1, len(df_raw) + 1)
                df_raw.index.name = "Measurement step"

                # %HRV dataframe
                a_pct = compute_pct_hrv(hrv_a) if hrv_a else []
                b_pct = compute_pct_hrv(hrv_b) if hrv_b else []
                c_pct = compute_pct_hrv(hrv_c) if hrv_c else []

                pct_dict = make_pct_dict(a_pct, b_pct, c_pct)
                df_pct = pd.DataFrame(pct_dict)
                df_pct.index = df_raw.index

                # Plots
                st.markdown("### Raw HRV (ms)")
                st.line_chart(df_raw, height=250)

                st.markdown("### Step-by-step %ΔHRV")
                st.line_chart(df_pct, height=250)

                st.markdown(
                    """
                    **Interpretation:**

                    - The top chart (raw HRV) still separates A, B and C clearly by baseline.
                    - The bottom chart (%ΔHRV) focuses on relative changes, so the three curves
                      move closer in shape – but they still do **not** fully overlap.
                    - This shows that percentage normalization improves things, but is not enough
                      to create a truly *unified* geometry for all individuals.

                    → In **App 3 (ET Mode)**, we will apply the Lorentz-based ET mapping, and the
                      three profiles collapse into almost the **same curve**, which %HRV alone
                      cannot achieve.
                    """
                )

        except ValueError as e:
            st.error(str(e))
    else:
        st.info(
            "Enter HRV values for A/B/C on the left, then click "
            "**“Compute & plot %HRV”** to see both raw and %HRV charts."
        )
