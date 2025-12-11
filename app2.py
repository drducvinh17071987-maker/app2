import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# App metadata
# ---------------------------------------------------------
APP_NAME = "App 2 – %HRV (Normalized HRV Dynamics)"
APP_VERSION = "1.0.0"

st.set_page_config(
    page_title=APP_NAME,
    layout="wide"
)

st.title(APP_NAME)
st.caption(f"Version: {APP_VERSION}")

st.write(
    "This app converts a raw HRV series (ms) into step-by-step percentage "
    "changes (%ΔHRV). It shows how the dynamics can be normalized, while the "
    "baseline HRV value is ignored."
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
    if len(hrv_list) == 0:
        return []

    pct = [0.0]  # first step = 0%
    for i in range(1, len(hrv_list)):
        prev = hrv_list[i - 1]
        curr = hrv_list[i]
        if prev == 0:
            pct.append(0.0)  # avoid division by zero, though physiologically HRV=0 shouldn't happen
        else:
            pct_change = 100.0 * (curr - prev) / prev
            pct.append(pct_change)
    return pct


# ---------------------------------------------------------
# Layout: 2 columns (input | plots)
# ---------------------------------------------------------

col_input, col_plot = st.columns([1, 2])

# ---------------- LEFT COLUMN: INPUT + BUTTON ----------------

with col_input:
    st.subheader("Input HRV series")

    default_series = "80, 75, 70, 78, 80, 76, 74, 77, 79, 78"

    hrv_text = st.text_area(
        "HRV values (ms), comma-separated:",
        value=default_series,
        height=100,
        help="Example: 80, 75, 70, 78, 80, 76..."
    )

    st.caption("Note: values must be separated by **commas**. No need for new lines.")

    calc_button = st.button("Compute %HRV and plot")

# ---------------- RIGHT COLUMN: PLOTS ----------------

with col_plot:
    st.subheader("HRV dynamics")

    if calc_button:
        try:
            hrv_values = parse_hrv_input(hrv_text)

            if not hrv_values:
                st.warning("No HRV data found. Please enter at least one value.")
            elif len(hrv_values) < 2:
                st.warning("Please enter at least 2 HRV values to compute percentage changes.")
            else:
                pct_values = compute_pct_hrv(hrv_values)

                # Build DataFrames for plotting
                index = range(1, len(hrv_values) + 1)

                df_raw = pd.DataFrame(
                    {"HRV_raw_ms": hrv_values},
                    index=index
                )
                df_raw.index.name = "Measurement step"

                df_pct = pd.DataFrame(
                    {"pct_HRV": pct_values},
                    index=index
                )
                df_pct.index.name = "Measurement step"

                # Plot raw HRV
                st.markdown("### Raw HRV (ms)")
                st.line_chart(df_raw, height=250)

                # Plot %ΔHRV
                st.markdown("### Step-by-step %ΔHRV")
                st.line_chart(df_pct, height=250)

                st.markdown(
                    """
                    **Quick interpretation:**
                    - Raw HRV (top chart) still depends on the absolute baseline.
                    - %ΔHRV (bottom chart) focuses only on *relative* changes between steps,  
                      so the same pattern of rise/fall can be compared across different baselines.
                    - However, as you will see in App 3, %HRV alone is still not enough to make
                      different individuals fully overlap in one common geometry.
                    """
                )

        except ValueError as e:
            st.error(str(e))
    else:
        st.info(
            "Paste comma-separated HRV values in the left column, then click "
            "**“Compute %HRV and plot”** to see the raw and %HRV charts here."
        )
