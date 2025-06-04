import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
import os, streamlit as st

"""
Streamlit front‑end for the hydro‑constrained Bitcoin‑mining model
------------------------------------------------------------------
Business logic tweak (2025‑06‑04):
    • The *model* — not the user — must search the fleet‑size space and report
      the economically optimal number of ASICs.
    • The app therefore no longer asks the analyst to pick a fleet size.
    • Instead the user chooses the ASIC model *and* their preferred
      optimisation metric (e.g. highest median NPV, shortest pay‑back, etc.).
    • The results table is automatically sorted by that metric and the top row
      is highlighted; clicking any row reveals the full NPV distribution.
"""

# -----------------------------------------------------------------------------
# 1. Load YAML configuration
# -----------------------------------------------------------------------------
@st.cache_data
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# -----------------------------------------------------------------------------
# 2. Sidebar – analyst tweaks
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Scenario builder")

asic_models = {m["model"]: m for m in config["asics"]}
model_name = st.sidebar.selectbox("ASIC model", list(asic_models.keys()))
asic = asic_models[model_name]

# Finance settings -------------------------------------------------------------
fin_box = st.sidebar.expander("Finance assumptions", expanded=False)
with fin_box:
    discount_rate = st.number_input("Discount rate (%)", value=config["discount_rate"] * 100.0, step=0.5)
    resale_pct = st.number_input("Resale value (% of cap‑ex after 3 y)", value=config["resale_pct"] * 100.0, min_value=0.0, max_value=100.0, step=5.0)
    horizon_yrs = st.slider("Model horizon (years)", 1, 10, config["horizon_years"])
    n_paths = st.slider("Monte‑Carlo paths", 500, 5000, config["n_paths"], step=500)

optim_metric = st.sidebar.selectbox(
    "Optimise for …",
    ("Median NPV (USD)", "Probability NPV >0 (%)", "Median Pay‑back (days)")
)

run_pressed = st.sidebar.button("▶ Run optimisation")

# -----------------------------------------------------------------------------
# 3. Main panel logic
# -----------------------------------------------------------------------------
if run_pressed:
    st.toast("Crunching Monte‑Carlo …", icon="⏳")

    from hydro_miner_model import run_monte_carlo

    results = run_monte_carlo(
        asic=asic,
        discount_rate=discount_rate / 100.0,
        resale_pct=resale_pct / 100.0,
        n_paths=n_paths,
        horizon_years=horizon_yrs,
        price_csv=Path(config["data"]["price_csv"]),
        diff_csv=Path(config["data"]["difficulty_csv"]),
        hydro_xlsx=Path(config["data"]["hydro_xlsx"]),
        fleet_sizes=range(config["fleet_min"], config["fleet_max"] + 1, config["fleet_step"]),
    )

    df = results["summary"].copy()

    # Determine sort order based on optimisation metric -----------------------
    sort_ascending = False
    if optim_metric == "Median Pay‑back (days)":
        sort_ascending = True
    df = df.sort_values(optim_metric, ascending=sort_ascending).reset_index(drop=True)

    st.success("Simulation finished ✅")

    st.subheader("Fleet‑size economics (sorted by chosen metric)")
    st.dataframe(df, use_container_width=True)

    best_row = df.iloc[0]
    st.markdown(
        f"### Optimal fleet size: **{int(best_row['ASICs'])} miners** "
        f"(by {optim_metric}: {best_row[optim_metric]:,.0f})"
    )

    # Interactive selection ----------------------------------------------------
    sel_idx = st.number_input(
        "Inspect NPV distribution for fleet row #", min_value=0, max_value=len(df) - 1, value=0,
        help="0 = best row above; change to any other row index to inspect its distribution"
    )
    npv_paths = results["npv_paths_all"][int(sel_idx)]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(npv_paths, bins=100)
    ax.set_xlabel("NPV (USD)")
    ax.set_ylabel("Frequency")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    st.pyplot(fig)

    # Download all results -----------------------------------------------------
    csv = df.to_csv(index=False).encode()
    st.download_button("Download full summary CSV", csv, file_name="fleet_optimisation.csv", mime="text/csv")

else:
    st.markdown("👈 Fill in scenario inputs on the left, then press **Run optimisation**.")

st.caption("Hydro‑constrained BTC mining economic optimisation • Streamlit prototype • v0.2")
st.write("CWD:", os.getcwd())