import streamlit as st, yaml
from pathlib import Path
import unicodedata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _norm(txt: str) -> str:
    """Replace any fancy dash with ASCII '-'."""
    return (
        unicodedata.normalize("NFKD", txt)
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )

# ────────────────── Robust config loader ──────────────────
@st.cache_data
def load_config() -> dict:
    base_dir = Path(__file__).parent
    cfg_path = base_dir / "config.yaml"

    # Show path info in sidebar for debugging
    # st.sidebar.caption(f"Config path: {cfg_path}  (exists: {cfg_path.exists()})")

    if not cfg_path.exists():
        st.error("config.yaml not found — place it next to app.py")
        st.stop()

    raw_config_content = cfg_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw_config_content)

    if cfg is None:
        st.error("config.yaml is empty or malformed.")
        st.stop()

    # Convert data paths to absolute so cwd never matters
    data_dir = base_dir / "data"
    for k in ("price_csv", "difficulty_csv", "hydro_xlsx"):
        cfg["data"][k] = str(data_dir / Path(cfg["data"][k]).name)

    return cfg


CONFIG = load_config()
config = CONFIG            # ← alias for backward compatibility

# ─────────────────────────────────────────────────────────────────────────────
# 2. Sidebar – analyst tweaks
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Scenario builder")

asic_models = {m["model"]: m for m in CONFIG["asics"]}
model_name   = st.sidebar.selectbox("ASIC model", list(asic_models.keys()))
asic         = asic_models[model_name]

# Finance settings -------------------------------------------------------------
fin_box = st.sidebar.expander("Finance assumptions", expanded=False)
with fin_box:
    discount_rate = st.number_input("Discount rate (%)", value=config["discount_rate"] * 100.0, step=0.5)
    resale_pct = st.number_input("Resale value (% of cap‑ex after 3 y)", value=config["resale_pct"] * 100.0, min_value=0.0, max_value=100.0, step=5.0)
    horizon_yrs = st.slider("Model horizon (years)", 1, 10, config["horizon_years"])
    n_paths = st.slider("Monte‑Carlo paths", 500, 5000, config["n_paths"], step=500)
    annual_fixed_cost = st.number_input("Annual fixed cost ($)", value=config.get("annual_fixed_cost", 60000), step=1000)

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
        annual_fixed_cost=annual_fixed_cost,
    )

    df = results["summary"].copy()

    # ---------- normalise dashed names so they match -----------------
    df.columns = [_norm(c) for c in df.columns]
    optim_metric = _norm(optim_metric)
    # -----------------------------------------------------------------

    # Determine sort order based on optimisation metric
    sort_ascending = optim_metric == "Median Pay-back (days)"
    df = df.sort_values(optim_metric, ascending=sort_ascending).reset_index(drop=True)

    st.success("Simulation finished ✅")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Fleet Economics", "⚡ Power Utilization", "📈 NPV Distribution", "📉 Detailed Analysis"])
    
    with tab1:
        st.subheader("Fleet‑size economics (sorted by chosen metric)")
        st.dataframe(df, use_container_width=True)

        best_row = df.iloc[0]
        st.markdown(
            f"### Optimal fleet size: **{int(best_row['ASICs'])} miners** "
            f"(by {optim_metric}: {best_row[optim_metric]:,.0f})"
        )

    with tab2:
        st.subheader("Power Utilization Analysis")
        
        # Load hydro data for visualization
        import pandas as pd
        hydro_df = pd.read_excel(Path(config["data"]["hydro_xlsx"]))
        hydro_kw = hydro_df["Generator Power (kW)"].values
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Utilization by fleet size
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.plot(df['ASICs'], df['Avg Utilization (%)'], 'b-o', linewidth=2, markersize=8)
            ax1.set_xlabel('Fleet Size (ASICs)', fontsize=12)
            ax1.set_ylabel('Average Utilization (%)', fontsize=12)
            ax1.set_title('Power Utilization vs Fleet Size', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            # Highlight optimal point
            optimal_idx = 0  # Best row
            ax1.plot(df.iloc[optimal_idx]['ASICs'], df.iloc[optimal_idx]['Avg Utilization (%)'], 
                    'ro', markersize=12, label=f"Optimal: {int(df.iloc[optimal_idx]['ASICs'])} ASICs")
            ax1.legend()
            st.pyplot(fig1)
            
        with col2:
            # Hydro power seasonal pattern
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            days = np.arange(len(hydro_kw))
            ax2.fill_between(days, 0, hydro_kw, alpha=0.5, color='blue', label='Available Power')
            ax2.set_xlabel('Day of Year', fontsize=12)
            ax2.set_ylabel('Power (kW)', fontsize=12)
            ax2.set_title('Hydro Power Availability Pattern', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add fleet power requirements as horizontal lines
            fleet_sizes_to_show = [30, 60, 90]
            colors = ['green', 'orange', 'red']
            for i, fleet in enumerate(fleet_sizes_to_show):
                power_req = fleet * asic["hash_rate_th"] * asic["watts_per_th"] / 1000
                ax2.axhline(y=power_req, color=colors[i], linestyle='--', 
                           label=f'{fleet} ASICs ({power_req:.0f} kW)')
            ax2.legend()
            st.pyplot(fig2)
        
        # Power utilization heatmap over time
        st.subheader("Daily Power Utilization Heatmap")
        
        # Calculate utilization for each day and fleet size
        fleet_sizes = df['ASICs'].values
        utilization_matrix = np.zeros((len(fleet_sizes), len(hydro_kw)))
        
        for i, fleet in enumerate(fleet_sizes):
            power_req = fleet * asic["hash_rate_th"] * asic["watts_per_th"] / 1000
            if power_req > 0:
                daily_util = np.minimum(hydro_kw / power_req, 1.0) * 100
                utilization_matrix[i, :] = daily_util
        
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        im = ax3.imshow(utilization_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
        ax3.set_yticks(range(len(fleet_sizes)))
        ax3.set_yticklabels(fleet_sizes)
        ax3.set_xlabel('Day of Year', fontsize=12)
        ax3.set_ylabel('Fleet Size (ASICs)', fontsize=12)
        ax3.set_title('Daily Power Utilization Heatmap (%)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Utilization (%)', fontsize=12)
        
        # Add text showing zero-power days
        zero_days = np.sum(hydro_kw == 0)
        st.caption(f"Note: {zero_days} days ({zero_days/len(hydro_kw)*100:.1f}%) have zero power availability")
        
        st.pyplot(fig3)

    with tab3:
        st.subheader("NPV Distribution Analysis")
        
        # Interactive selection
        sel_idx = st.number_input(
            "Inspect NPV distribution for fleet row #", 
            min_value=0, 
            max_value=len(df) - 1, 
            value=0,
            help="0 = best row; change to any other row index to inspect its distribution"
        )
        
        selected_fleet = df.iloc[sel_idx]['ASICs']
        selected_util = df.iloc[sel_idx]['Avg Utilization (%)']
        npv_paths = results["npv_paths_all"][int(sel_idx)]
        
        # Create figure with subplots
        fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(15, 6))
        
        # NPV histogram
        ax4.hist(npv_paths, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax4.axvline(np.median(npv_paths), color='green', linestyle='-', linewidth=2, label='Median')
        ax4.set_xlabel('NPV (USD)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title(f'NPV Distribution - {int(selected_fleet)} ASICs ({selected_util:.1f}% util)', 
                     fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # NPV percentile plot
        percentiles = np.arange(0, 101, 5)
        npv_percentiles = [np.percentile(npv_paths, p) for p in percentiles]
        ax5.plot(percentiles, npv_percentiles, 'b-', linewidth=2)
        ax5.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax5.fill_between(percentiles, npv_percentiles, 0, 
                        where=(np.array(npv_percentiles) > 0), 
                        alpha=0.3, color='green', label='Profitable zone')
        ax5.fill_between(percentiles, npv_percentiles, 0, 
                        where=(np.array(npv_percentiles) <= 0), 
                        alpha=0.3, color='red', label='Loss zone')
        ax5.set_xlabel('Percentile', fontsize=12)
        ax5.set_ylabel('NPV (USD)', fontsize=12)
        ax5.set_title('NPV Percentile Chart', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        st.pyplot(fig4)
        
        # Key statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("P5 (Worst case)", f"${np.percentile(npv_paths, 5):,.0f}")
        with col2:
            st.metric("P50 (Median)", f"${np.median(npv_paths):,.0f}")
        with col3:
            st.metric("P95 (Best case)", f"${np.percentile(npv_paths, 95):,.0f}")
        with col4:
            st.metric("Profit Probability", f"{(npv_paths > 0).mean() * 100:.1f}%")

    with tab4:
        st.subheader("Detailed Economic Analysis")
        
        # Create comparison metrics
        analysis_df = df[['ASICs', 'Avg Utilization (%)', 'CAPEX', 'Median NPV (USD)', 
                         'Probability NPV >0 (%)', 'Median Pay-back (days)']].copy()
        
        # Add additional calculated metrics
        analysis_df['NPV/CAPEX Ratio'] = analysis_df['Median NPV (USD)'] / analysis_df['CAPEX']
        analysis_df['Daily Profit'] = analysis_df['Median NPV (USD)'] / (horizon_yrs * 365)
        analysis_df['Annual ROI (%)'] = (analysis_df['Daily Profit'] * 365 / analysis_df['CAPEX']) * 100
        
        st.dataframe(analysis_df.style.format({
            'CAPEX': '${:,.0f}',
            'Median NPV (USD)': '${:,.0f}',
            'NPV/CAPEX Ratio': '{:.2f}',
            'Daily Profit': '${:,.0f}',
            'Annual ROI (%)': '{:.1f}%',
            'Avg Utilization (%)': '{:.1f}%',
            'Probability NPV >0 (%)': '{:.1f}%'
        }), use_container_width=True)
        
        # Summary insights
        st.info(f"""
        **Key Insights:**
        - Optimal fleet size: **{int(best_row['ASICs'])} ASICs** with {best_row['Avg Utilization (%)']}% utilization
        - Annual fixed cost impact: ${annual_fixed_cost:,} reduces daily profit by ${annual_fixed_cost/365:.0f}
        - Best NPV/CAPEX ratio: {analysis_df['NPV/CAPEX Ratio'].max():.2f}x at {analysis_df.loc[analysis_df['NPV/CAPEX Ratio'].idxmax(), 'ASICs']} ASICs
        - Hydro constraint: Power availability limits larger fleets despite potential scale economies
        """)

    # Download all results
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download full summary CSV", csv, file_name="fleet_optimisation.csv", mime="text/csv")

else:
    st.markdown("👈 Fill in scenario inputs on the left, then press **Run optimisation**.")
    
    # Show placeholder visualizations
    st.info("Run the optimization to see detailed power utilization analysis, NPV distributions, and economic comparisons.")

st.caption("Hydro‑constrained BTC mining economic optimisation • Streamlit prototype • v0.3")