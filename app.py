"""
Production Streamlit app for hydro-constrained Bitcoin mining optimization.
Implements dynamic fleet sizing with comprehensive reporting and error handling.
"""

import streamlit as st
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional

# Import our production model
from hydro_miner_model import (
    run_monte_carlo, ASICSpec, log_sizing_summary,
    calculate_utilization_metrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Bitcoin Mining Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONFIGURATION LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_config() -> Dict[str, Any]:
    """Load and validate configuration file."""
    try:
        config_path = Path(__file__).parent / "config.yaml"
        
        if not config_path.exists():
            st.error("❌ Configuration file not found. Please ensure config.yaml exists.")
            st.stop()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Convert relative paths to absolute
        data_dir = Path(__file__).parent / "data"
        for key in config["data"]:
            if config["data"][key] and key.endswith(("csv", "xlsx")):
                config["data"][key] = str(data_dir / Path(config["data"][key]).name)
        
        # Validate configuration
        validate_config(config)
        
        return config
        
    except Exception as e:
        st.error(f"❌ Failed to load configuration: {str(e)}")
        st.stop()


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure and values."""
    required_sections = ["simulation", "data", "fleet_optimization", "asics"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate numeric ranges
    if not 0 < config["simulation"]["discount_rate"] < 1:
        raise ValueError("Discount rate must be between 0 and 1")
    
    if config["fleet_optimization"]["min_utilization"] >= config["fleet_optimization"]["max_utilization"]:
        raise ValueError("min_utilization must be less than max_utilization")


# ─────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────

def render_sidebar(config: Dict[str, Any]) -> Dict[str, Any]:
    """Render sidebar controls and return user inputs."""
    
    st.sidebar.markdown("# ⚡ Bitcoin Mining Optimizer")
    st.sidebar.markdown("---")
    
    # ASIC Selection
    st.sidebar.markdown("### 🖥️ ASIC Selection")
    asic_models = {m["model"]: ASICSpec(**m) for m in config["asics"]}
    selected_model = st.sidebar.selectbox(
        "ASIC Model",
        options=list(asic_models.keys()),
        help="Select the ASIC miner model to optimize for"
    )
    asic = asic_models[selected_model]
    
    # Display ASIC specs
    with st.sidebar.expander("📋 ASIC Specifications", expanded=False):
        st.markdown(f"""
        - **Hashrate**: {asic.hash_rate_th:.0f} TH/s
        - **Power**: {asic.power_kw:.1f} kW ({asic.watts_per_th:.1f} W/TH)
        - **Price**: ${asic.price_usd:,.0f} (${asic.price_usd_per_th:.1f}/TH)
        """)
    
    st.sidebar.markdown("---")
    
    # Scenario Selection
    st.sidebar.markdown("### 📊 Scenario Analysis")
    
    scenario_presets = {
        "Conservative": {
            "discount_rate": 0.20,
            "salvage_pct": 0.00,
            "include_all_costs": True
        },
        "Moderate": {
            "discount_rate": 0.15,
            "salvage_pct": 0.05,
            "include_all_costs": True
        },
        "Aggressive": {
            "discount_rate": 0.10,
            "salvage_pct": 0.20,
            "include_all_costs": False
        },
        "Custom": None
    }
    
    scenario = st.sidebar.selectbox(
        "Risk Profile",
        options=list(scenario_presets.keys()),
        help="Select a pre-configured scenario or customize"
    )
    
    # Apply scenario or show custom controls
    if scenario == "Custom":
        with st.sidebar.expander("💰 Financial Assumptions", expanded=True):
            discount_rate = st.slider(
                "Discount Rate (%)",
                min_value=5.0,
                max_value=30.0,
                value=15.0,
                step=0.5,
                help="Annual discount rate for NPV calculation"
            ) / 100
            
            salvage_pct = st.slider(
                "Salvage Value (% of CAPEX)",
                min_value=0.0,
                max_value=30.0,
                value=5.0,
                step=1.0,
                help="Expected resale value after 3 years"
            ) / 100
            
            include_all_costs = st.checkbox(
                "Include all operating costs",
                value=True,
                help="Include insurance, monitoring, and per-ASIC maintenance"
            )
    else:
        preset = scenario_presets[scenario]
        discount_rate = preset["discount_rate"]
        salvage_pct = preset["salvage_pct"]
        include_all_costs = preset["include_all_costs"]
        
        # Show applied values
        with st.sidebar.expander("📍 Applied Settings", expanded=False):
            st.markdown(f"""
            - **Discount Rate**: {discount_rate:.1%}
            - **Salvage Value**: {salvage_pct:.0%}
            - **All Costs**: {"Yes" if include_all_costs else "No"}
            """)
    
    st.sidebar.markdown("---")
    
    # Advanced Settings
    with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
        # Simulation quality
        sim_quality = st.select_slider(
            "Simulation Quality",
            options=["Draft", "Standard", "High", "Maximum"],
            value="Standard",
            help="Higher quality = more accurate but slower"
        )
        
        quality_map = {
            "Draft": 100,
            "Standard": 2000,
            "High": 5000,
            "Maximum": 10000
        }
        n_paths = quality_map[sim_quality]
        
        # Fleet sizing method
        sizing_method = st.radio(
            "Fleet Sizing Method",
            options=["Dynamic (Recommended)", "Manual"],
            help="Dynamic sizing optimizes based on hydro capacity"
        )
        
        if sizing_method == "Manual":
            col1, col2 = st.columns(2)
            with col1:
                fleet_min = st.number_input("Min Fleet Size", value=10, min_value=1)
            with col2:
                fleet_max = st.number_input("Max Fleet Size", value=100, min_value=fleet_min)
            fleet_sizes = list(range(fleet_min, fleet_max + 1, 10))
        else:
            fleet_sizes = None  # Will be calculated dynamically
        
        # Optimization metric
        optim_metric = st.selectbox(
            "Optimize for",
            options=[
                "Median NPV (USD)",
                "Probability NPV >0 (%)",
                "Sharpe Ratio",
                "Median IRR (%)"
            ],
            help="Primary metric for fleet size selection"
        )
    
    st.sidebar.markdown("---")
    
    # Run button
    run_simulation = st.sidebar.button(
        "🚀 Run Optimization",
        type="primary",
        use_container_width=True
    )
    
    # Package inputs
    inputs = {
        "asic": asic,
        "scenario": scenario,
        "discount_rate": discount_rate,
        "salvage_pct": salvage_pct,
        "include_all_costs": include_all_costs,
        "n_paths": n_paths,
        "fleet_sizes": fleet_sizes,
        "optim_metric": optim_metric,
        "run_simulation": run_simulation
    }
    
    return inputs


# ─────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    
    # Load configuration
    config = load_config()
    
    # Get user inputs from sidebar
    inputs = render_sidebar(config)
    
    # Header
    st.markdown("# ⚡ Hydro-Constrained Bitcoin Mining Optimizer")
    st.markdown("*Dynamic fleet sizing for maximum profitability*")
    
    # Information boxes
    if not inputs["run_simulation"]:
        st.info("👈 Configure your scenario and click **Run Optimization** to begin")
        
        # Show example visualizations
        with st.expander("📊 Example Analysis Preview", expanded=True):
            fig = create_example_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        return
    
    # Run simulation
    try:
        with st.spinner("⚡ Running Monte Carlo simulation..."):
            # Update config with user inputs
            sim_config = config.copy()
            sim_config["simulation"]["discount_rate"] = inputs["discount_rate"]
            sim_config["simulation"]["salvage_pct"] = inputs["salvage_pct"]
            sim_config["simulation"]["n_paths"] = inputs["n_paths"]
            
            # Adjust costs based on selection
            if not inputs["include_all_costs"]:
                sim_config["simulation"]["insurance_cost"] = 0
                sim_config["simulation"]["monitoring_cost"] = 0
                sim_config["simulation"]["maintenance_per_asic"] = 0
            
            # Run simulation
            results = run_monte_carlo(
                asic=inputs["asic"],
                config=sim_config,
                price_csv=Path(config["data"]["price_csv"]),
                diff_csv=Path(config["data"]["difficulty_csv"]),
                hydro_xlsx=Path(config["data"]["hydro_xlsx"]),
                fleet_sizes=inputs["fleet_sizes"]
            )
        
        st.success("✅ Optimization complete!")
        
        # Display results
        display_results(results, inputs, config)
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        st.error(f"❌ Simulation failed: {str(e)}")
        st.error("Please check your data files and configuration.")


def display_results(results, inputs, config):
    """Display comprehensive results with multiple tabs."""
    
    # Create tabs for different views
    tabs = st.tabs([
        "📊 Summary",
        "⚡ Fleet Analysis", 
        "📈 Economics",
        "🎲 Risk Analysis",
        "📋 Detailed Report"
    ])
    
    # Get best configuration
    optim_col = inputs["optim_metric"]
    if "Median NPV" in optim_col or "Sharpe" in optim_col:
        best_idx = results.summary_df[optim_col].idxmax()
    else:
        best_idx = results.summary_df[optim_col].idxmin()
    
    best_config = results.summary_df.iloc[best_idx]
    
    # Tab 1: Summary
    with tabs[0]:
        display_summary_tab(results, best_config, inputs)
    
    # Tab 2: Fleet Analysis
    with tabs[1]:
        display_fleet_analysis_tab(results, config)
    
    # Tab 3: Economics
    with tabs[2]:
        display_economics_tab(results, best_idx)
    
    # Tab 4: Risk Analysis
    with tabs[3]:
        display_risk_analysis_tab(results, best_idx)
    
    # Tab 5: Detailed Report
    with tabs[4]:
        display_detailed_report_tab(results, inputs, config)


def display_summary_tab(results, best_config, inputs):
    """Display executive summary."""
    
    st.markdown("## 📊 Executive Summary")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Optimal Fleet Size",
            f"{int(best_config['ASICs'])} ASICs",
            f"{best_config['Avg Utilization (%)']:.0f}% utilization"
        )
    
    with col2:
        st.metric(
            "Expected NPV",
            f"${best_config['Median NPV (USD)']:,.0f}",
            f"{best_config['Probability NPV >0 (%)']:.0f}% profit chance"
        )
    
    with col3:
        median_irr = best_config.get('Median IRR (%)')
        irr_display = f"IRR: {median_irr:.0f}%" if median_irr is not None else "IRR: N/A"
        st.metric(
            "Payback Period",
            f"{best_config['Median Pay-back (days)']:.0f} days" if best_config['Median Pay-back (days)'] else "N/A",
            irr_display
        )
    
    with col4:
        st.metric(
            "Total Investment",
            f"${best_config['CAPEX']:,.0f}",
            f"Sharpe: {best_config.get('Sharpe Ratio', 0):.2f}"
        )
    
    # Warnings if any
    if results.sizing_info and results.sizing_info.warnings:
        st.markdown("### ⚠️ Warnings")
        for warning in results.sizing_info.warnings:
            st.warning(warning)
    
    # Quick recommendation
    st.markdown("### 💡 Recommendation")
    
    if best_config['Probability NPV >0 (%)'] > 70:
        st.success(
            f"✅ **Strong Investment**: The optimal configuration of {int(best_config['ASICs'])} ASICs "
            f"shows a {best_config['Probability NPV >0 (%)']:.0f}% probability of profit with "
            f"expected NPV of ${best_config['Median NPV (USD)']:,.0f}."
        )
    elif best_config['Probability NPV >0 (%)'] > 50:
        st.warning(
            f"⚠️ **Moderate Risk**: The optimal configuration shows positive expected returns but with "
            f"only {best_config['Probability NPV >0 (%)']:.0f}% probability of profit. Consider risk tolerance."
        )
    else:
        st.error(
            f"❌ **High Risk**: All configurations show less than 50% probability of profit. "
            f"Consider waiting for better market conditions or reducing costs."
        )


def display_fleet_analysis_tab(results, config):
    """Display fleet sizing and utilization analysis."""
    
    st.markdown("## ⚡ Fleet Sizing Analysis")
    
    # Load hydro data for visualization
    hydro_df = pd.read_excel(Path(config["data"]["hydro_xlsx"]))
    hydro_kw = hydro_df["Generator Power (kW)"].values
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Fleet size vs metrics
        fig = px.line(
            results.summary_df,
            x='ASICs',
            y=['Median NPV (USD)', 'Probability NPV >0 (%)'],
            title='Fleet Size Impact on Profitability',
            labels={'value': 'Value', 'ASICs': 'Fleet Size'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Utilization curve
        fig = px.scatter(
            results.summary_df,
            x='Avg Utilization (%)',
            y='Median NPV (USD)',
            size='CAPEX',
            color='Probability NPV >0 (%)',
            title='Utilization vs Profitability',
            hover_data=['ASICs']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Hydro capacity visualization
    st.markdown("### 💧 Hydro Power Profile")
    
    # Create seasonal pattern chart
    days = np.arange(len(hydro_kw))
    
    fig = go.Figure()
    
    # Add hydro capacity
    fig.add_trace(go.Scatter(
        x=days,
        y=hydro_kw,
        mode='lines',
        name='Available Power',
        fill='tozeroy',
        line=dict(color='blue', width=1)
    ))
    
    # Add fleet power requirements for key sizes
    if len(results.summary_df) > 0:
        # Use ASIC power from inputs['asic']
        from inspect import currentframe
        frame = currentframe()
        try:
            # Try to get 'inputs' from the calling frame
            inputs = frame.f_back.f_locals.get('inputs', None)
            asic_power_kw = inputs['asic'].power_kw if inputs else 1
        finally:
            del frame
        for idx in [0, len(results.summary_df)//2, len(results.summary_df)-1]:
            if idx < len(results.summary_df):
                row = results.summary_df.iloc[idx]
                fleet_power = row['ASICs'] * asic_power_kw
                fig.add_hline(
                    y=fleet_power,
                    line_dash="dash",
                    annotation_text=f"{int(row['ASICs'])} ASICs",
                    annotation_position="right"
                )
    
    fig.update_layout(
        title='Hydro Power Availability vs Fleet Requirements',
        xaxis_title='Day of Year',
        yaxis_title='Power (kW)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_economics_tab(results, best_idx):
    """Display detailed economic analysis."""
    
    st.markdown("## 📈 Economic Analysis")
    
    # NPV distribution for best configuration
    best_npv_paths = results.npv_paths_all[best_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # NPV histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=best_npv_paths,
            nbinsx=50,
            name='NPV Distribution',
            marker_color='blue',
            opacity=0.7
        ))
        
        # Add markers
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.add_vline(x=np.median(best_npv_paths), line_dash="solid", line_color="green", annotation_text="Median")
        
        fig.update_layout(
            title='NPV Distribution (Optimal Fleet)',
            xaxis_title='NPV (USD)',
            yaxis_title='Frequency',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Percentile chart
        percentiles = np.arange(0, 101, 5)
        npv_percentiles = [np.percentile(best_npv_paths, p) for p in percentiles]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=percentiles,
            y=npv_percentiles,
            mode='lines',
            fill='tozeroy',
            line=dict(color='green', width=2)
        ))
        
        # Color regions
        fig.add_hrect(y0=min(npv_percentiles), y1=0, fillcolor="red", opacity=0.2)
        
        fig.update_layout(
            title='NPV Percentile Analysis',
            xaxis_title='Percentile',
            yaxis_title='NPV (USD)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Sensitivity analysis
    st.markdown("### 📊 Sensitivity Analysis")
    
    # Create sensitivity data
    sensitivity_params = ['BTC Price', 'Difficulty', 'Utilization', 'Discount Rate']
    sensitivity_changes = [-20, -10, 0, 10, 20]
    
    # This would need actual recalculation in production
    # For demo, using approximate sensitivities
    base_npv = results.summary_df.iloc[best_idx]['Median NPV (USD)']
    
    sensitivity_data = []
    for param in sensitivity_params:
        for change in sensitivity_changes:
            # Simplified sensitivity calculation
            if param == 'BTC Price':
                impact = base_npv * (1 + change/100) - base_npv
            elif param == 'Difficulty':
                impact = base_npv * (1 - change/100*0.5) - base_npv
            elif param == 'Utilization':
                impact = base_npv * (1 + change/100*0.8) - base_npv
            else:  # Discount Rate
                impact = base_npv * (1 - change/100*0.3) - base_npv
            
            sensitivity_data.append({
                'Parameter': param,
                'Change (%)': change,
                'NPV Impact ($)': impact
            })
    
    sens_df = pd.DataFrame(sensitivity_data)
    
    fig = px.line(
        sens_df,
        x='Change (%)',
        y='NPV Impact ($)',
        color='Parameter',
        title='NPV Sensitivity to Key Parameters',
        markers=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_risk_analysis_tab(results, best_idx):
    """Display risk metrics and scenarios."""
    
    st.markdown("## 🎲 Risk Analysis")
    
    best_npv_paths = results.npv_paths_all[best_idx]
    
    # Risk metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        var_5 = np.percentile(best_npv_paths, 5)
        st.metric(
            "Value at Risk (5%)",
            f"${var_5:,.0f}",
            "5% chance of losing more"
        )
    
    with col2:
        cvar_5 = best_npv_paths[best_npv_paths <= var_5].mean()
        st.metric(
            "Conditional VaR (5%)",
            f"${cvar_5:,.0f}",
            "Expected loss if worst 5% occurs"
        )
    
    with col3:
        max_loss = best_npv_paths.min()
        st.metric(
            "Maximum Loss",
            f"${max_loss:,.0f}",
            "Worst case scenario"
        )
    
    # Risk distribution
    st.markdown("### 📊 Risk Distribution")
    
    # Create risk buckets
    risk_buckets = [
        (best_npv_paths < -results.summary_df.iloc[best_idx]['CAPEX'] * 0.5).sum(),
        ((best_npv_paths >= -results.summary_df.iloc[best_idx]['CAPEX'] * 0.5) & 
         (best_npv_paths < 0)).sum(),
        ((best_npv_paths >= 0) & 
         (best_npv_paths < results.summary_df.iloc[best_idx]['CAPEX'])).sum(),
        (best_npv_paths >= results.summary_df.iloc[best_idx]['CAPEX']).sum()
    ]
    
    risk_labels = [
        'Severe Loss (>50% CAPEX)',
        'Moderate Loss (<50% CAPEX)',
        'Moderate Profit (<100% ROI)',
        'Strong Profit (>100% ROI)'
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=risk_labels,
            y=risk_buckets,
            text=[f'{x/len(best_npv_paths)*100:.1f}%' for x in risk_buckets],
            textposition='auto',
            marker_color=['red', 'orange', 'lightgreen', 'darkgreen']
        )
    ])
    
    fig.update_layout(
        title='Outcome Probability Distribution',
        yaxis_title='Number of Scenarios',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stress test results
    st.markdown("### 🔥 Stress Test Scenarios")
    
    stress_scenarios = [
        {"name": "Base Case", "btc_factor": 1.0, "diff_factor": 1.0},
        {"name": "BTC Crash (-50%)", "btc_factor": 0.5, "diff_factor": 0.8},
        {"name": "Mining Boom", "btc_factor": 1.2, "diff_factor": 2.0},
        {"name": "Perfect Storm", "btc_factor": 0.3, "diff_factor": 0.5}
    ]
    
    stress_results = []
    base_npv = results.summary_df.iloc[best_idx]['Median NPV (USD)']
    
    for scenario in stress_scenarios:
        # Simplified stress calculation
        adjusted_npv = base_npv * scenario["btc_factor"] / scenario["diff_factor"]
        stress_results.append({
            "Scenario": scenario["name"],
            "NPV": adjusted_npv,
            "Impact": (adjusted_npv - base_npv) / base_npv * 100
        })
    
    stress_df = pd.DataFrame(stress_results)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stress_df['Scenario'],
        y=stress_df['NPV'],
        text=[f"${x:,.0f}<br>{i:+.0f}%" for x, i in zip(stress_df['NPV'], stress_df['Impact'])],
        textposition='auto',
        marker_color=['blue', 'red', 'orange', 'darkred']
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title='Stress Test Results',
        yaxis_title='NPV (USD)',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_detailed_report_tab(results, inputs, config):
    """Display comprehensive downloadable report."""
    
    st.markdown("## 📋 Detailed Analysis Report")
    
    # Report timestamp
    st.info(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Full results table
    st.markdown("### 📊 Complete Results Table")
    
    # Format DataFrame for display
    display_df = results.summary_df.copy()
    
    # Format numeric columns with safe handling for None/NaN
    def safe_fmt(fmt):
        return lambda x: fmt.format(x) if pd.notnull(x) else "N/A"
    format_dict = {
        'CAPEX': safe_fmt('${:,.0f}'),
        'Median NPV (USD)': safe_fmt('${:,.0f}'),
        'Mean NPV (USD)': safe_fmt('${:,.0f}'),
        'NPV p5': safe_fmt('${:,.0f}'),
        'NPV p95': safe_fmt('${:,.0f}'),
        'Probability NPV >0 (%)': safe_fmt('{:.1f}%'),
        'Avg Utilization (%)': safe_fmt('{:.1f}%'),
        'Median IRR (%)': safe_fmt('{:.1f}%'),
        'Sharpe Ratio': safe_fmt('{:.2f}')
    }
    st.dataframe(
        display_df.style.format(format_dict),
        use_container_width=True,
        height=400
    )
    
    # Configuration summary
    st.markdown("### ⚙️ Configuration Used")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Financial Parameters:**")
        st.markdown(f"""
        - Discount Rate: {inputs['discount_rate']:.1%}
        - Salvage Value: {inputs['salvage_pct']:.0%}
        - Horizon: {config['simulation']['horizon_years']} years
        - Monte Carlo Paths: {inputs['n_paths']:,}
        """)
    
    with col2:
        st.markdown("**ASIC Specifications:**")
        st.markdown(f"""
        - Model: {inputs['asic'].model}
        - Hashrate: {inputs['asic'].hash_rate_th:.0f} TH/s
        - Power: {inputs['asic'].power_kw:.1f} kW
        - Price: ${inputs['asic'].price_usd:,.0f}
        """)
    
    # Fleet sizing info
    if results.sizing_info:
        st.markdown("### 🎯 Fleet Sizing Details")
        
        with st.expander("View Sizing Process", expanded=False):
            st.text(get_sizing_summary_text(results.sizing_info))
    
    # Download options
    st.markdown("### 💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = results.summary_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Results (CSV)",
            data=csv_data,
            file_name=f"btc_mining_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = results.summary_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📄 Download Results (JSON)",
            data=json_data,
            file_name=f"btc_mining_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        report_text = generate_text_report(results, inputs, config)
        st.download_button(
            label="📝 Download Report (TXT)",
            data=report_text,
            file_name=f"btc_mining_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def create_example_chart():
    """Create example visualization for landing page."""
    
    # Generate example data
    fleet_sizes = np.arange(10, 101, 10)
    npv = -fleet_sizes**2 * 100 + fleet_sizes * 20000 - 500000
    probability = 100 / (1 + np.exp(-0.1 * (fleet_sizes - 50)))
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fleet_sizes,
        y=npv,
        name='Expected NPV',
        line=dict(color='blue', width=3),
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=fleet_sizes,
        y=probability,
        name='Profit Probability (%)',
        line=dict(color='green', width=3, dash='dash'),
        mode='lines+markers',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Example: Fleet Size Optimization',
        xaxis_title='Fleet Size (ASICs)',
        yaxis_title='NPV (USD)',
        yaxis2=dict(
            title='Probability (%)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=400
    )
    
    return fig


def get_sizing_summary_text(sizing_info):
    """Convert sizing info to text format."""
    
    lines = []
    lines.append("FLEET SIZING SUMMARY")
    lines.append("="*50)
    
    if sizing_info.warnings:
        lines.append("\nWARNINGS:")
        for w in sizing_info.warnings:
            lines.append(f"  - {w}")
    
    lines.append(f"\nHYDRO RESOURCE:")
    lines.append(f"  Peak: {sizing_info.hydro_stats['peak_kw']:,.0f} kW")
    lines.append(f"  Zero days: {sizing_info.hydro_stats['zero_days']} ({sizing_info.hydro_stats['zero_days_pct']:.1f}%)")
    
    lines.append(f"\nSIZING PROCESS:")
    lines.append(f"  Initial range: {sizing_info.sizing_process['initial_range'][0]}-{sizing_info.sizing_process['initial_range'][1]} units")
    lines.append(f"  Final candidates: {sizing_info.sizing_process['final_candidates']}")
    
    return "\n".join(lines)


def generate_text_report(results, inputs, config):
    """Generate comprehensive text report."""
    
    best_idx = results.summary_df['Median NPV (USD)'].idxmax()
    best = results.summary_df.iloc[best_idx]
    
    report = []
    report.append("BITCOIN MINING OPTIMIZATION REPORT")
    report.append("="*60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("EXECUTIVE SUMMARY")
    report.append("-"*30)
    report.append(f"Optimal Fleet Size: {int(best['ASICs'])} ASICs")
    report.append(f"Expected NPV: ${best['Median NPV (USD)']:,.0f}")
    report.append(f"Success Probability: {best['Probability NPV >0 (%)']:.1f}%")
    report.append(f"Payback Period: {best['Median Pay-back (days)']} days")
    report.append(f"Total Investment: ${best['CAPEX']:,.0f}")
    report.append("")
    
    report.append("CONFIGURATION")
    report.append("-"*30)
    report.append(f"ASIC Model: {inputs['asic'].model}")
    report.append(f"Discount Rate: {inputs['discount_rate']:.1%}")
    report.append(f"Investment Horizon: {config['simulation']['horizon_years']} years")
    report.append(f"Simulation Quality: {inputs['n_paths']:,} paths")
    report.append("")
    
    report.append("DETAILED RESULTS")
    report.append("-"*30)
    report.append(results.summary_df.to_string())
    
    return "\n".join(report)


# ─────────────────────────────────────────────────────────────
# APPLICATION ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()