import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Ensure the root directory is in the path to import from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_engine import DataEngine
from src.quant_engine import QuantEngine
from src.valuation import MonteCarloValuation

# --- Page Configuration ---
st.set_page_config(
    page_title="Probabilistic Valuation Tool",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Probabilistic Firm Valuation")
st.markdown("""
This tool implements the probabilistic framework described in 
*Bottazzi et al. (2023) - "Uncertainty in firm valuation and a cross-sectional misvaluation measure"*.
Instead of a deterministic DCF, it generates a **probability distribution** of the Fair Value.
""")

# --- Sidebar Configuration ---
st.sidebar.header("Simulation Parameters")
ticker_input = st.sidebar.text_input("Ticker Symbol (e.g., AAPL, NVDA, MSFT)", value="AAPL").upper()
n_sims = st.sidebar.number_input("Monte Carlo Iterations", min_value=1000, max_value=50000, value=10000, step=1000)
horizon = st.sidebar.slider("Projection Horizon (Years)", min_value=5, max_value=20, value=10)

run_button = st.sidebar.button("Run Valuation", type="primary")

# --- Main Logic ---
if run_button:
    if not ticker_input:
        st.warning("Please enter a valid Ticker to proceed.")
    else:
        with st.spinner(f"Fetching data and estimating econometric parameters for {ticker_input}..."):
            try:
                # 1. Data Pipeline
                de = DataEngine(ticker_input)
                clean_df, mkt_data = de.get_cleaned_financial_dataset()
                
                # 2. Quantitative Pipeline
                qe = QuantEngine(clean_df)
                qe.analyze_revenue_growth()
                qe.estimate_fundamental_moments()
                params = qe.get_simulation_inputs()
                
                # Merge market data with stochastic parameters
                params['risk_free_rate'] = mkt_data.get('risk_free_rate', 0.04)
                params['beta'] = mkt_data.get('beta', 1.0)
                
                current_revenue = clean_df['revenue'].iloc[-1]
                market_cap = mkt_data.get('market_cap', 1)
                
                # Proxy for Net Debt (Total Debt - Cash) 
                # Note: For production, extract this explicitly from the balance sheet
                net_debt_proxy = clean_df['total_assets'].iloc[-1] * 0.10 

                # 3. Valuation Engine (Monte Carlo)
                mc = MonteCarloValuation(
                    params=params,
                    current_revenue=current_revenue,
                    net_debt=net_debt_proxy,
                    market_cap=market_cap,
                    n_sims=n_sims,
                    horizon=horizon
                )
                
                mc.run_simulation()
                results = mc.calculate_misvaluation()

                # --- Results Dashboard ---
                st.success("Simulation completed!")
                
                st.subheader("Valuation Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Current Market Cap", f"${results['market_cap'] / 1e9:,.2f} B")
                col2.metric("Mean Intrinsic Value", f"${results['mean_equity_value'] / 1e9:,.2f} B")
                
                # Percentile interpretation
                p_value = results['market_percentile']
                delta_msg = "Discount" if p_value < 50 else "Premium"
                col3.metric("Market Percentile", f"{p_value:.1f}%", 
                            delta=f"{delta_msg} vs Distribution",
                            delta_color="inverse")
                
                # Dynamic Signal Styling
                signal = results['signal']
                signal_emoji = "🟢" if "Undervalued" in signal else "🔴" if "Overvalued" in signal else "🟡"
                col4.metric("Valuation Signal", f"{signal_emoji} {signal}")

                # --- Probability Distribution Plot ---
                st.subheader("Fair Value Probability Distribution (PDF)")
                
                fig, ax = plt.subplots(figsize=(12, 5))
                dist = mc.equity_value_dist
                # Filter outliers for better visualization
                upper_bound = np.percentile(dist, 97.5)
                lower_bound = np.percentile(dist, 0.5)
                filtered_dist = dist[(dist <= upper_bound) & (dist >= lower_bound)]
                
                ax.hist(filtered_dist / 1e9, bins=70, density=True, alpha=0.75, color='#34495e', edgecolor='white')
                
                # Vertical Reference Lines
                mean_b = results['mean_equity_value'] / 1e9
                mkt_b = results['market_cap'] / 1e9
                
                ax.axvline(mean_b, color='#27ae60', linestyle='--', linewidth=2, label=f'Mean Fair Value (${mean_b:,.1f}B)')
                ax.axvline(mkt_b, color='#e74c3c', linestyle='-', linewidth=2, label=f'Market Price (${mkt_b:,.1f}B)')
                
                ax.set_title(f'Monte Carlo Simulation for {ticker_input}', fontsize=14)
                ax.set_xlabel('Equity Value (Billions USD)', fontsize=12)
                ax.set_ylabel('Probability Density', fontsize=12)
                ax.legend()
                ax.grid(axis='y', alpha=0.2)
                
                st.pyplot(fig)
                
                # --- Parameter Breakdown ---
                with st.expander("View Estimated Stochastic Parameters"):
                    st.write("These parameters were derived from historical financial statements and serve as the 'drift' and 'diffusion' for the simulation.")
                    st.json(params)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("Tip: Ensure the ticker is correct and the company has at least 3-4 years of public financial data.")
