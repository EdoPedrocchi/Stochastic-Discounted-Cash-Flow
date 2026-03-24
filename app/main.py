import sys
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Ensure the root directory is in the path to import from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.startup_model import StartupTrajectoryModel
from src.cap_table_engine import CapTableEngine
from src.exit_engine import ExitEngine

# --- Page Configuration ---
st.set_page_config(
    page_title="Venture Stochastic Exit (VSE) Tool",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Venture Stochastic Exit (VSE) Framework")
st.markdown("""
This tool simulates the **Power Law** of Venture Capital. It models stochastic revenue growth, 
monthly cash burn (survival), future dilution rounds, and exit multiples to estimate your **MOIC** (Multiple on Invested Capital).
""")

# --- Sidebar: Input Parameters ---
st.sidebar.header("1. Current Financials")
curr_arr = st.sidebar.number_input("Current ARR ($)", value=2_000_000, step=500_000)
curr_cash = st.sidebar.number_input("Cash in Bank ($)", value=5_000_000, step=500_000)
monthly_burn = st.sidebar.number_input("Monthly Net Burn ($)", value=200_000, step=50_000)

st.sidebar.header("2. Investment & Stake")
initial_inv = st.sidebar.number_input("Your Initial Investment ($)", value=1_000_000, step=100_000)
curr_ownership = st.sidebar.slider("Current Ownership (%)", 1.0, 50.0, 10.0) / 100.0

st.sidebar.header("3. Market & Growth")
sector_choice = st.sidebar.selectbox("Sector (from JSON)", ["SaaS", "Fintech", "Biotech", "AI_ML", "DeepTech"])
growth_mu = st.sidebar.slider("Expected Annual Growth (Mean %)", 10, 200, 60) / 100.0
growth_sigma = st.sidebar.slider("Growth Volatility (%)", 5, 100, 30) / 100.0
exit_horizon = st.sidebar.slider("Exit Horizon (Years)", 3, 10, 5)

n_sims = 10000

# --- Simulation Execution ---
# Start of the main Conditional Block
if st.sidebar.button("Run Stochastic Simulation", type="primary"):
    with st.spinner("Simulating 10,000 startup lifecycles..."):
        # 1. Operational Simulation (Revenue & Survival)
        startup = StartupTrajectoryModel(
            initial_revenue=curr_arr,
            initial_cash=curr_cash,
            monthly_burn=monthly_burn,
            growth_mu=growth_mu,
            growth_sigma=growth_sigma
        )
        op_results = startup.simulate_path(years=exit_horizon, n_sims=n_sims)
        
        # 2. Dilution Simulation (Cap Table)
        cap_table = CapTableEngine(initial_ownership=curr_ownership)
        final_stakes = cap_table.simulate_dilution(
            revenue_paths=op_results["full_revenue_paths"],
            n_sims=n_sims,
            years=exit_horizon
        )
        
        # 3. Exit Simulation (Multiples & Returns)
        # Ensure 'data/sector_multiples.json' exists in your repo
        exit_eng = ExitEngine(sector=sector_choice, json_path="data/sector_multiples.json")
        final_results = exit_eng.run_exit_analysis(
            revenue_at_exit=op_results["revenue_at_exit"],
            survival_mask=op_results["survival_mask"],
            final_ownership=final_stakes,
            initial_investment=initial_inv
        )
        
        # --- UI: Metrics & Visualization ---
        stats = final_results["stats"]
        
        st.subheader("Venture Return Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected MOIC (Mean)", f"{stats['mean_moic']:.2f}x")
        m2.metric("Probability of Total Loss", f"{stats['probability_of_loss']:.1%}")
        m3.metric("Probability of 10x+ Exit", f"{stats['probability_of_homerun']:.1%}")
        m4.metric("Survival Rate", f"{np.mean(op_results['survival_mask']):.1%}")

        st.divider()

        # --- Plots ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.write("### MOIC Probability Distribution")
            fig_moic, ax_moic = plt.subplots()
            moic_data = final_results["moic_dist"]
            # Showing only cases where MOIC > 0.1 to focus on success tail
            success_moic = moic_data[moic_data > 0.1]
            
            if len(success_moic) > 0:
                ax_moic.hist(success_moic, bins=50, color="#2ecc71", alpha=0.7, edgecolor='white')
                ax_moic.set_xlabel("Multiple on Invested Capital (MOIC)")
                ax_moic.set_ylabel("Frequency")
                ax_moic.set_title("Distribution of Non-Zero Outcomes")
                st.pyplot(fig_moic)
            else:
                st.warning("No successful exits in this simulation. Survival rate was too low.")

        with col_right:
            st.write("### Survival vs. Failure")
            fig_pie, ax_pie = plt.subplots()
            survived_count = np.sum(op_results['survival_mask'])
            failed_count = n_sims - survived_count
            ax_pie.pie([survived_count, failed_count], labels=["Survived", "Failed"], 
                       autopct='%1.1f%%', colors=["#3498db", "#e74c3c"], startangle=90)
            ax_pie.axis('equal') 
            st.pyplot(fig_pie)

        with st.expander("Technical Log"):
            st.write(f"Average Revenue at Exit (Survivors only): ${stats['expected_exit_value_mean']:,.2f}")
            st.write(f"Mean Diluted Ownership at Exit: {np.mean(final_stakes):.2%}")
            st.info("The simulation uses a Log-Normal distribution for exit multiples.")

# This 'else' is now properly aligned with the 'if' on line 47
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Stochastic Simulation' to see the Venture outcome distribution.")
