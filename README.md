# Stochastic-Discounted-Cash-Flow

This `README.md` is designed to provide a professional, technical overview of the project. it targets both users and developers, explaining the econometric rationale and the software architecture.

---

# Probabilistic Firm Valuation Framework
### Based on Bottazzi et al. (2023)

## Overview
This repository implements a probabilistic asset pricing framework designed to overcome the limitations of deterministic Discounted Cash Flow (DCF) models. Instead of relying on single-point estimates for growth and margins, this tool treats fundamental financial drivers as stochastic variables. 

By utilizing Monte Carlo simulations, the model generates a probability distribution of a firm's Intrinsic Value, allowing for a quantitative measure of "Misvaluation" based on where the current market price sits within that distribution (e.g., as a Z-score or percentile).

## Core Methodology
The valuation engine follows a three-stage econometric process:

1.  **Revenue Dynamics**: Revenue growth is modeled as a stochastic process. The framework estimates mean growth ($\mu$) and volatility ($\sigma$) using historical log-returns: $g_t = \ln(R_t / R_{t-1})$.
2.  **Fundamental Moments**: The model extracts the historical mean and standard deviation for key operational efficiency ratios:
    * **EBIT Margin**: Operating profitability.
    * **Sales-to-Capital Ratio**: Capital efficiency and reinvestment requirements.
    * **Effective Tax Rate**: Normalized tax leakage.
3.  **Monte Carlo Simulation**: The system projects 10,000+ independent paths for the next 10 years. Each path calculates NOPAT and Reinvestment (derived from the Sales-to-Capital ratio) to arrive at the Free Cash Flow to the Firm (FCFF).

## Project Structure
```text
PROB-VALUATION-TOOL/
├── app/
│   └── main.py          # Streamlit Web Interface (Entry Point)
├── src/
│   ├── data_engine.py    # Automated fetching via yfinance and data cleaning
│   ├── quant_engine.py   # Econometric modeling and parameter estimation
│   ├── valuation.py      # Monte Carlo engine and DCF logic
│   └── utils.py          # Mathematical helpers
├── tests/                # Unit tests for financial logic validation
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Mathematical Implementation
The revenue projection uses a Geometric Brownian Motion (GBM) approach in discrete time:

$$R_{t} = R_{t-1} \cdot e^{(\mu - 0.5\sigma^2) + \sigma \epsilon}$$

Where:
* $\mu$ and $\sigma$ are derived from the historical log-growth of the ticker.
* $\epsilon \sim N(0,1)$ represents the random shock for each specific simulation year.

The Terminal Value (TV) is calculated using the Gordon Growth Model, with a safety cap ensuring the perpetual growth rate ($g$) does not exceed the Risk-Free Rate or the Cost of Capital (WACC).

## Installation and Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/prob-valuation-tool.git
cd prob-valuation-tool
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app/main.py
```

## Technical Requirements
* **Python 3.9+**
* **yfinance**: For real-time market and fundamental data.
* **statsmodels**: For Dickey-Fuller stationarity testing.
* **numpy & scipy**: For vectorized Monte Carlo simulations.
* **Streamlit**: For the analytical dashboard.

## Disclaimer
This tool is for educational and research purposes only. It does not constitute financial advice. Valuation outputs are highly dependent on historical data quality and the assumptions used in the simulation.
