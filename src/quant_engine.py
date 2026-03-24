import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Any, Optional

class QuantEngine:
    """
    Motore quantitativo per la modellazione stocastica dei parametri aziendali.
    Trasforma i dati storici in parametri di distribuzione per la simulazione.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: DataFrame pulito proveniente da DataEngine.get_cleaned_financial_dataset()
        """
        self.data = data
        self.params: Dict[str, Any] = {}

    def analyze_revenue_growth(self) -> Dict[str, float]:
        """
        Stima la dinamica dei ricavi usando i log-returns: g_t = ln(R_t / R_{t-1}).
        Esegue il test di stazionarietà Dickey-Fuller.
        """
        # Calcolo log-growth
        rev_series = self.data['revenue']
        log_growth = np.log(rev_series / rev_series.shift(1)).dropna()

        if len(log_growth) < 2:
            raise ValueError("Dati storici insufficienti per calcolare la crescita (servono almeno 3 anni).")

        mu_g = log_growth.mean()
        sigma_g = log_growth.std()

        # Test di stazionarietà (Augmented Dickey-Fuller)
        # Se p-value < 0.05, rifiutiamo H0: la serie è stazionaria.
        adf_stat, p_value, *_ = adfuller(log_growth)
        
        self.params.update({
            "mu_rev": mu_g,
            "sigma_rev": sigma_g,
            "revenue_stationary": p_value < 0.05
        })

        return {
            "mu_rev": mu_g,
            "sigma_rev": sigma_g,
            "p_value_adf": p_value
        }

    def estimate_fundamental_moments(self) -> pd.DataFrame:
        """
        Calcola i parametri stocastici per i margini e l'efficienza del capitale.
        Fornisce la base per campionare EBIT Margin e Sales-to-Capital nella Week 2.
        """
        # 1. EBIT Margin (EBIT / Revenues)
        ebit_margin = (self.data['ebit'] / self.data['revenue']).dropna()
        
        # 2. Sales-to-Capital Ratio (Revenues / Invested Capital)
        # Nota: L'invested capital è già calcolato nel DataEngine
        sales_to_cap = (self.data['revenue'] / self.data['invested_capital']).replace([np.inf, -np.inf], np.nan).dropna()
        
        # 3. Effective Tax Rate (Tax Provision / Pretax Income)
        tax_rate = (self.data['tax_provision'] / self.data['pretax_income']).replace([np.inf, -np.inf], np.nan).fillna(0.21)
        # Clipping per evitare tax rate assurdi (es. rimborsi fiscali temporanei)
        tax_rate = tax_rate.clip(0, 0.45)

        self.params.update({
            "avg_ebit_margin": ebit_margin.mean(),
            "std_ebit_margin": ebit_margin.std(),
            "avg_sales_to_cap": sales_to_cap.mean(),
            "std_sales_to_cap": sales_to_cap.std(),
            "avg_tax_rate": tax_rate.mean()
        })

        return pd.DataFrame({
            "Parameter": ["EBIT Margin", "Sales-to-Capital", "Tax Rate"],
            "Mean": [ebit_margin.mean(), sales_to_cap.mean(), tax_rate.mean()],
            "StdDev": [ebit_margin.std(), sales_to_cap.std(), tax_rate.std()]
        })

    def get_simulation_inputs(self) -> Dict[str, Any]:
        """Restituisce il dizionario completo dei parametri per la fase di simulazione."""
        if not self.params:
            self.analyze_revenue_growth()
            self.estimate_fundamental_moments()
        return self.params

# --- Test Autonomo ---
if __name__ == "__main__":
    from data_engine import DataEngine
    
    print("Inizializzazione test quantitativo...")
    de = DataEngine("MSFT")
    clean_df, mkt = de.get_cleaned_financial_dataset()
    
    qe = QuantEngine(clean_df)
    rev_stats = qe.analyze_revenue_growth()
    moments = qe.estimate_fundamental_moments()
    
    print("\n--- Revenue Growth Stats ---")
    print(f"Mean Growth (log): {rev_stats['mu_rev']:.2%}")
    print(f"Volatility: {rev_stats['sigma_rev']:.2%}")
    print(f"P-Value ADF Test: {rev_stats['p_value_adf']:.4f}")
    
    print("\n--- Fundamental Moments ---")
    print(moments)
