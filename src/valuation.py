import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, Tuple

class MonteCarloValuation:
    """
    Motore di simulazione Monte Carlo per la valutazione probabilistica (DCF Stocastico).
    Implementa le dinamiche di Bottazzi et al. (2023).
    """

    def __init__(self, params: Dict[str, Any], current_revenue: float, net_debt: float, market_cap: float, n_sims: int = 10000, horizon: int = 10):
        """
        Inizializza il motore di valutazione.
        
        Args:
            params: Dizionario dei parametri (output di quant_engine.py).
            current_revenue: Ricavi dell'ultimo anno solare (T=0).
            net_debt: Debito Netto (Total Debt - Cash). Necessario per passare da Firm Value a Equity Value.
            market_cap: Capitalizzazione di mercato attuale (per il calcolo della Misvaluation).
            n_sims: Numero di traiettorie Monte Carlo.
            horizon: Orizzonte temporale di proiezione esplicita (anni).
        """
        self.params = params
        self.current_revenue = current_revenue
        self.net_debt = net_debt
        self.market_cap = market_cap
        self.n_sims = n_sims
        self.horizon = horizon
        
        # Risk & Return assumptions
        self.risk_free_rate = params.get("risk_free_rate", 0.04)
        self.beta = params.get("beta", 1.0)
        self.erp = 0.055  # Equity Risk Premium di default (5.5%)
        
        # Calcolo Costo dell'Equity (CAPM). Per semplicità, in questo modello base WACC = Ke.
        self.wacc = self.risk_free_rate + (self.beta * self.erp)
        
        # Sanity Check 1: Tasso di crescita terminale (g) prudenziale e strettamente minore del WACC
        self.terminal_g = min(self.risk_free_rate, self.wacc - 0.01)

    def run_simulation(self) -> np.ndarray:
        """
        Esegue la simulazione vettorizzata dei flussi di cassa e calcola il Fair Value.
        """
        # 1. Stochastic Revenue Path (Geometric Brownian Motion log-space)
        mu = self.params.get("mu_rev", 0.05)
        sigma = self.params.get("sigma_rev", 0.15)
        
        # Matrice di shock casuali normali (n_sims, horizon)
        Z = np.random.standard_normal((self.n_sims, self.horizon))
        
        # Calcolo dei fattori di crescita: exp((mu - 0.5 * sigma^2) + sigma * Z)
        drift = mu - 0.5 * (sigma ** 2)
        growth_factors = np.exp(drift + sigma * Z)
        
        # Traiettorie cumulative dei ricavi
        # np.cumprod moltiplica i fattori di crescita in sequenza temporale (asse 1)
        revenue_paths = self.current_revenue * np.cumprod(growth_factors, axis=1)
        
        # Per calcolare la variazione dei ricavi (Delta Revenue), ci serve la matrice traslata
        rev_t_minus_1 = np.c_[np.full((self.n_sims, 1), self.current_revenue), revenue_paths[:, :-1]]
        delta_revenue = revenue_paths - rev_t_minus_1
        
        # 2. Campionamento stocastico dei parametri fondamentali
        # Margini EBIT e Sales-to-Capital variano per ogni scenario e ogni anno
        ebit_margins = np.random.normal(
            self.params.get("avg_ebit_margin", 0.15), 
            self.params.get("std_ebit_margin", 0.03), 
            (self.n_sims, self.horizon)
        )
        # Sanity check: limitiamo i margini tra -50% e 100%
        ebit_margins = np.clip(ebit_margins, -0.5, 1.0)
        
        sales_to_cap = np.random.normal(
            self.params.get("avg_sales_to_cap", 2.0),
            self.params.get("std_sales_to_cap", 0.5),
            (self.n_sims, self.horizon)
        )
        # Sanity check: un S2C negativo o vicino allo zero causa esplosioni matematiche.
        sales_to_cap = np.clip(sales_to_cap, 0.1, 10.0)
        
        tax_rate = self.params.get("avg_tax_rate", 0.21)

        # 3. Calcolo Free Cash Flow to the Firm (FCFF)
        ebit = revenue_paths * ebit_margins
        nopat = ebit * (1 - tax_rate)
        
        # Reinvestment = Delta Revenue / Sales-to-Capital
        reinvestment = delta_revenue / sales_to_cap
        
        # Sanity Check 2: Se i ricavi scendono, il reinvestimento sarebbe negativo (rilascio di capitale).
        # Per prudenza, limitiamo il recupero massiccio di capitale a un massimo del 5% dei ricavi.
        reinvestment = np.maximum(reinvestment, -0.05 * revenue_paths)
        
        fcff = nopat - reinvestment

        # 4. Attualizzazione (Discounting)
        # Array di fattori di sconto: 1 / (1 + WACC)^t
        years = np.arange(1, self.horizon + 1)
        discount_factors = 1 / ((1 + self.wacc) ** years)
        
        pv_fcff = np.sum(fcff * discount_factors, axis=1)
        
        # Terminal Value: Gordon Growth Model sull'ultimo FCFF proiettato
        # TV = FCFF_T * (1 + g) / (WACC - g)
        terminal_value = (fcff[:, -1] * (1 + self.terminal_g)) / (self.wacc - self.terminal_g)
        pv_tv = terminal_value / ((1 + self.wacc) ** self.horizon)
        
        # Enterprise Value (Firm Value)
        enterprise_value = pv_fcff + pv_tv
        
        # Equity Value = Enterprise Value - Net Debt
        equity_value = enterprise_value - self.net_debt
        
        # Sanity Check 3: L'Equity Value non può essere negativo (opzione di default degli azionisti)
        equity_value = np.maximum(equity_value, 0)
        
        self.equity_value_dist = equity_value
        return equity_value

    def calculate_misvaluation(self) -> Dict[str, Any]:
        """
        Calcola le metriche di misvaluation posizionando il prezzo di mercato 
        sulla distribuzione generata.
        """
        if not hasattr(self, 'equity_value_dist'):
            self.run_simulation()
            
        mean_val = np.mean(self.equity_value_dist)
        median_val = np.median(self.equity_value_dist)
        std_val = np.std(self.equity_value_dist)
        
        # Percentile del Market Cap attuale rispetto alla distribuzione generata
        percentile = stats.percentileofscore(self.equity_value_dist, self.market_cap)
        
        # Segnale di Misvaluation
        if percentile < 20:
            signal = "Strongly Undervalued"
        elif percentile < 40:
            signal = "Undervalued"
        elif percentile < 60:
            signal = "Fairly Valued"
        elif percentile < 80:
            signal = "Overvalued"
        else:
            signal = "Strongly Overvalued"
            
        return {
            "mean_equity_value": mean_val,
            "median_equity_value": median_val,
            "std_dev": std_val,
            "market_cap": self.market_cap,
            "market_percentile": percentile,
            "signal": signal
        }

    def plot_distribution(self, ticker: str = "Company"):
        """Genera il grafico della Probability Density Function (PDF) del Fair Value."""
        if not hasattr(self, 'equity_value_dist'):
            self.run_simulation()
            
        plt.figure(figsize=(10, 6))
        
        # Istogramma della distribuzione (troncato al 95° percentile per leggibilità)
        cap_val = np.percentile(self.equity_value_dist, 95)
        filtered_dist = self.equity_value_dist[self.equity_value_dist <= cap_val]
        
        plt.hist(filtered_dist, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
        
        # Linee di riferimento
        plt.axvline(np.mean(filtered_dist), color='green', linestyle='dashed', linewidth=2, label=f'Mean Intrinsic Value')
        plt.axvline(self.market_cap, color='red', linestyle='solid', linewidth=2, label=f'Current Market Cap')
        
        plt.title(f'{ticker} - Fair Value Probability Distribution')
        plt.xlabel('Equity Value (Billions)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

# --- Test Autonomo ---
if __name__ == "__main__":
    # Parametri fittizi (Normalmente iniettati da quant_engine.py)
    mock_params = {
        "mu_rev": 0.08,           # Crescita media 8%
        "sigma_rev": 0.12,        # Volatilità ricavi 12%
        "avg_ebit_margin": 0.25,  # Margine medio 25%
        "std_ebit_margin": 0.02,  # Volatilità margine 2%
        "avg_sales_to_cap": 1.5,
        "std_sales_to_cap": 0.2,
        "avg_tax_rate": 0.21,
        "risk_free_rate": 0.042,
        "beta": 1.1
    }
    
    # Inizializzazione motore per una simulazione test
    mc_model = MonteCarloValuation(
        params=mock_params,
        current_revenue=380e9,  # Es. 380 Miliardi (Proxy Apple/Microsoft)
        net_debt=50e9,          # 50 Miliardi di debito netto
        market_cap=2500e9,      # 2.5 Trilioni di Market Cap
        n_sims=10000
    )
    
    print("Avvio Simulazione Monte Carlo (10,000 traiettorie)...")
    dist = mc_model.run_simulation()
    
    print("\n--- Analisi Misvaluation ---")
    results = mc_model.calculate_misvaluation()
    print(f"Valore Intrinseco Medio: ${results['mean_equity_value'] / 1e9:,.2f} B")
    print(f"Market Cap Attuale: ${results['market_cap'] / 1e9:,.2f} B")
    print(f"Percentile di Mercato: {results['market_percentile']:.1f}°")
    print(f"Segnale: {results['signal']}")
    
    # mc_model.plot_distribution(ticker="TEST-TICKER") # Decommentare per vedere il grafico
