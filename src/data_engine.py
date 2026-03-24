import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


class DataEngine:
    """
    Responsabile del fetching, della pulizia e dell'allineamento dei dati finanziari
    tramite l'API di Yahoo Finance (yfinance).
    """

    def __init__(self, ticker: str):
        self.ticker_symbol = ticker.upper()
        self.ticker = yf.Ticker(self.ticker_symbol)
        
    def fetch_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Scarica i dati finanziari grezzi (Income Statement, Balance Sheet) e i dati di mercato.
        
        Returns:
            Tuple contenente (financials, balance_sheet, market_info)
        """
        try:
            # Annual Financials (yfinance traspone le date sulle colonne, noi le vogliamo come indice)
            financials = self.ticker.financials.T
            balance_sheet = self.ticker.balance_sheet.T
            
            if financials.empty or balance_sheet.empty:
                raise ValueError(f"Dati di bilancio non disponibili per il ticker {self.ticker_symbol}")

            # Dati di mercato e info generali
            info = self.ticker.info
            market_info = {
                "current_price": info.get("currentPrice"),
                "market_cap": info.get("marketCap"),
                "beta": info.get("beta", 1.0),  # Default a 1.0 se non presente
                "industry": info.get("industry", "Unknown"),
                "currency": info.get("currency", "USD")
            }

            # Aggiungiamo il tasso Risk-Free (Treasury 10Y)
            market_info["risk_free_rate"] = self._get_risk_free_rate()

            return financials, balance_sheet, market_info

        except Exception as e:
            raise RuntimeError(f"Errore critico nel recupero dati per {self.ticker_symbol}: {e}")

    def get_cleaned_financial_dataset(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Estrae i dati grezzi, li pulisce e calcola le metriche fondamentali grezze 
        allineate temporalmente.
        
        Returns:
            Dataframe allineato e dizionario di mercato.
        """
        raw_fin, raw_bs, market_info = self.fetch_raw_data()

        # Uniamo i due statement basandoci sulla data di chiusura dell'anno fiscale
        merged_df = pd.concat([raw_fin, raw_bs], axis=1)
        
        # Eliminiamo eventuali duplicati di colonne derivanti dalla concatenazione
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        
        # Ordiniamo dal passato al presente
        merged_df = merged_df.sort_index(ascending=True)

        # Mappatura standard dei dati richiesti per il modello di Bottazzi et al.
        required_columns = {
            'Total Revenue': 'revenue',
            'EBIT': 'ebit',
            'Tax Provision': 'tax_provision',
            'Pretax Income': 'pretax_income',
            'Total Assets': 'total_assets',
            'Cash And Cash Equivalents': 'cash_and_equiv'
        }

        cleaned_data = pd.DataFrame(index=merged_df.index)

        for yf_col, standard_col in required_columns.items():
            if yf_col in merged_df.columns:
                cleaned_data[standard_col] = pd.to_numeric(merged_df[yf_col], errors='coerce')
            else:
                cleaned_data[standard_col] = np.nan

        # Gestione dei valori mancanti (Backfill/Forwardfill leggero, poi drop)
        cleaned_data = cleaned_data.ffill().bfill().dropna(subset=['revenue'])

        # Calcolo Invested Capital Grezzo per il Sales-to-Capital Ratio futuro
        cleaned_data['invested_capital'] = (
            cleaned_data['total_assets'] - cleaned_data['cash_and_equiv'].fillna(0)
        )

        return cleaned_data, market_info

    def _get_risk_free_rate(self) -> float:
        """
        Recupera il rendimento corrente del US Treasury a 10 anni (^TNX).
        """
        try:
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1] / 100)
        except Exception:
            pass
        return 0.042  # Fallback conservativo al 4.2% se Yahoo Finance fallisce


# --- Test Autonomo (permette di lanciare il file direttamente) ---
if __name__ == "__main__":
    print("Testando il DataEngine...")
    engine = DataEngine("AAPL")
    
    df_clean, mkt_data = engine.get_cleaned_financial_dataset()
    
    print("\n✅ Dati di Mercato:")
    for k, v in mkt_data.items():
        print(f"  - {k}: {v}")
        
    print("\n✅ DataFrame Storico Pulito (Ultime 3 righe):")
    print(df_clean[['revenue', 'ebit', 'invested_capital']].tail(3))
