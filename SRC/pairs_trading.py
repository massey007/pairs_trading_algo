import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from statistics import correlation
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
import seaborn as sns
from sklearn.linear_model import LinearRegression
from fredapi import Fred
fred = Fred(api_key='17a30f1e933d9c9de2afc4bcd7a5fbcb')
import requests

from SRC.data_loader import DataLoader

import warnings
warnings.filterwarnings("ignore")

class PairsBacktest:

    def __init__(
        self, years, tickers, market_ticker = '^GSPC', lookback_window=30, entry_z=2.0
        ):

        """
        Initializes the backtester with historical price dataframes.

        """

        self.tickers = tickers
        self.years = years
        self.market_ticker = market_ticker
        self.window = lookback_window
        self.entry_z = entry_z

#########################################################################################################################

    def generate_signals(self):

        """

        Calculates the mathematical spread, rolling metrics, and Z-Scores.

        """

        loader = DataLoader(self.years)
        loader.market_data(self.years, self.tickers)
        data = loader.data
        self.data = data

        # 1. Calculate beta using Linear regression (regress Log_A on Log_B)
        X = self.data["Log_B"].values.reshape(-1, 1)
        y = self.data["Log_A"].values.reshape(-1, 1)

        lr = LinearRegression()
        lr.fit(X, y)
        self.beta = lr.coef_[0][0]

        # 2. Calculate spreads - spread = lnPa - beta*lnPb
        self.data["Spread"] = self.data["Log_A"] - (self.beta * self.data["Log_B"])

        # 3. Calculate rolling statistics
        self.data["Mean"] = (self.data["Spread"].rolling(window=self.window).mean())
        self.data["Std"] = self.data["Spread"].rolling(window=self.window).std()

        # 4. Calculate the mathematical Z-Score
        self.data["Z_Score"] = (self.data["Spread"] - self.data["Mean"]) / self.data["Std"]

        # 5. Initialize Position States (1 = Long, -1 = Short, 0 = Flat)
        self.data["Pos_A"] = 0.0
        self.data["Pos_B"] = 0.0

        # 6. Loop or vectorized logic to apply threshold triggers
        self.data.loc[self.data["Z_Score"] > self.entry_z, "Pos_A"] = -1.0
        self.data.loc[self.data["Z_Score"] > self.entry_z, "Pos_B"] = 1.0

        self.data.loc[self.data["Z_Score"] < -self.entry_z, "Pos_A"] = 1.0
        self.data.loc[self.data["Z_Score"] < -self.entry_z, "Pos_B"] = -1.0

        # 7. Shift positions by 1 day to strictly prevent look-ahead bias
        self.data["Active_Pos_A"] = self.data["Pos_A"].shift(1).fillna(0)
        self.data["Active_Pos_B"] = self.data["Pos_B"].shift(1).fillna(0)

#########################################################################################################################
    def run_simulation(self, initial_capital=1_00.00, borrow_fee_annual=0.02, transaction_cost_per_trade = 0.05):

        self.generate_signals()
        self.initial_capital = initial_capital # Store initial_capital as an instance variable

        """

        Simulates trading execution and tracks mathematical portfolio returns.

        """

        # Calculate asset returns
        self.data["Ret_A"] = self.data["Price_A"].pct_change()
        self.data["Ret_B"] = self.data["Price_B"].pct_change()

        # Calculate directional strategy returns per leg
        # (Remember: short returns are the inverse of price changes)
        self.data["Strat_Ret_A"] = self.data["Ret_A"] * self.data["Active_Pos_A"]
        self.data["Strat_Ret_B"] = self.data["Ret_B"] * self.data["Active_Pos_B"]

        # Subtract trading costs and borrow fees
        self.data["Strat_Ret_A"] -= (
            self.data["Active_Pos_A"] * borrow_fee_annual / 252
            )
        # Deduct a percentage from Strat_Ret when Active_Pos changes (slippage/commissions)
        self.data["Strat_Ret_A"] -= (
            self.data["Active_Pos_A"].diff().fillna(0) * transaction_cost_per_trade / 252
           )
        # Deduct (borrow_fee_annual / 252) daily whenever a position is short (-1)
        self.data["Strat_Ret_B"] -= (
            self.data["Active_Pos_B"].diff().fillna(0) * borrow_fee_annual / 252
           )
        # Deduct a percentage from Strat_Ret when Active_Pos changes (slippage/commissions)
        self.data["Strat_Ret_B"] -= (
            self.data["Active_Pos_B"].diff().fillna(0) * transaction_cost_per_trade / 252
           )

        # Combine returns assuming a beta capital split between the legs (summing individual leg returns)
        self.data["Total_Strat_Return"] = (
            self.data["Strat_Ret_A"] + self.data["Strat_Ret_B"]
        )

        # Calculate final equity curve
        self.data["Equity"] = initial_capital * (
            1 + self.data["Total_Strat_Return"]
        ).cumprod().fillna(1)

##########################################################################################################################
    def compute_metrics(self, risk_free = 'DFF'):

        self.run_simulation()
        loader = DataLoader(self.years)
        loader.market_data(self.years, self.tickers)
        self.market_data = loader.market_data

        """

        Computes key performance metrics for strategy health evaluation.

        """

        risk_free_series = pd.Series([], dtype='float64') # Initialize as empty Series
        try:
          risk_free_df = fred.get_series(risk_free, self.start_date, self.end_date).to_frame(name='Close')
          if not risk_free_df.empty and 'Close' in risk_free_df.columns:
              risk_free_series = risk_free_df['Close'].squeeze() # Added .squeeze()
          else:
              print(f"Warning: Risk-free data for {risk_free} is empty or 'Close' column missing.")
        except KeyError as error:
          print(f"yfinance has returned for risk-free data: {error}")
        except Exception as e:
          print(f"An unexpected error occurred loading risk-free data: {e}")

        # Ensure market_data is not empty before calculating market_return_series
        market_return_series = pd.Series([], dtype='float64') # Initialize as empty Series
        if not self.market_data.empty and 'Close' in self.market_data.columns:
            close_market_series = self.market_data["Close"].squeeze() # Added .squeeze()
            market_return_series = np.log(close_market_series / close_market_series.shift(1))
        else:
            print("Warning: Market data is empty or 'Close' column missing for market return calculation.")

        # Reindex all series to the main self.data index before combining
        # This will fill NaNs for dates not present in all series
        strategy_return_aligned = self.data["Total_Strat_Return"]
        market_return_aligned = market_return_series.reindex(strategy_return_aligned.index)
        risk_free_aligned = risk_free_series.reindex(strategy_return_aligned.index)

        # Create a single DataFrame from the reindexed series, ensuring 1D arrays
        temp_df = pd.DataFrame({
            'strategy_return': strategy_return_aligned.to_numpy().flatten(),
            'market_return': market_return_aligned.to_numpy().flatten(),
            'risk_free_rate': risk_free_aligned.to_numpy().flatten()
        })

        # Drop any rows where any of the series has a NaN value
        aligned_data = temp_df.dropna()

        # Check if aligned_data is empty after dropping NaNs
        if aligned_data.empty:
            return {"Error": "Insufficient aligned data for metric calculation after dropping NaNs."}

        total_strat_return_aligned = aligned_data['strategy_return']
        market_return_for_regression = aligned_data['market_return'] # Rename for clarity
        risk_free_for_regression = aligned_data['risk_free_rate'] # Rename for clarity

        # Convert risk-free rate from annual percentage points to daily decimal
        # Assuming risk_free_for_regression values are annual percentage points (e.g., 3.628 for 3.628%)
        daily_risk_free_rate_for_metrics = (risk_free_for_regression / 100) / 252

        # Regression to get alpha
        X = (market_return_for_regression - daily_risk_free_rate_for_metrics).values.reshape(-1, 1)
        y = total_strat_return_aligned.values.reshape(-1, 1)

        # Check if X or y are empty after alignment, which would cause issues with lr.fit
        if X.size == 0 or y.size == 0:
            return {"Error": "No valid data points for regression after alignment."}

        lr = LinearRegression()
        lr.fit(X, y)

        # Alpha according to the CAPM
        daily_alpha = lr.intercept_[0] + daily_risk_free_rate_for_metrics.mean()
        alpha = daily_alpha * 252 # Annualize alpha

        # Calculate total return
        if not self.data["Equity"].empty:
            total_return = (self.data["Equity"].iloc[-1] / self.data["Equity"].iloc[0]) - 1
        else:
            total_return = 0.0 # Or handle as appropriate

        # Maximum Drawdown calculation
        # Ensure that total_strat_return_aligned is not empty and has variance for std() calculation
        sharpe_ratio = 0.0 # Default value
        if not total_strat_return_aligned.empty and total_strat_return_aligned.std() != 0:
            daily_sharpe_ratio = (total_strat_return_aligned.mean() - daily_risk_free_rate_for_metrics.mean() )/ total_strat_return_aligned.std()
            sharpe_ratio = daily_sharpe_ratio * np.sqrt(252) # Annualize Sharpe Ratio

        peak = self.data["Equity"].cummax()
        drawdown = (self.data["Equity"] - peak) / peak
        max_dd = drawdown.min()

        metrics = {
            "Sharpe Ratio"    : f"{sharpe_ratio:.5f}",
            "Alpha"           : f"{alpha * 100:.5f}%",
            "Total Return"    : f"{total_return:.2%}",
            "Max Drawdown"    : f"{max_dd:.2%}",
            "Start Date"      : self.start_date.strftime("%Y-%m-%d"),
            "End Date"        : self.end_date.strftime("%Y-%m-%d"),
            "Starting Capital": f"${self.initial_capital:.2f}",
            "Ending Capital"  : f"${self.data['Equity'].iloc[-1]:.2f}",
            "Capital Gained"  : f"${self.data['Equity'].iloc[-1] - self.initial_capital:.2f}",
            "Total Trades"    : len(self.data[self.data["Active_Pos_A"] != 0]),
                  }

        # Put matrics in a dataframe
        metrics1 = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        
        return metrics1