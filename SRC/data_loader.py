"""
This module contains the DataLoader class, which is responsible for loading and processing financial 
data from various sources, including Yahoo Finance and the Federal Reserve Economic Data (FRED) API. 
The class provides methods to retrieve historical yield data, market data for specified tickers, 
and perform basic data analysis

"""

import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from dateutil.relativedelta import relativedelta
import pandas_datareader as pdr
from scipy.cluster.hierarchy import linkage, dendrogram
from fredapi import Fred
fred = Fred(api_key='17a30f1e933d9c9de2afc4bcd7a5fbcb')


class DataLoader:
           
    def __init__(self, years):

        self.years = years

#########################################################################################################################

    def _build(self):

        self.yields = None
        self.SERIES = {
        'Repo': 'DFF',
        'One_Month_US':'DGS1MO',
        'Three_Month_US': 'DGS3MO',
        'Six_Month_US': 'DGS6MO',
        'One_Year_US':'DGS1',
        'Two_Year_US':'DGS2',
        'Three_year_US': 'DGS3',
        'Five_Year_US':'DGS5',
        'Ten_Year_US': 'DGS10',
        'Twenty_Year_US': 'DGS20',
        'Thirty_Year_US':'DGS30'
                     }
        if self.yields is not None:

            return

        today = datetime.datetime.today()
        start_date = today - relativedelta(years=self.years)

        data1 = {}

        for column_name, fred_series in self.SERIES.items():

            series1 = fred.get_series(

                fred_series,
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=today.strftime('%Y-%m-%d')

            )

            data1[column_name] = series1

        self.yields = pd.DataFrame(data1)
        self.yields = self.yields.ffill()
        self.yields = self.yields.dropna()

#########################################################################################################################

    def usyields(self):

        self._build()
        return self.yields.copy()
    
#########################################################################################################################

    
    def market_data(self, years: int, tickers: list, market_ticker: str = 'SPY', lookback_window: int = 30) -> pd.DataFrame:

        """
        Loads historical price data from Yahoo Finance for the specified tickers and market index.
        
        Parameters:
        years (int): Number of years of historical data to load.
        tickers (list): List of stock tickers to load data for.
        market_ticker (str): Ticker symbol for the market index (default is S&P 500).
        lookback_window (int): Lookback window for calculating rolling statistics (default is 30 days).
        -> Returns: 
        pd.DataFrame: A DataFrame containing the raw and log prices for the specified tickers and the market index, along with the highest correlated asset pair.
        """
        self.years = years
        self.tickers = tickers
        self.market_ticker = market_ticker
        self.lookback_window = lookback_window

        """

        Loads historical price data from Yahoo Finance.

        """

        n = self.years
        start_date = datetime.datetime.now() - datetime.timedelta(days=365*n)
        end_date = datetime.datetime.now()
        self.start_date = start_date
        self.end_date = end_date

        try:
            stock_data = yf.download(self.tickers, start=start_date, end=end_date)
            market_data = yf.download(self.market_ticker, start=start_date, end=end_date)
            self.market_data = market_data

        except KeyError as error:
            print(f"yfinance has returned: {error}")

        variables = {}

        # Safely handle single ticker vs mutli-ticker yfinance structural output
        if isinstance(market_data.columns, pd.MultiIndex):variables["market_log_price"] = np.log(
                        market_data["Close"][self.market_ticker])
        else:
            variables["market_log_price"] = np.log(market_data["Close"])

        # Loop through tickers and strictly store raw close and log prices seperately
        for i in self.tickers:
            # Flatten multiindex columns from stock_data
            series_close = stock_data["Close"][i]
            variables[i + "_raw_price"] = series_close
            variables[i + "_log_price"] = np.log(series_close)

        # Create primary working dataframe synced to market trading days
        self.df = pd.DataFrame(variables, index=market_data.index).dropna()

        # Isolate log price columns exclusively to calculate correlation matrix
        log_cols = [i + "_log_price" for i in self.tickers]
        corr_matrix = self.df[log_cols].corr()

        # Unstack upper triangle to locate the highest correlated asset pair
        unstacked = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        ).unstack()

        highest_pairs = unstacked.abs().sort_values(ascending=False)

        # Top_pair holds strings like ('AAPL_log_price', 'MSFT_log_price')
        top_pair = highest_pairs.index[0]

        # Extract the matching base ticker names by stripping out suffix
        ticker_a = top_pair[0].replace("_log_price", "")
        ticker_b = top_pair[1].replace("_log_price", "")

        # Target raw price columns for simulation, log prices for signals
        self.data = pd.DataFrame(index=self.df.index)
        self.data["Price_A"] = self.df[ticker_a + "_raw_price"]
        self.data["Price_B"] = self.df[ticker_b + "_raw_price"]
        self.data["Log_A"] = self.df[ticker_a + "_log_price"]
        self.data["Log_B"] = self.df[ticker_b + "_log_price"]

        self.market_data = market_data
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b

def sayields(date: str) -> pd.DataFrame:

    """
    Fetches South African yield curve data for a given date from the RBond API.
    
    Parameters:
    date (str): The date for which to fetch the yield curve data in 'YYYY-MM-DD' format.
    -> Returns:
    pd.DataFrame: A DataFrame containing the yield curve data with 'years_to_maturity' as the index and 'yield_pct' and 'bond_type' as columns.
    """
    
    url = f'https://rbond.co.za/api/v1/curve/{date}'
    
    data = pd.read_json(url)[['data'][0]]
    data = data.apply(lambda x: pd.Series(x))

    data = data.set_index('years_to_maturity')
    data = data.sort_index()

    data = data[data['yield_pct'] != 0]  # Filter out rows where yield is 0
    data = data[data['bond_type'] != 'ilb']  # Filter out rows where bond_type is 'ilb'
    data = data[data['bond_type'] != 'swap']  # Filter out rows where bond_type is 'swap'

    data = data[~data.index.duplicated(keep='first')]  # Remove duplicate indices, keeping the first occurrence

    return data