# This code will pull the entire yield curve from the FRED API and save it to a CSV file.
# It will also print the yield curve to the console.

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
    
    def market_data(self, tickers, market_ticker = '^GSPC', lookback_window=30):

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
        df = pd.DataFrame(variables, index=market_data.index).dropna()

        # Isolate log price columns exclusively to calculate correlation matrix
        log_cols = [i + "_log_price" for i in self.tickers]
        corr_matrix = df[log_cols].corr()

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
        market_data = pd.DataFrame(index=df.index)
        market_data["Price_A"] = df[ticker_a + "_raw_price"]
        market_data["Price_B"] = df[ticker_b + "_raw_price"]
        market_data["Log_A"] = df[ticker_a + "_log_price"]
        market_data["Log_B"] = df[ticker_b + "_log_price"]

        return market_data