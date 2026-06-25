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


class USYieldCurveMarketData:
           
    SERIES = {
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

    def __init__(self, years):

        self.years = years

        self._data = None

    def _build(self):

        # Only download data if it hasn't been downloaded yet

        if self._data is not None:

            return

        today = datetime.datetime.today()

        start_date = today - relativedelta(years=self.years)

        data = {}

        for column_name, fred_series in self.SERIES.items():

            series = fred.get_series(

                fred_series,

                observation_start=start_date.strftime('%Y-%m-%d'),

                observation_end=today.strftime('%Y-%m-%d')

            )

            data[column_name] = series

        self._data = pd.DataFrame(data)

        # Forward-fill weekends and holidays

        self._data = self._data.ffill()

        # Remove any remaining NaNs

        self._data = self._data.dropna()

    @property
    def data(self):

        self._build()

        # Return a copy to prevent accidental modification

        return self._data.copy()