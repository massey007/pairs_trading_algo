# This code will pull the entire yield curve from the FRED API and save it to a CSV file.
# It will also print the yield curve to the console.

import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import pandas_datareader as pdr
from scipy.cluster.hierarchy import linkage, dendrogram
from fredapi import Fred
fred = Fred(api_key='17a30f1e933d9c9de2afc4bcd7a5fbcb')


class us_yield_curve_market_data:

    def __init__(self, years):

        self.years = years
        self.repo = None
        self.one_month = None
        self.three_month = None
        self.six_month = None
        self.one_year = None
        self.two_year = None
        self.five_year = None
        self.ten_year = None
        self.twenty_year = None
        self.thirty_year = None
        self.data = None

    def us_yield_curve(self):

        today = datetime.datetime.today()
        n = self.years
        n_year_ago = today - datetime.timedelta(days=365*n)
        start_date = n_year_ago # get the date n years ago in the format YYYY-MM-DD
        end_date = today.strftime('%Y-%m-%d') # get today's date in the format YYYY-MM-DD

        # Fetch the yield curve data from FRED
        self.repo = fred.get_series('DFF', start_date, end_date)
        self.repo = pd.DataFrame(self.repo)
        self.repo.rename(columns={0:'Repo_Rate_US'}, inplace=True)

        self.one_month = fred.get_series('DGS1MO', start_date, end_date)
        self.one_month = pd.DataFrame(self.one_month)
        self.one_month.rename(columns={0:'One_Month_US'}, inplace=True)

        self.three_month = fred.get_series('DGS3MO', start_date, end_date)
        self.three_month = pd.DataFrame(self.three_month)
        self.three_month.rename(columns={0:'Three_Month_US'}, inplace=True)

        self.six_month = fred.get_series('DGS6MO', start_date, end_date)
        self.six_month = pd.DataFrame(self.six_month)
        self.six_month.rename(columns={0:'Six_Month_US'}, inplace=True)

        self.one_year = fred.get_series('DGS1', start_date, end_date)
        self.one_year = pd.DataFrame(self.one_year)
        self.one_year.rename(columns={0:'One_Year_US'}, inplace=True)

        self.two_year = fred.get_series('DGS2', start_date, end_date)
        self.two_year = pd.DataFrame(self.two_year)
        self.two_year.rename(columns={0:'Two_Year_US'}, inplace=True)

        self.five_year = fred.get_series('DGS5', start_date, end_date)
        self.five_year = pd.DataFrame(self.five_year)
        self.five_year.rename(columns={0:'Five_Year_US'}, inplace=True)

        self.ten_year = fred.get_series('DGS10', start_date, end_date)
        self.ten_year = pd.DataFrame(self.ten_year)
        self.ten_year.rename(columns={0:'Ten_Year_US'}, inplace=True)

        self.twenty_year = fred.get_series('DGS20', start_date, end_date)
        self.twenty_year = pd.DataFrame(self.twenty_year)
        self.twenty_year.rename(columns={0:'Twenty_Year_US'}, inplace=True)

        self.thirty_year = fred.get_series('DGS30', start_date, end_date)
        self.thirty_year = pd.DataFrame(self.thirty_year)
        self.thirty_year.rename(columns={0:'Thirty_Year_US'}, inplace=True)

        return self

    def merge_data(self):
  
        self.data = pd.concat(
    [
        self.repo,
        self.one_month,
        self.three_month,
        self.six_month,
        self.one_year,
        self.two_year,
        self.five_year,
        self.ten_year,
        self.twenty_year,
        self.thirty_year
    ],
    axis=1
        )   

        self.data = self.data.ffill()
        self.data = self.data.dropna()


