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

    def __init__(self, start_date, end_date):

        self.start_date = start_date
        self.end_date = end_date
        self.repo = None
        self.one_month = None
        self.three_month = None
        self.six_month = None
        self.one_year = None
        self.two_year = None
        self.three_year = None
        self.five_year = None
        self.ten_year = None
        self.twenty_year = None
        self.thirty_year = None
        self.data = None

    def us_yield_curve(self):

        # Fetch the yield curve data from FRED
        self.repo = fred.get_series('DFF', self.start_date, self.end_date)
        self.repo = pd.DataFrame(self.repo)
        self.repo.rename(columns={0:'Repo_Rate_US'}, inplace=True)

        self.one_month = fred.get_series('DGS1MO', self.start_date, self.end_date)
        self.one_month = pd.DataFrame(self.one_month)
        self.one_month.rename(columns={0:'One_Month_US'}, inplace=True)

        self.three_month = fred.get_series('DGS3MO', self.start_date, self.end_date)
        self.three_month = pd.DataFrame(self.three_month)
        self.three_month.rename(columns={0:'Three_Month_US'}, inplace=True)

        self.six_month = fred.get_series('DGS6MO', self.start_date, self.end_date)
        self.six_month = pd.DataFrame(self.six_month)
        self.six_month.rename(columns={0:'Six_Month_US'}, inplace=True)

        self.one_year = fred.get_series('DGS1', self.start_date, self.end_date)
        self.one_year = pd.DataFrame(self.one_year)
        self.one_year.rename(columns={0:'One_Year_US'}, inplace=True)

        self.two_year = fred.get_series('DGS2', self.start_date, self.end_date)
        self.two_year = pd.DataFrame(self.two_year)
        self.two_year.rename(columns={0:'Two_Year_US'}, inplace=True)

        self.five_year = fred.get_series('DGS5', self.start_date, self.end_date)
        self.five_year = pd.DataFrame(self.five_year)
        self.five_year.rename(columns={0:'Five_Year_US'}, inplace=True)

        self.ten_year = fred.get_series('DGS10', self.start_date, self.end_date)
        self.ten_year = pd.DataFrame(self.ten_year)
        self.ten_year.rename(columns={0:'Ten_Year_US'}, inplace=True)

        self.twenty_year = fred.get_series('DGS20', self.start_date, self.end_date)
        self.twenty_year = pd.DataFrame(self.twenty_year)
        self.twenty_year.rename(columns={0:'Twenty_Year_US'}, inplace=True)

        self.thirty_year = fred.get_series('DGS30', self.start_date, self.end_date)
        self.thirty_year = pd.DataFrame(self.thirty_year)
        self.thirty_year.rename(columns={0:'Thirty_Year_US'}, inplace=True)

    def merge_data(self):
  
        self.data = pd.DataFrame({
            'One_Month_US': self.one_month['One_Month_US'].squeeze(),
            'Three_Month_US': self.three_month['Three_Month_US'].squeeze().reindex(self.one_month.index),
            'Six_Month_US': self.six_month['Six_Month_US'].squeeze().reindex(self.one_month.index),
            'One_Year_US': self.one_year['One_Year_US'].squeeze().reindex(self.one_month.index),
            'Two_Year_US': self.two_year['Two_Year_US'].squeeze().reindex(self.one_month.index),
            'Five_Year_US': self.five_year['Five_Year_US'].squeeze().reindex(self.one_month.index),
            'Ten_Year_US': self.ten_year['Ten_Year_US'].squeeze().reindex(self.one_month.index),
            'Twenty_Year_US': self.twenty_year['Twenty_Year_US'].squeeze().reindex(self.one_month.index),
            'Thirty_Year_US': self.thirty_year['Thirty_Year_US'].squeeze().reindex(self.one_month.index)
        })
        self.data = self.data.set_index()
        self.data.dropna(inplace=True)


    def plot_yield_curve(self):
  
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Repo_Rate_US'], label='Repo Rate')
        plt.plot(self.data.index, self.data['One_Month_US'], label='1 Month')
        plt.plot(self.data.index, self.data['Three_Month_US'], label='3 Month')
        plt.plot(self.data.index, self.data['Six_Month_US'], label='6 Month')
        plt.plot(self.data.index, self.data['One_Year_US'], label='1 Year')
        plt.plot(self.data.index, self.data['Two_Year_US'], label='2 Year')
        plt.plot(self.data.index, self.data['Five_Year_US'], label='5 Year')
        plt.plot(self.data.index, self.data['Ten_Year_US'], label='10 Year')
        plt.plot(self.data.index, self.data['Twenty_Year_US'], label='20 Year')
        plt.plot(self.data.index, self.data['Thirty_Year_US'], label='30 Year')
        plt.title('US Yield Curve Over Time')
        plt.xlabel('Date')
        plt.ylabel('Yield (%)')
        plt.legend()
        plt.grid()
        plt.show()

        def dataframe(self):
            if self.data is not None:
                return self.data
            else:
                print("No data available. Please fetch and merge data first.")
                return None