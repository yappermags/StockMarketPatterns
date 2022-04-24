"""
View license @ LICENCE file

This is a program that attempts to find common bullish and bearish patterns of stocks.

Currently, it can only find percentages, but this will change in the near future
"""

from tkinter import E
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import csv

thistory = {}

ticker = "AMZN"
icker = yf.Ticker(ticker)
exporter = yf.download(ticker, period="max")
# earnings = icker.calendar

t_df = pd.DataFrame(data=exporter)
ticker_df = t_df.to_dict()
# print(exporter)
keys_tr = ("Open", "High", "Low", "Adj Close", "Volume")


def pop_multiple(var, *args):
    """
    This is a simple function that can be used to pop multiple items from a list or dictonary

    var -> The variable you want to pop the items or keys

    *args -> The items or keys you want to pop
    """
    args = args[0]
    iter(list(args))
    for x in args:
        var.pop(x)


pop_multiple(ticker_df, keys_tr)
# print(ticker_df)
# print(len(ticker_df["Close"].keys())-1)

# percentage_change = []

# print(percentage_change)


d_percent_change = []
# print(percent_change)
# eg.append(percent_change)

t_df_keys = ticker_df["Close"].keys()
li_close = []
# Note that lt_df_keys is in a function, but it is a global variable stated with the global keyword.


class FinanceIndicators():
    """
    This class is what stores all the indicators used commonly by investors for figuring out if a security
    """

    def __init__(self, simple_ma1_length=None, simple_ma2_length=None, ema1_ol=None, ema2_ol=None, ema1_length=None, ema2_length=None):
        self.simple_ma1_length = simple_ma1_length
        self.simple_ma2_length = simple_ma2_length
        self.ema1_ol = ema1_ol
        self.ema2_ol = ema2_ol
        self.ema1_length = ema1_length
        self.ema2_length = ema2_length
        # self.simple_ma3_length = simple_ma3_length

    def extract_values(self):
        """
        This extracts the values of the keys so they can be refered to in the extract_cprices method below
        """
        global lt_df_keys
        lt_df_keys = list(t_df_keys)
        # print(lt_df_keys)

    def extract_cprices(self):
        """
        This class method extracts the closing prices for a stock
        """
        for x in range(0, len(t_df_keys), 1):
            try:
                dict_i_close = ticker_df["Close"][lt_df_keys[x]]
                li_close.append(dict_i_close)
            except KeyError:
                continue
            finally:
                pass

        print(li_close[slice(1, 51, 1)])

    def percent_change(self):
        """
        This class method is what gives us the percentage change information required to somewhat judge a stock

        How % change works:

        ((today's price - yesterday's price)/yesterday's price)*100 (to return a percentage) 
        """
        print(self.simple_ma1_length)
        for x in range(1, len(t_df_keys), 1):
            percent_change = ((li_close[x]-li_close[x-1])/li_close[x-1])*100
            d_percent_change.append(percent_change)
        # print(d_percent_change)

    def simple_ma(self):
        """
        Credit for helping me figure this out goes to https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/

        This class method finds the Simple Moving Average for a stock over a certain period of time. Currently only supports 2 different values

        There are no parameters you have to enter in for the function, but rather the self.simple_ma_length value held in the __init__ function
        """
        averages = []
        i = 0

        while i < len(li_close) - self.simple_ma1_length + 1:
            timeframe = li_close[i:i+self.simple_ma1_length]

            tf_average = sum(timeframe) / self.simple_ma1_length
            averages.append(tf_average)
            i += 1
        # print(averages)
        averages2 = []
        j = 0

        while j < len(li_close) - self.simple_ma2_length + 1:
            timeframe2 = li_close[i:i+self.simple_ma2_length]

            tf_average2 = sum(timeframe2) / self.simple_ma2_length
            averages2.append(tf_average2)
            j += 1
        # print(averages)

    def exponential_ma(self, prices, length, output_list):
        """
        Calculates the exponential average for a specified number of days.

        Parameters: prices - The list of prices you want to find the EMA of

        length: The length / window of time you want to find for the EMA.

        output_list: The list you want the EMA values to be extracted to.

        Calculation for a 20-day exponential moving average is:

        First day = SMA

        Multiplier = (2 / (20 + 1)) 

        EMA = (Close - EMA(previous day)) * multiplier + EMA (previous day)
        """
        multiplier = (2 / (self.ema1_length+1))
        # print(multiplier)
        i = 0
        e_fv = sum(prices[0:length]) / length
        output_list.append(e_fv)
        while i < len(prices) - length+1:
            window = prices[i:i+length]
            e_window_average = (
                window[-1] - output_list[-1]) * multiplier + output_list[-1]
            output_list.append(e_window_average)
            i += 1
        print(output_list)

    def rsi(self):
        pass

    def macd(self):
        self.ema1_length = 12
        self.ema2_length = 26
        print(self.ema1_length)


fi = FinanceIndicators()
# fi.__init__()
fi.simple_ma1_length = 50
fi.simple_ma2_length = 200
fi.ema1_ol = e1_moving_averages = []
fi.ema2_ol = e2_moving_averages = []
fi.ema1_length = 20
fi.ema2_length = 50
# fi.simple_ma3_length = 0
fi.extract_values()
fi.extract_cprices()
fi.percent_change()
fi.simple_ma()
fi.exponential_ma(li_close, fi.ema1_length, fi.ema1_ol)
# fi.exponential_ma(li_close, fi.ema2_length, )
# print(len(t_df_keys))
