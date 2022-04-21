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
exporter = yf.download(ticker, period="5y")
# earnings = icker.calendar

t_df = pd.DataFrame(data=exporter)
ticker_df = t_df.to_dict()
# print(exporter)
keys_tr = ("Open", "High", "Low", "Adj Close", "Volume")
# print(ticker_df)

# print(datetime.now())
# print(datetime(2021,1,1)+timedelta(1))
# print(datetime(2022,1,1).weekday())


trading_list = []
# def trading_days(year):
#     """
#     This is a function used to find a whole year, then remove all the weekends from it (Note that STAT holidays are not removed as of right now)

#     year -> Year you want to find (int only)
#     """
#     if type(year) != int:
#         raise NameError("Only integers are acceptable here")
#     else:
#         # weekends = []
#         trading_list.append(datetime(year, 1, 1))
#         for x in range(1, 365, 1):
#             eg = pd.Timestamp(datetime(year, 1, 1)+timedelta(x))
#             trading_list.append(eg)
#         for y in range(0, 313, 1):
#             if trading_list[y].weekday() == 5:
#                 trading_list.pop(y)
#         for y in range(0, 260, 1):
#             if trading_list[y].weekday() == 6:
#                 trading_list.pop(y)
#         print(trading_list)


# trading_days(2022)


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


class PcPatterns():
    """
    This is the class which I am going to use to run all my programs
    """

    def __init__(self, simple_ma1_length=None, simple_ma2_length=None, simple_ma3_length=None):
        self.simple_ma1_length = simple_ma1_length
        self.simple_ma2_length = simple_ma2_length
        # self.simple_ma3_length = simple_ma3_length
        # self.len_of_list = len_of_list

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

        There are no parametes you have to enter in for the function, but rather the self.simple_ma_length value held in the __init__ function
        """
        averages = []
        i = 0

        while i < len(li_close) - self.simple_ma1_length + 1:
            timeframe = li_close[i:i+self.simple_ma1_length]

            tf_average = sum(timeframe) / self.simple_ma1_length
            averages.append(tf_average)
            i += 1
        print(averages)
        averages2 = []
        j = 0

        while j < len(li_close) - self.simple_ma2_length + 1:
            timeframe2 = li_close[i:i+self.simple_ma2_length]

            tf_average2 = sum(timeframe2) / self.simple_ma2_length
            averages2.append(tf_average2)
            j += 1
        print(averages)


pc = PcPatterns()
# pc.__init__()
pc.simple_ma1_length = 50
pc.simple_ma2_length = 200
# pc.simple_ma3_length = 0
pc.extract_values()
pc.extract_cprices()
pc.percent_change()
pc.simple_ma()

print(len(t_df_keys))
