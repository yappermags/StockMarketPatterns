"""
View license @ ./LICENSE

This is a program that attempts to find common bullish and bearish patterns of stocks.

Currently, it can only find percentages, but this will change in the near future
"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import csv
import numpy as np
from time import time

thistory = {}

ticker = "AMZN"
exporter = yf.download(ticker, period="max")

t_df = pd.DataFrame(data=exporter)
ticker_df = t_df.to_dict()

d_percent_change = []

t_df_keys = ticker_df["Close"].keys()
t_df_dates = []
li_close = []
# Note that lt_df_keys is in a function, but it is a global variable stated with the global keyword.


class FinanceIndicators():
    """
    This class is what stores all the indicators used commonly by investors for figuring out if a security is at a good point for investing or not.
    """

    def __init__(self, sma1_ol=None, sma2_ol=None, simple_ma1_length=None, simple_ma2_length=None, ema1_ol=None, ema2_ol=None, ema1_length=None, ema2_length=None, rsi_ol=None,ppo_ol=None):
        self.simple_ma1_length = simple_ma1_length
        self.simple_ma2_length = simple_ma2_length
        self.sma1_ol = sma1_ol
        self.sma2_ol = sma2_ol
        self.ema1_ol = ema1_ol
        self.ema2_ol = ema2_ol
        self.ema1_length = ema1_length
        self.ema2_length = ema2_length
        self.rsi_ol = rsi_ol
        self.ppo_ol = ppo_ol
        # self.simple_ma3_length = simple_ma3_length

    def extract_dates(self):
        t_df_values = []
        for v in ticker_df["Close"].keys():
            t_df_dates.append(v)

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

        # print(li_close[slice(1, 51, 1)])

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

    def simple_ma(self, prices, length, output):
        """
        Credit for helping me figure this out goes to https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/

        This class method finds the Simple Moving Average for a stock over a certain period of time. Currently only supports 2 different values

        Parameters: prices - The list of prices you want to find the SMA of

        length: The length / window of time you want to find for the SMA.

        output_list: The list you want the SMA values to be extracted to.
        """
        i = 0

        while i < len(prices) - length + 1:
            timeframe = prices[i:i+length]

            tf_average = sum(timeframe) / length
            output.append(tf_average)
            i += 1
        print(output)

    def exponential_ma(self, prices, length, output_list):
        """
        Calculates the exponential average for a specified number of days.

        Parameters: prices - The list of prices you want to find the EMA of

        length: The length / window of time you want to find for the EMA.

        output_list: The list you want the EMA values to be extracted to.

        EMA: https://school.stockcharts.com/doku.php?id=technical_indicators:moving_averages
        """
        multiplier = (2 / (length+1))
        i = 0
        e_fv = sum(prices[0:length]) / length
        output_list.append(e_fv)
        while i < len(prices) - length+1:
            window = prices[i:i+length]
            e_window_average = (
                window[-1] - output_list[-1]) * multiplier + output_list[-1]
            output_list.append(e_window_average)
            i += 1

    def macd_whistogram(self, macd_short=12, macd_long=26, macd_sl=9):
        """
        The MACD is a calculation of the 12-period EMA - the 26-period EMA. A more advanced calculation (MACD histogram) uses a signal line, which is the 9-period EMA of the MACD itself, which is subtracted by the original MACD value.
        """
        global m_histogram
        global macd_9ema
        global macd
        m_histogram = []
        macd_short = 12
        macd_long = 26
        macd_sl = 9
        macd12 = []
        macd26 = []
        macd = []
        macd_9ema = []
        self.exponential_ma(li_close, macd_short, macd12)
        self.exponential_ma(li_close, macd_long, macd26)
        # print(len(macd12))
        # print(len(macd26))
        for x in range(0, len(macd26), 1):
            macd_calc = macd12[x+14]-macd26[x]
            macd.append(macd_calc)
        # print(macd)
        self.exponential_ma(macd, macd_sl, macd_9ema)
        for y in range(0, len(macd_9ema), 1):
            hist_calc = macd[y+7] - macd_9ema[y]
            m_histogram.append(hist_calc)
        # print(m_histogram)

    def rsi(self, output_list, length=14):
        """
        Calculates the RSI based on the closing price of a security. 
        Full credit for helping me figure out how to implement this into my program goes to alpharitims.com (link inside README.md)

        Parameters
        ----------
        output_list:  `list`
        The list you want to extract the RSI values to.

        length: `int`
        The length to calculate RSI. Default is 14, and a change is not recommended.

        """
        df = pd.DataFrame(li_close)
        # Get positive and negative price differences
        df[1] = df.diff(1)
        df[2] = df[1].clip(lower=0).round(2)
        df[3] = df[1].clip(upper=0).abs().round(2)

        # Get Average positive and negative price difference
        # Get WMS averages
        df[4] = df[2].rolling(window=length, min_periods=length).mean()[
            :length+1]
        df[5] = df[3].rolling(window=length, min_periods=length).mean()[
            :length+1]
        # Average Gains
        for i, row in enumerate(df[4].iloc[length+1:]):
            df[4].iloc[i + length + 1] = (df[4].iloc[i + length]
                                          * (length - 1) + df[2].iloc[i + length + 1]) / length

        # Average Losses
        for i, row in enumerate(df[5].iloc[length+1:]):
            df[5].iloc[i + length + 1] = (df[5].iloc[i + length]
                                          * (length - 1) + df[3].iloc[i + length + 1]) / length

        # View initial results
        df[6] = df[4] / df[5]
        df[7] = 100 - (100 / (1 + df[6]))
        output_list.append(list(df[7]))
    def ppo(self,output_list):
        output_list = []
        macd26 = []
        self.exponential_ma(li_close, 26, macd26)
        for x in range(0,len(macd),1):
            output_list.append((macd[x] / macd26[x]) * 100)


# Defining fi and the variables inside
fi = FinanceIndicators()
fi.sma1_ol = []
fi.sma2_ol = []
fi.simple_ma1_length = 50
fi.simple_ma2_length = 200
fi.ema1_ol = []
# fi.ema2_ol = []
fi.ema1_length = 20
fi.rsi_ol = []
# fi.ema2_length = 50
# fi.simple_ma3_length = 0
# eg = [7,7,9,674,6,78,8,6,7,4,7,8,6]


def nan_generator(amount, output_list):
    """ Takes the np.zeros() array and adds it to a non-numpy list
    This very simple function takes an arroy of np.zeros() and puts it into the first values of a non-numpy list.

    Parameters
    ----------
    amount : int
        The amount of zeros you need to use.
    output_list : list
        The list you want to add the zeros to.
    """
    zeros_var = []
    for x in range(0, amount, 1):
        output_list.insert(0, float("NaN"))
    # print(zeros_var)


if __name__ == "__main__":
    fi.extract_dates()
    fi.extract_values()
    fi.extract_cprices()
    fi.percent_change()
    fi.simple_ma(li_close, fi.simple_ma1_length, fi.sma1_ol)
    fi.simple_ma(li_close, fi.simple_ma2_length, fi.sma2_ol)
    fi.macd_whistogram()
    fi.rsi(fi.rsi_ol)
    nan_generator(1, d_percent_change)
    nan_generator(fi.simple_ma1_length-1, fi.sma1_ol)
    nan_generator(fi.simple_ma2_length-1, fi.sma2_ol)
    nan_generator(24, macd)
    nan_generator(31, macd_9ema)
    nan_generator(31, m_histogram)
    # This runs the zip() method and exports the values to a CSV file. Only enabled in stable packages, otherwise commented out.
    # DO NOT DELETE THE BELOW LINES
    all_indicators = zip(t_df_dates, li_close, d_percent_change, fi.sma1_ol,
                         fi.sma2_ol, macd, macd_9ema, m_histogram, fi.rsi_ol[0])
    ai_df = pd.DataFrame(all_indicators, columns=[
        "Timestamp", "Close", "Daily Percent Change", "50-Day MA", "200-Day MA", "MACD", "MACD Signal Line", "MACD Histogram", "RSI"])
    # ai_df.to_csv(f'CSVinfo/{ticker} {time()}.csv', index=False)
