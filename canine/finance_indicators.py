"""
View license @ ./LICENSE

This is a program that attempts to find common bullish and bearish patterns of stocks.

Notes
-----
To call in another python script, use the following line for importing:

`from finance_indicators import FinanceIndicators as fi
"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import csv
import numpy as np
from time import time
import math
import os

# Note that lstock_symbol.t_df_keys is in a function, but it is a global variable stated with the global keyword.


class FinanceIndicators():
    """
    This class is what stores all the indicators used commonly by investors for figuring out if a security is at a good point for investing or not.
    """

    def __init__(self, ticker=None, sma1_ol=None, sma2_ol=None, simple_ma1_length=None, simple_ma2_length=None, ema1_ol=None, ema2_ol=None, ema1_length=None, ema2_length=None, rsi_ol=None, ppo_ol=None, stdev=None):
        self.ticker = ticker
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
        self.stdev = stdev
        # self.simple_ma3_length = simple_ma3_length

    def stock_symbol(self):
        # stock_symbol
        """
        Gets the stock symbol and MAX data from the yfinance library
        """
        thistory = {}
        exporter = yf.download(self.ticker, period="max")

        self.t_df = pd.DataFrame(data=exporter)
        self.ticker_df = self.t_df.to_dict()

        self.d_percent_change = []

        self.t_df_keys = self.ticker_df["Close"].keys()
        self.t_df_dates = []
        self.li_close = []

    def extract_dates(self):
        t_df_values = []
        for v in self.ticker_df["Close"].keys():
            self.t_df_dates.append(v)

    def extract_values(self):
        """
        This extracts the values of the keys so they can be refered to in the extract_cprices method below
        """
        self.t_df_keys
        self.t_df_keys = list(self.t_df_keys)
        # print(lself.t_df_keys)

    def extract_cprices(self):
        """
        This class method extracts the closing prices for a stock
        """
        for x in range(0, len(self.t_df_keys), 1):
            try:
                dict_i_close = self.ticker_df["Close"][self.t_df_keys[x]]
                self.li_close.append(dict_i_close)
            except KeyError:
                continue
            finally:
                pass

        # print(self.li_close[slice(1, 51, 1)])

    def percent_change(self):
        """
        This class method is what gives us the percentage change information required to somewhat judge a stock

        How % change works:

        ((today's price - yesterday's price)/yesterday's price)*100 (to return a percentage) 
        """
        for x in range(1, len(self.t_df_keys), 1):
            percent_change = (
                (self.li_close[x]-self.li_close[x-1])/self.li_close[x-1])*100
            self.d_percent_change.append(percent_change)
        # print(self.d_percent_change)

    def simple_ma(self, prices, length, output):
        """
        Credit for helping me figure this out goes to https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/

        This class method finds the Simple Moving Average for a stock over a certain period of time. Currently only supports 2 different values

        Parameters
        ----------
        prices: `list`
        The list of prices you want to find the SMA of

        length: `int`
        The length / window of time you want to find for the SMA.

        output_list: `int`
        The list you want the SMA values to be extracted to.
        """
        i = 0

        while i < len(prices) - length + 1:
            timeframe = prices[i:i+length]

            tf_average = sum(timeframe) / length
            output.append(tf_average)
            i += 1

    def exponential_ma(self, prices, length, output_list):
        """
        Calculates the exponential average for a specified number of days.

        Parameters
        ----------
        prices: `list`
        The list of prices you want to find the EMA of

        length: `int`
        The length / window of time you want to find for the EMA.

        output_list: `int`
        The list you want the EMA values to be extracted to.

        Notes
        -----
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
        self.exponential_ma(self.li_close, macd_short, macd12)
        self.exponential_ma(self.li_close, macd_long, macd26)
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

    def ppo_whistogram(self, ppo_divide=26):
        """
        ppo_whistogram() -> Finds the Percentage Price Oscillator for a security.

        To find Percentage Price Oscillator, you take the MACD value above, divide it by the longer term MACD value, than multiply that value by 100 (to get percentage)

        Parameters
        ----------
        None
        """
        self.ppo_ol = []
        self.ppo = []
        self.ppo_sl = []
        output_list = []
        macd26 = []
        self.exponential_ma(self.li_close, ppo_divide, macd26)
        for x in range(0, len(macd), 1):
            self.ppo.append((macd[x] / macd26[x]) * 100)
        self.exponential_ma(self.ppo, 9, self.ppo_sl)
        for x in range(0, len(self.ppo_sl), 1):
            self.ppo_ol.append(self.ppo[x+7] - self.ppo_sl[x])

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
        df = pd.DataFrame(self.li_close)
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

    def standard_deviation(self, input_list, output_list, sd_length=10):
        """Calculates the Standard Deviation measurement used for measuring volatility

        Parameters
        ----------
        input: `list`
        The input list of daily closing prices. Default: fi.li_close

        output: `list`
        The list you want to extract the standard deviation values to. Default: fi.stdev

        sd_length: `int`
        The time window you want 

        Notes
        -----
        `self.stadev` is used for running inside other 

        """
        input_loop = []
        sma = []
        sma_t10 = []
        deviation = np.empty((0, 1), float)
        dsq_ma = np.empty((0, 1), float)
        std_dev = np.empty((0, 1), float)

        ip_df = pd.DataFrame({0: input_list})
        self.simple_ma(input_list, sd_length, sma)
        for i in range(0, len(sma), 1):
            input_loop.append(input_list[i:i+sd_length])
            for x in range(0, sd_length, 1):
                sma_t10.append(sma[i])

        np_sma_t10 = np.array(sma_t10).reshape(
            int(round(len(sma_t10)/sd_length, 0)), sd_length)
        np_input_loop = np.array(input_loop)
        for x in range(0, len(np_input_loop), 1):
            for y in range(0, len(np_input_loop[0]), 1):
                deviation = np.append(
                    deviation, (np_input_loop[x][y] - np_sma_t10[x][y])**2)
        deviation = deviation.reshape(int(len(deviation)/sd_length), sd_length)

        # for x in range(0,len(deviation),1):
        dsq_ma = np.mean(deviation, axis=1)
        self.stadev = np.sqrt(dsq_ma)
        output_list.append(list(np.sqrt(dsq_ma)))
    def bollinger_bands(self,bb_length=20,multiplier=2):
        """This gathers the bollinger bands as created by John Bollinger

        Notes
        
        """
        stddev = []
        bb_stddev = []
        self.bb_middle = []
        self.bb_upper = []
        self.bb_lower = []
        self.simple_ma(self.li_close,bb_length,self.bb_middle)
        self.standard_deviation(self.li_close,stddev,bb_length)
        for x in range(0,len(stddev[0]),1):
            bb_stddev.append(stddev[0][x]*multiplier)
        for x in range(0,len(self.bb_middle),1):
            self.bb_lower.append(self.bb_middle[x] - bb_stddev[x])
            self.bb_upper.append(self.bb_middle[x] + bb_stddev[x])
    def ulcer_index(self,ui_length=14):
        i = 0
        seperated_close = []
        np_seperated_close = []
        max_prices = []
        np_max_prices = np.empty((0,1))
        percent_drawdown = []
        pd_squared = []
        sq_average = []
        self.ulcer_index = []
        while i < len(self.li_close)-ui_length:
            window = self.li_close[i:i+ui_length]
            np_seperated_close = np.append(np_seperated_close,window,axis=0)
            i += 1
        np_seperated_close = np_seperated_close.reshape(int(len(np_seperated_close)/ui_length),ui_length)
        # np_seperated_close = np.array(seperated_close).reshape(len(np_seperated_close),ui_length)
        for x in range(0,len(np_seperated_close),1):
            for y in range(1,len(np_seperated_close[0]),1):
                np_max_prices = np.append(np_max_prices,max(np_seperated_close[x][0:y]))
        # np_max_prices = np.amax(np_seperated_close,axis=1)
        # np_max_prices = np.array(max_prices)
        for x in range(0,int(len(np_max_prices)/(ui_length-1)),1):
            np_max_prices = np.insert(np_max_prices,0,np_seperated_close[x][0])
        np_max_prices = np_max_prices.reshape(int(len(np_max_prices)/ui_length),ui_length)

        # percent_drawdown = np_seperated_close - np_max_prices
        while i < len(pd_squared):
            window = pd_squared[i:i+ui_length]
            self.ulcer_index.append(math.sqrt(sum(window)/ui_length))
            i += 1



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
    for x in range(0, amount, 1):
        output_list.insert(0, float("NaN"))


if __name__ == "__main__":
    # Defining fi and the variables inside
    fi = FinanceIndicators()
    fi.ticker = "BNS.TO"
    fi.sma1_ol = []
    fi.sma2_ol = []
    fi.simple_ma1_length = 50
    fi.simple_ma2_length = 200
    fi.ema1_length = 20
    fi.rsi_ol = []
    stdev = []
    fi.stdev = []
    stddev_ol = []
    fi.stock_symbol()
    fi.extract_dates()
    fi.extract_values()
    fi.extract_cprices()
    fi.percent_change()
    fi.simple_ma(fi.li_close, fi.simple_ma1_length, fi.sma1_ol)
    fi.simple_ma(fi.li_close, fi.simple_ma2_length, fi.sma2_ol)
    fi.macd_whistogram()
    fi.rsi(fi.rsi_ol)
    fi.ppo_whistogram()
    fi.standard_deviation(fi.li_close, stdev)
    fi.bollinger_bands()
    fi.ulcer_index()
    fi.stdev.append(list(stdev[0]))
    nan_generator(1, fi.d_percent_change)
    nan_generator(fi.simple_ma1_length-1, fi.sma1_ol)
    nan_generator(fi.simple_ma2_length-1, fi.sma2_ol)
    nan_generator(24, macd)
    nan_generator(31, macd_9ema)
    nan_generator(31, m_histogram)
    nan_generator(24, fi.ppo)
    nan_generator(31, fi.ppo_sl)
    nan_generator(31, fi.ppo_ol)
    nan_generator(9, fi.stdev[0])
    nan_generator(19, fi.bb_lower)
    nan_generator(19, fi.bb_middle)
    nan_generator(19, fi.bb_upper)
    nan_generator(19, fi.bb_upper)
    nan_generator(14, fi.ulcer_index)
    # fi.stdev
    # This runs the zip() method and exports the values to a CSV file. Only enabled in stable packages, otherwise commented out.
    # DO NOT DELETE THE BELOW LINES
    all_indicators = zip(fi.t_df_dates, fi.li_close, fi.d_percent_change, fi.sma1_ol,
                         fi.sma2_ol, macd, macd_9ema, m_histogram, fi.ppo, fi.ppo_sl, fi.ppo_ol, fi.rsi_ol[0],fi.stdev[0],fi.bb_lower,fi.bb_middle,fi.bb_upper,fi.ulcer_index)
    ai_df = pd.DataFrame(all_indicators, columns=[
        "Timestamp", "Close", "Daily Percent Change", "50-Day MA", "200-Day MA", "MACD", "MACD Signal Line", "MACD Histogram", "PPO", "PPO Signal Line", "PPO Histogram", "RSI","Standard Deviation","Lower Bollinger Band","Middle Bollinger Band","Upper Bollinger Band","Ulcer Index"])
    os.chdir("..")
    print(os.getcwd())
    ai_df.to_csv(f'StockMarketPatterns/CSVinfo/{fi.ticker} {time()}.csv', index=False)
