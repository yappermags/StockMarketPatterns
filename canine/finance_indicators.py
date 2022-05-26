"""
View license @ ./LICENSE

This is a program that attempts to find common bullish and bearish patterns of stocks.

Notes
-----
To call in another python script, use the following line for importing:

`from finance_indicators import FinanceIndicators as fi
"""

import yfinance as yf
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from time import time
import os
import re

# Note that lstock_symbol.t_df_keys is in a function, but it is a global variable stated with the global keyword.


class FinanceIndicators():
    """
    This class is what stores all the indicators used commonly by investors for figuring out if a security is at a good point for investing or not.
    """
    def __init__(self, ticker=None, sma1_ol=None, sma2_ol=None, simple_ma1_length=None, simple_ma2_length=None, ema1_ol=None, ema2_ol=None, ema1_length=None, ema2_length=None, rsi_ol=None, ppo_ol=None, stdev=None,fallback=None):
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
        self.fallback = fallback
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
        self.change = []

        self.t_df_keys = self.ticker_df["Close"].keys()
        self.t_df_dates = []
        self.li_open = []
        self.li_high = []
        self.li_low = []
        self.li_close = []

    def extract_dates(self):
        t_df_values = []
        for v in self.ticker_df["Close"].keys():
            self.t_df_dates.append(v)

    def future_dates(self, dates_ahead):
        wd = datetime.weekday(self.t_df_dates[-1])
        first_date_calc = str(self.t_df_dates[-1])
        first_date_calc = first_date_calc.replace(' 00:00:00', '')
        delta = timedelta(days=round((26 + 26), 0))
        final_date = str(delta + date.fromisoformat(first_date_calc))
        if type(re.search(".TO", self.ticker)) == re.Match:
            tsx = mcal.get_calendar('TSX')
            fd = tsx.schedule(start_date=first_date_calc, end_date=final_date)
            self.new_dates = list(fd.market_open._stat_axis)
            for x in range(0, len(self.new_dates)-dates_ahead, 1):
                self.new_dates.pop()
        elif type(re.search(".TO", self.ticker)) != re.Match:
            nyse = mcal.get_calendar('NYSE')
            fd = nyse.schedule(start_date=first_date_calc, end_date=final_date)
            self.new_dates = list(fd.market_open._stat_axis)
            for x in range(0, len(self.new_dates)-dates_ahead, 1):
                self.new_dates.pop()
        else:
            raise NameError(
                f"We do not currently have that exchange in our databases.")
        for x in range(0, len(self.new_dates), 1):
            self.t_df_dates.append(self.new_dates[x])

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
                dict_i_open = self.ticker_df["Open"][self.t_df_keys[x]]
                dict_i_high = self.ticker_df["High"][self.t_df_keys[x]]
                dict_i_low = self.ticker_df["Low"][self.t_df_keys[x]]
                dict_i_close = self.ticker_df["Close"][self.t_df_keys[x]]
                self.li_open.append(dict_i_open)
                self.li_high.append(dict_i_high)
                self.li_low.append(dict_i_low)
                self.li_close.append(dict_i_close)
                self.np_close = np.array(self.li_close)
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
            self.change.append(self.li_close[x]-self.li_close[x-1])
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

    def bollinger_bands(self, bb_length=20, multiplier=2):
        """This gathers the bollinger bands as created by John Bollinger

        Notes

        """
        stddev = []
        bb_stddev = []
        self.bb_middle = []
        self.bb_upper = []
        self.bb_lower = []
        self.simple_ma(self.li_close, bb_length, self.bb_middle)
        self.standard_deviation(self.li_close, stddev, bb_length)
        for x in range(0, len(stddev[0]), 1):
            bb_stddev.append(stddev[0][x]*multiplier)
        for x in range(0, len(self.bb_middle), 1):
            self.bb_lower.append(self.bb_middle[x] - bb_stddev[x])
            self.bb_upper.append(self.bb_middle[x] + bb_stddev[x])
        bb_middle = np.array(self.bb_middle)
        bb_upper = np.array(self.bb_upper)
        bb_lower = np.array(self.bb_lower)
        self.bbw = (bb_upper - bb_lower) / bb_middle
        self.np_close = np.delete(self.np_close,np.arange(bb_length-1))
        self.percent_b = (self.np_close - bb_lower) / (bb_upper - bb_lower)

    def seperated_close(self, prices, sc_length, output_list):
        """Uses numpy to seperate the values in a list

        Parameters
        ----------
        prices: `list`
        The list you want the for the prices to be sepeated

        sc_length: `int`
        How long each individual seperation is

        output_list: `list`
        The list you want to extract the seperated_close values to.
        """
        sc_lc = int(len(prices)-(sc_length-1))
        sc = np.empty((0, 1), float)
        for x in range(0, sc_lc, 1):
            window = prices[x:x+sc_length]
            sc = np.append(sc, window)
        sc = sc.reshape(int(len(sc)/sc_length), sc_length)
        output_list.append(list(sc))

    def waldo_volatility_indicator(self, wvi_length=28):
        """Measures volatility and works a lot like the Ulcer Index.

        Parameters
        ----------

        wvi_length: `int`
        The length of time you want to measure the Waldo Volatility Indicator

        """
        seperated_close = np.empty((0, 1), float)
        self.waldo_vola_indicator = []
        for x in range(0, len(self.li_close)-wvi_length, 1):
            window = self.li_close[x:x+wvi_length]
            seperated_close = np.append(seperated_close, window)
        seperated_close = seperated_close.reshape(
            int(len(seperated_close)/wvi_length), wvi_length)
        mp = np.amax(seperated_close, axis=1)
        max_prices = np.repeat(mp, wvi_length)
        max_prices = max_prices.reshape(
            int(len(max_prices)/wvi_length), wvi_length)
        price_dropdown = np.abs(seperated_close - max_prices)
        pd_mean = np.mean(price_dropdown, axis=1)
        self.waldo_vola_indicator = np.sqrt(pd_mean)
        self.waldo_vola_indicator = np.insert(
            self.waldo_vola_indicator, 0, np.zeros(wvi_length))
        self.waldo_vola_indicator = np.append(
            self.waldo_vola_indicator, np.zeros(self.fallback))

    def ichimoku_cloud(self, tenkan_sen_len=9, kijun_sen_len=26, senkou_b_len=52, fallback=26):
        """Calcaulates all 5 Ichimoku Cloud calculations that make the full cloud.

        Parameters
        ----------
        tenkan_sen_len=9: `int`
        The time period you want to use for the Tenkan Sen (Conversion Line)

        kijun_sen_len=26: `int`
        The time period you want to use for the Tenkan Sen (Base Line)

        senkou_b_len=52: `int`
        The time period you want to use for Senkou Span B (Leading Span B)

        fallback=26: `int`
        The amount of time you want the Senkou A+B to be set in the future, and the time you want the chikou_span to be set in the past
        """
        ts_sc_min = []
        ts_sc_max = []
        ks_sc_min = []
        ks_sc_max = []
        se_b_min = []
        se_b_max = []
        self.tenkan_sen = []
        ks_sc = []
        self.kijun_sen = []
        self.senkou_a = []
        self.senkou_b = []
        self.chikou_span = []
        self.seperated_close(self.li_low, tenkan_sen_len, ts_sc_min)
        self.seperated_close(self.li_high, tenkan_sen_len, ts_sc_max)
        ts_sc_min = np.array(ts_sc_min[0])
        ts_sc_max = np.array(ts_sc_max[0])
        self.seperated_close(self.li_low, kijun_sen_len, ks_sc_min)
        self.seperated_close(self.li_high, kijun_sen_len, ks_sc_max)
        ks_sc_min = np.array(ks_sc_min[0])
        ks_sc_max = np.array(ks_sc_max[0])
        self.seperated_close(self.li_low, senkou_b_len, se_b_min)
        self.seperated_close(self.li_high, senkou_b_len, se_b_max)
        se_b_min = np.array(se_b_min[0])
        se_b_max = np.array(se_b_max[0])

        def find_min_max(min_iv, max_iv, output_number):
            max_iv_list = []
            min_iv_list = []
            for x in range(0, len(max_iv), 1):
                max_iv_list.append(max(max_iv[x]))
                min_iv_list.append(min(min_iv[x]))
            max_iv_list = np.array(max_iv_list)
            min_iv_list = np.array(min_iv_list)
            if output_number == 0:
                self.tenkan_sen = ((min_iv_list + max_iv_list)/2)
                self.tenkan_sen = np.insert(
                    self.tenkan_sen, 0, np.zeros(tenkan_sen_len-1))
            elif output_number == 1:
                self.kijun_sen = ((min_iv_list + max_iv_list)/2)
                self.kijun_sen = np.insert(
                    self.kijun_sen, 0, np.zeros(kijun_sen_len-1))
            elif output_number == 2:
                self.senkou_b = ((min_iv_list + max_iv_list)/2)
                self.senkou_b = np.insert(
                    self.senkou_b, 0, np.zeros((senkou_b_len-1)+fallback))
            else:
                raise SyntaxError(f"{output_number} not available.")
        find_min_max(ts_sc_min, ts_sc_max, 0)
        find_min_max(ks_sc_min, ks_sc_max, 1)
        self.senkou_a = (self.tenkan_sen + self.kijun_sen)/2
        self.senkou_a = np.insert(self.senkou_a, 0, np.zeros(fallback))
        self.tenkan_sen = np.append(self.tenkan_sen, np.zeros(fallback))
        self.kijun_sen = np.append(self.kijun_sen, np.zeros(fallback))
        find_min_max(se_b_min, se_b_max, 2)
        np_close = np.array(self.li_close)
        np_close = np.delete(np_close, np.arange(fallback))
        np_close = np.append(np_close, np.zeros(fallback+fallback))
        self.chikou_span = np_close
    
    def kaufmans_adaptive_ma(self,efficency_ratio_len=10,fast_ema=2,slow_ema=30):
        """Calculates Kaufman's Adaptive Moving Average as created by Perry Kaufman.

        Parameters
        ----------

        efficency_ratio_len=10: `int`
        The length for the efficency ratio. A higher value is more sensitive to volatility and vise-versa
        
        fast_ema=2: `int`
        The smaller EMA smoothing value. Perry Kaufman recommends a value of 2
        slow_ema=30: `int`
        The larger EMA smoothing value. Perry Kaufman recommends a value of 10
        """
        # np.seterr(all="ignore")
        efficency_ratio =     []
        kama_change =         []
        kama_v_sc =           []
        kama_volatility =     []
        smoothing_const =     []
        first_simple_ma =     []
        kama_one_period_off = []
        self.kaufmans_moving_avg = []
        self.change = np.array(self.change)
        np_close = np.array(self.li_close)
        np_close = np.delete(np_close, np.arange(efficency_ratio_len))
        for x in range(efficency_ratio_len,len(self.li_close),1):
            kama_change.append(abs(self.li_close[x] - self.li_close[x-(efficency_ratio_len)]))
        kama_change = np.array(kama_change)
        kama_volatility = np.abs(self.change)
        kama_volatility = np.insert(kama_volatility, 0, np.ones(1))
        k_vol = []
        self.seperated_close(kama_volatility, 10, k_vol)
        kama_volatility = np.array(np.sum(k_vol[0],axis=1))
        kama_volatility = np.delete(kama_volatility, 0)
        efficency_ratio = kama_change / kama_volatility
        efficency_ratio = np.delete(efficency_ratio, 0)
        
        fema = 2/(fast_ema+1)
        sema = 2/(slow_ema+1)

        kama = []
        smoothing_const = (efficency_ratio * (fema - sema) + sema)**2
        smoothing_const = np.insert(smoothing_const, 0, np.zeros(efficency_ratio_len+1))
        sc = list(smoothing_const)
        kama.append(self.li_close[efficency_ratio_len-1])
        

        for x in range(efficency_ratio_len,len(self.li_close),1):
            kama.append(kama[-1]+sc[x]*(self.li_close[x]-kama[-1]))
        self.kaufmans_moving_avg = np.array(kama)
        self.kaufmans_moving_avg = np.insert(self.kaufmans_moving_avg,0, np.zeros(efficency_ratio_len-1))
        self.kaufmans_moving_avg = np.append(self.kaufmans_moving_avg, np.zeros(self.fallback))
    
    def ema_sma_envelope(self,ema_or_sma,env_length,strength,lower_output_list,upper_output_list):
        """Finds either the EMA or SMA envelopes.

        Parameters
        ----------

        ema_or_sma: `bool`
        If True, will find the EMA envelope, otherwise will find the SMA envelope.

        env_length: `int`
        The length to use for the EMA or SMA envelopes.

        strength: `float | int`
        This is the strength of the envelopes represented in a percentage (eg. 2.5 strength = 2.5% / 100 = 0.025).

        lower_output_list: `list`
        The list you want to store the outputted lower envelope results to.

        upper_output_list: `list`
        The list you want to store the outputted upper envelope results to.
        """
        env_middle = []
        if ema_or_sma:
            self.exponential_ma(self.li_close, env_length, env_middle)
        else:
            self.simple_ma(self.li_close, env_length, env_middle)

        env_upper = []
        env_lower = []
        sma_weighted = []
        env_middle = np.array(env_middle)
        sma_weighted = env_middle * (strength/100)
        env_lower = env_middle - sma_weighted
        env_upper = env_middle + sma_weighted

        if ema_or_sma:
            env_lower = np.insert(env_lower, 0, np.zeros(env_length-2))
            env_upper = np.insert(env_upper, 0, np.zeros(env_length-2))
            env_lower = np.append(env_lower, np.zeros(self.fallback))
            env_upper = np.append(env_upper, np.zeros(self.fallback))
        else:
            env_lower = np.insert(env_lower, 0, np.zeros(env_length-1))
            env_upper = np.insert(env_upper, 0, np.zeros(env_length-1))
            env_lower = np.append(env_lower, np.zeros(self.fallback))
            env_upper = np.append(env_upper, np.zeros(self.fallback))

        lower_output_list.append(env_lower)
        upper_output_list.append(env_upper)





def nan_generator(amount, input_list):
    """ Takes the np.zeros() array and adds it to a non-numpy list
    This very simple function takes an arroy of np.zeros() and puts it into the first values of a non-numpy list.

    Parameters
    ----------
    amount: `int`
        The amount of zeros you need to use.
    input_list: `list`
        The list you want to add the zeros to.
    """
    for x in range(0, amount, 1):
        input_list.insert(0, float("NaN"))




if __name__ == "__main__":
    # Defining fi and the variables inside
    fi = FinanceIndicators()
    fi.fallback = 26
    def zero_after_values(input_list, amount=fi.fallback):
        for x in range(0, amount, 1):
            input_list.append(0)
    fi.ticker = "AMZN"
    fi.sma1_ol = []
    fi.sma2_ol = []
    fi.simple_ma1_length = 50
    fi.simple_ma2_length = 200
    fi.ema1_length = 20
    fi.rsi_ol = []
    stdev = []
    fi.stdev = []
    stddev_ol = []
    senv_l_ol = []
    senv_u_ol = []
    eenv_l_ol = []
    eenv_u_ol = []
    fi.stock_symbol()
    fi.extract_dates()
    fi.future_dates(26)
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
    fi.stdev.append(list(stdev[0]))
    fi.waldo_volatility_indicator()
    fi.ichimoku_cloud()
    fi.kaufmans_adaptive_ma()
    fi.ema_sma_envelope(0, 20, 2.5, senv_l_ol, senv_u_ol)
    fi.ema_sma_envelope(1, 20, 2.5, eenv_l_ol, eenv_u_ol)
    senv_l_ol = np.array(senv_l_ol[0])
    senv_u_ol = np.array(senv_u_ol[0])
    eenv_l_ol = np.array(eenv_l_ol[0])
    eenv_u_ol = np.array(eenv_u_ol[0])

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
    len_of_dates = len(fi.t_df_dates)
    # for x in range(0, 26, 1):
    # fi.t_df_dates.append(len_of_dates+x)
    # zero_after_values(fi.t_df_dates)
    zero_after_values(fi.li_close)
    zero_after_values(fi.d_percent_change)
    zero_after_values(fi.sma1_ol)
    zero_after_values(fi.sma2_ol)
    zero_after_values(macd)
    zero_after_values(macd_9ema)
    zero_after_values(m_histogram)
    zero_after_values(fi.ppo)
    zero_after_values(fi.ppo_sl)
    zero_after_values(fi.ppo_ol)
    zero_after_values(fi.rsi_ol[0])
    zero_after_values(fi.stdev[0])
    zero_after_values(fi.bb_lower)
    zero_after_values(fi.bb_middle)
    zero_after_values(fi.bb_upper)
    # This runs the zip() method and exports the values to a CSV file. Only enabled in stable packages, otherwise commented out.
    # DO NOT DELETE THE BELOW LINES
    all_indicators = zip(fi.t_df_dates, fi.li_close, fi.d_percent_change, fi.sma1_ol,
                         fi.sma2_ol, macd, macd_9ema, m_histogram, fi.ppo, fi.ppo_sl, fi.ppo_ol, fi.rsi_ol[0], fi.stdev[0], fi.bb_lower, fi.bb_middle, fi.bb_upper, fi.waldo_vola_indicator, fi.tenkan_sen, fi.kijun_sen, fi.senkou_a, fi.senkou_b, fi.chikou_span, fi.kaufmans_moving_avg,senv_l_ol,senv_u_ol,eenv_l_ol,eenv_u_ol)
    ai_df = pd.DataFrame(all_indicators, columns=pd.MultiIndex.from_tuples([
        ("Timestamp", "Timestamp"), ("Close", "Close"), ("Daily Percent Change", "Daily Percent Change"), ("Moving Averages", "50-Day MA"), ("Moving Averages", "200-Day MA"), ("MACD", "MACD"), 
        ("MACD", "MACD Signal Line"), ("MACD", "MACD Histogram"), ("PPO", "PPO"), ("PPO", "PPO Signal Line"), ("PPO", "PPO Histogram"), ("RSI", "RSI"), ("Standard Deviation", "Standard Deviation"), 
        ("Bollinger Bands", "Lower Bollinger Band"), ("Bollinger Bands", "Middle Bollinger Band"), ("Bollinger Bands", "Upper Bollinger Band"), ("Waldo Volatility Indicator", "Waldo Volatility Indicator"), 
        ("Ichimoku Clouds", "Tenkan Sen"), ("Ichimoku Clouds", "Kijun Sen"), ("Ichimoku Clouds", "Senkou A"), ("Ichimoku Clouds", "Senkou B"), ("Ichimoku Clouds", "Chikou Span"), 
        ("Kaufman's Adaptive Moving Average", "KAMA"),("Simple Moving Average Enevlopes"," Lower SMA Envelope"),("Simple Moving Average Enevlopes"," Higher SMA Envelope"),("Exponential Moving Average Enevlopes"," Lower EMA Envelope"),
        ("Exponential Moving Average Enevlopes"," Higher EMA Envelope")]))
    # os.chdir("..")
    print(os.getcwd())
    ai_df.to_csv(
        f'CSVinfo/{fi.ticker} {time()}.csv', index=False)
