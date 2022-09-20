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

    def __init__(self, ticker=None, sma1_ol=None, sma2_ol=None, simple_ma1_length=None, simple_ma2_length=None, ema1_ol=None, ema2_ol=None, ema1_length=None, ema2_length=None, rsi_ol=None, stadev=None, fallback=None):
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
        self.stadev = stadev
        self.fallback = fallback
        # self.simple_ma3_length = simple_ma3_length

    def stock_symbol(self):
        # stock_symbol
        """
        Gets the stock symbol and MAX data from the yfinance library
        """
        thistory = {}
        self.ticker_info = yf.Ticker(self.ticker).info
        exporter = yf.download(self.ticker, period="max")

        self.t_df = pd.DataFrame(data=exporter)
        self.ticker_df = self.t_df.to_dict()

        self.d_percent_change = []
        self.change = []

        self.t_df_dates = list(self.ticker_df["Close"].keys())
        self.li_open = []
        self.li_high = []
        self.li_low = []
        self.li_close = []
        self.li_open = np.array(list(self.ticker_df["Open"].values()))
        self.li_high = np.array(list(self.ticker_df["High"].values()))
        self.li_low = np.array(list(self.ticker_df["Low"].values()))
        self.li_close = np.array(list(self.ticker_df["Close"].values()))
        self.volume = np.array(list(self.ticker_df["Volume"].values()))



    def future_dates(self, dates_ahead):
        ticker_length = len(self.ticker)
        wd = datetime.weekday(self.t_df_dates[-1])
        first_date_calc = str(self.t_df_dates[-1])
        first_date_calc = first_date_calc.replace(' 00:00:00', '')
        delta = timedelta(days=round((self.fallback*2), 0))
        final_date = str(delta + date.fromisoformat(first_date_calc))
        if wd == 4:
            first_date_calc = date.fromisoformat(first_date_calc) + timedelta(days=3)
        else:
            first_date_calc = date.fromisoformat(first_date_calc) + timedelta(days=1)

        if self.ticker_info["country"] == "Canada":
            tsx = mcal.get_calendar('TSX')
            fd = tsx.schedule(start_date=first_date_calc, end_date=final_date)
            self.new_dates = list(fd.market_open._stat_axis)
            for x in range(0, len(self.new_dates)-dates_ahead, 1):
                self.new_dates.pop()
        elif self.ticker_info["country"] == "United States":
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

    def percent_change(self):
        """
        This class method is what gives us the percentage change information required to somewhat judge a stock

        How % change works:

        ((today's price - yesterday's price)/yesterday's price)*100 (to return a percentage) 
        """

        self.change = self.li_close[1::] - self.li_close[:-1]
        self.d_percent_change = (self.change/self.li_close[:-1])*100

    def seperated_close(self, sc_prices, sc_length):
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
        sc_lc = int(len(sc_prices)-(sc_length-1))
        sc = np.empty((0, 1), float)
        for x in range(0, sc_lc, 1):
            window = sc_prices[x:x+sc_length]
            sc = np.append(sc, window)
        return sc.reshape(int(len(sc)/sc_length), sc_length)

    def simple_ma(self, prices, length):
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

        timeframe = self.seperated_close(prices, length)

        return np.sum(timeframe, 1) / length

    def exponential_ma(self, prices, length, smooth_const=2):
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
        output_list = []
        multiplier = (smooth_const / (length+1))
        output_list.append(sum(prices[0:length]) / length)
        timeframe = self.seperated_close(prices, length)
        list_close = np.array(prices)
        list_close = np.delete(list_close, np.arange(length))
        list_close = list(list_close)
        len_lc = len(list_close)
        for x in range(0, len(list_close), 1):
            e_window_average = (
                list_close[x] - output_list[x]) * multiplier + output_list[x]
            # output_list.append(e_window_average)
            output_list = np.append(output_list, e_window_average)
        return output_list

    def macd_whistogram(self, macd_short=12, macd_long=26, macd_sl=9):
        """
        The MACD is a calculation of the 12-period EMA - the 26-period EMA. A more advanced calculation (MACD histogram) uses a signal line, which is the 9-period EMA of the MACD itself, which is subtracted by the original MACD value.
        """
        self.m_histogram = []
        macd12 = []
        macd26 = []
        self.macd = []
        self.macd_9ema = []
        macd12 = self.exponential_ma(self.li_close, macd_short)
        macd26 = self.exponential_ma(self.li_close, macd_long)
        self.macd = macd12[14:] - macd26
        self.macd_9ema = self.exponential_ma(self.macd, macd_sl)

        self.m_histogram = self.macd[int(macd_sl-1):] - self.macd_9ema

    def ppo_whistogram(self, ppo_short=12, ppo_long=26, ppo_sl=9):
        """
        ppo_whistogram() -> Finds the Percentage Price Oscillator for a security.

        To find Percentage Price Oscillator, you take the MACD value above, divide it by the longer term MACD value, than multiply that value by 100 (to get percentage)

        Parameters
        ----------
        None
        """
        self.ppo_ol = []
        self.ppo = []
        self.ppo_hist = []
        output_list = []
        macd26 = []
        macd26 = self.exponential_ma(self.li_close, ppo_long)
        self.ppo = (self.macd / macd26) * 100
        self.ppo_sl = self.exponential_ma(self.ppo, ppo_sl)
        self.ppo_hist = self.ppo[int(ppo_sl-1):] - self.ppo_sl

    def rsi(self, length=14):
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
        seperated_change = []
        avg_positive_changes = []
        avg_negative_changes = []
        rs = []
        rsi = []
        for x in np.nditer(self.change[0:length]):
            if x >= 0:
                avg_positive_changes = np.append(avg_positive_changes,x)
            else:
                avg_negative_changes = np.append(avg_negative_changes,np.abs(x))
        avg_positive_changes = np.append([], np.sum(avg_positive_changes) / length)
        avg_negative_changes = np.append([],np.sum(avg_negative_changes) / length)
        for x in np.nditer(self.change[length-1:]):
            if x >= 0:
                avg_positive_changes = np.append(avg_positive_changes,(avg_positive_changes[-1] *  (length-1) + x)/length)
                avg_negative_changes = np.append(avg_negative_changes,(avg_negative_changes[-1] * (length-1) + 0)/length)
            else:
                avg_positive_changes = np.append(avg_positive_changes,(avg_positive_changes[-1] * (length-1) + 0)/length)
                avg_negative_changes = np.append(avg_negative_changes,(avg_negative_changes[-1] * (length-1) + np.abs(x))/length)
        rs = avg_positive_changes / avg_negative_changes
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def standard_deviation(self, input_list, sd_length=10):
        """Calculates the Standard Deviation measurement used for measuring volatility

        Parameters
        ----------
        input: `list`
        The input list of daily closing prices. Default: fi.li_close

        output: `list`
        The list you want to extract the standard deviation values to. Default: fi.stadev

        sd_length: `int`
        The time window you want 

        Notes
        -----
        `self.stadev` is used for running inside other 

        """
        seperated_prices = []
        sma = []
        sma_t10 = []
        deviation = []
        dsq_ma = []
        std_dev = []

        seperated_prices = self.seperated_close(self.li_close, sd_length)
        sma = np.mean(seperated_prices,axis=1)
        sma = sma.reshape(len(sma),1)
        deviation_squared = (seperated_prices - sma)**2
        d_sq_avg = np.mean(deviation_squared,axis=1)
        return np.sqrt(d_sq_avg)

    def bollinger_bands(self, bb_length=20, multiplier=2):
        """This gathers the bollinger bands as created by John Bollinger

        Notes

        """
        stddev = []
        bb_stddev = []
        self.bb_middle = []
        self.bb_upper = []
        self.bb_lower = []
        self.bb_middle = self.simple_ma(self.li_close, bb_length)
        stddev = self.standard_deviation(self.li_close, bb_length)
        bb_stddev = stddev*multiplier
        self.bb_lower = self.bb_middle - bb_stddev
        self.bb_upper = self.bb_middle + bb_stddev
        self.bbw = ((self.bb_upper - self.bb_lower) / self.bb_middle) * 100
        # copy_close = self.np_close.copy()
        # copy_close = np.delete(copy_close, np.arange(bb_length-1))
        self.percent_b = (
            self.li_close[(bb_length-1):] - self.bb_lower) / (self.bb_upper - self.bb_lower)
        self.bbw = np.insert(self.bbw, 0, np.zeros(bb_length-1))
        self.bbw = np.append(self.bbw, np.zeros(self.fallback))
        self.percent_b = np.insert(self.percent_b, 0, np.zeros(bb_length-1))
        self.percent_b = np.append(self.percent_b, np.zeros(self.fallback))

    def waldo_volatility_indicator(self, wvi_length=28):
        """Measures volatility and works a lot like the Ulcer Index.

        Parameters
        ----------

        wvi_length: `int`
        The length of time you want to measure the Waldo Volatility Indicator

        """
        seperated_close = np.empty((0, 1), float)
        self.waldo_vola_indicator = []
        seperated_close = self.seperated_close(self.li_close, wvi_length)
        mp = np.amax(seperated_close, axis=1)
        max_prices = np.repeat(mp, wvi_length)
        max_prices = max_prices.reshape(
            int(len(max_prices)/wvi_length), wvi_length)
        price_dropdown = np.abs(seperated_close - max_prices)
        pd_mean = np.mean(price_dropdown, axis=1)
        self.waldo_vola_indicator = np.sqrt(pd_mean)
        self.waldo_vola_indicator = np.insert(
            self.waldo_vola_indicator, 0, np.zeros(wvi_length-1))
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
        ts_sc_min = self.seperated_close(self.li_low, tenkan_sen_len)
        ts_sc_max = self.seperated_close(self.li_high, tenkan_sen_len)
        ks_sc_min = self.seperated_close(self.li_low, kijun_sen_len)
        ks_sc_max = self.seperated_close(self.li_high, kijun_sen_len)
        se_b_min = self.seperated_close(self.li_low, senkou_b_len)
        se_b_max = self.seperated_close(self.li_high, senkou_b_len)

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
        np_close = self.li_close.copy()
        np_close = np.delete(np_close, np.arange(fallback))
        np_close = np.append(np_close, np.zeros(fallback+fallback))
        self.chikou_span = np_close

    def kaufmans_adaptive_ma(self, efficency_ratio_len=10, fast_ema=2, slow_ema=30):
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
        efficency_ratio = []
        kama_change = []
        kama_v_sc = []
        kama_volatility = []
        smoothing_const = []
        first_simple_ma = []
        kama_one_period_off = []
        self.kaufmans_moving_avg = []
        self.change = np.array(self.change)
        np_close = np.array(self.li_close)
        np_close = np.delete(np_close, np.arange(efficency_ratio_len))
        # for x in range(efficency_ratio_len, len(self.li_close), 1):
        kama_change.append(
            abs(self.li_close[:-efficency_ratio_len] - self.li_close[efficency_ratio_len:]))
        kama_change = np.array(kama_change)
        kama_volatility = np.abs(self.change)
        kama_volatility = np.insert(kama_volatility, 0, np.ones(1))
        k_vol = []
        k_vol = self.seperated_close(kama_volatility, 10)
        kama_volatility = np.array(np.sum(k_vol, axis=1))
        kama_volatility = np.delete(kama_volatility, 0)
        efficency_ratio = kama_change / kama_volatility
        efficency_ratio = np.delete(efficency_ratio, 0)

        fema = 2/(fast_ema+1)
        sema = 2/(slow_ema+1)

        kama = []
        smoothing_const = (efficency_ratio * (fema - sema) + sema)**2
        smoothing_const = np.insert(
            smoothing_const, 0, np.zeros(efficency_ratio_len+1))
        sc = list(smoothing_const)
        kama.append(self.li_close[efficency_ratio_len-1])

        for x in range(efficency_ratio_len, len(self.li_close), 1):
            kama.append(kama[-1]+sc[x]*(self.li_close[x]-kama[-1]))
        self.kaufmans_moving_avg = np.array(kama)
        self.kaufmans_moving_avg = np.insert(
            self.kaufmans_moving_avg, 0, np.zeros(efficency_ratio_len-1))
        self.kaufmans_moving_avg = np.append(
            self.kaufmans_moving_avg, np.zeros(self.fallback))

    def ema_sma_envelope(self, ema_or_sma, env_length, strength):
        """Finds either the EMA or SMA envelopes.

        Parameters
        ----------

        ema_or_sma: `bool`
        If True, will find the EMA envelope, otherwise will find the SMA envelope.

        env_length: `int`
        The length to use for the EMA or SMA envelopes.

        strength: `float | int`
        This is the strength of the envelopes represented in a percentage (eg. 2.5 strength = 2.5% / 100 = 0.025).

        Returns
        -------

        lower_output_list: `np.ndarray`
        The extracted values for the lower SMA or EMA envelope.

        uppper_output_list: `np.ndarray`
        The extracted values for the upper SMA or EMA envelope.
        """
        env_middle = []
        if ema_or_sma:
            env_middle = self.exponential_ma(self.li_close, env_length)
        else:
            env_middle = self.simple_ma(self.li_close, env_length)

        upper_output_list = []
        lower_output_list = []
        sma_weighted = []
        sma_weighted = env_middle * (strength/100)
        lower_output_list = env_middle - sma_weighted
        upper_output_list = env_middle + sma_weighted

        lower_output_list = np.insert(
            lower_output_list, 0, np.zeros(env_length-1))
        upper_output_list = np.insert(
            upper_output_list, 0, np.zeros(env_length-1))
        lower_output_list = np.append(
            lower_output_list, np.zeros(self.fallback))
        upper_output_list = np.append(
            upper_output_list, np.zeros(self.fallback))

        return lower_output_list, upper_output_list
    




if __name__ == "__main__":
    # Defining fi and the variables inside
    fi = FinanceIndicators()
    fi.fallback = 26
    fi.ticker = "AMZN"
    fi.sma1_ol = []
    fi.sma2_ol = []
    fi.simple_ma1_length = 50
    fi.simple_ma2_length = 200
    fi.ema1_length = 20
    fi.rsi_ol = []
    fi.stadev = []
    senv_l_ol = []
    senv_u_ol = []
    eenv_l_ol = []
    eenv_u_ol = []
    fi.stock_symbol()
    fi.future_dates(fi.fallback)
    fi.percent_change()
    fi.sma1_ol = fi.simple_ma(fi.li_close, fi.simple_ma1_length)
    fi.sma2_ol = fi.simple_ma(fi.li_close, fi.simple_ma2_length)
    fi.macd_whistogram()
    fi.ppo_whistogram()
    fi.rsi_ol = fi.rsi()
    fi.stadev = fi.standard_deviation(fi.li_close)
    fi.bollinger_bands()
    fi.waldo_volatility_indicator()
    fi.ichimoku_cloud()
    fi.kaufmans_adaptive_ma()
    fi.accumulation_distribution_line()
    senv_l_ol, senv_u_ol = fi.ema_sma_envelope(0, 20, 2.5)
    eenv_l_ol, eenv_u_ol = fi.ema_sma_envelope(1, 20, 2.5)
    fi.balance_of_power()
    fi.chaikin_money_flow()
    fi.ease_of_movement()
    fi_three_teen = fi.force_index(13)
    # fi.aroon()

    fi.d_percent_change = np.insert(fi.d_percent_change, 0, np.zeros(1))
    fi.sma1_ol = np.insert(fi.sma1_ol, 0, np.zeros(fi.simple_ma1_length-1))
    fi.sma2_ol = np.insert(fi.sma2_ol, 0, np.zeros(fi.simple_ma2_length-1))
    fi.macd = np.insert(fi.macd, 0, np.zeros(len(fi.li_close) - len(fi.macd)))
    fi.macd_9ema = np.insert(fi.macd_9ema, 0, np.zeros(
        len(fi.li_close) - len(fi.macd_9ema)))
    fi.m_histogram = np.insert(fi.m_histogram, 0, np.zeros(
        len(fi.li_close) - len(fi.m_histogram)))
    fi.ppo = np.insert(fi.ppo, 0, np.zeros(len(fi.li_close) - len(fi.ppo)))
    fi.ppo_sl = np.insert(fi.ppo_sl, 0, np.zeros(
        len(fi.li_close) - len(fi.ppo_sl)))
    fi.ppo_hist = np.insert(fi.ppo_hist, 0, np.zeros(
        len(fi.li_close) - len(fi.ppo_hist)))
    fi.rsi_ol = np.insert(fi.rsi_ol, 0, np.zeros(
        len(fi.li_close) - len(fi.rsi_ol)))
    fi.stadev = np.insert(fi.stadev, 0, np.zeros(
        len(fi.li_close) - len(fi.stadev)))
    fi.bb_lower = np.insert(fi.bb_lower, 0, np.zeros(
        len(fi.li_close) - len(fi.bb_lower)))
    fi.bb_middle = np.insert(fi.bb_middle, 0, np.zeros(
        len(fi.li_close) - len(fi.bb_middle)))
    fi.bb_upper = np.insert(fi.bb_upper, 0, np.zeros(
        len(fi.li_close) - len(fi.bb_upper)))
    fi_three_teen = np.insert(fi.bb_upper, 0, np.zeros(len(fi.li_close) - len(fi_three_teen)))

    fi.li_open = np.append(fi.li_close, np.zeros(fi.fallback))
    fi.li_high = np.append(fi.li_close, np.zeros(fi.fallback))
    fi.li_low = np.append(fi.li_close, np.zeros(fi.fallback))
    fi.li_close = np.append(fi.li_close, np.zeros(fi.fallback))
    fi.d_percent_change = np.append(fi.d_percent_change, np.zeros(fi.fallback))
    fi.sma1_ol = np.append(fi.sma1_ol, np.zeros(fi.fallback))
    fi.sma2_ol = np.append(fi.sma2_ol, np.zeros(fi.fallback))
    fi.macd = np.append(fi.macd, np.zeros(fi.fallback))
    fi.macd_9ema = np.append(fi.macd_9ema, np.zeros(fi.fallback))
    fi.m_histogram = np.append(fi.m_histogram, np.zeros(fi.fallback))
    fi.ppo = np.append(fi.ppo, np.zeros(fi.fallback))
    fi.ppo_sl = np.append(fi.ppo_sl, np.zeros(fi.fallback))
    fi.ppo_hist = np.append(fi.ppo_hist, np.zeros(fi.fallback))
    fi.rsi_ol = np.append(fi.rsi_ol, np.zeros(fi.fallback))
    fi.stadev = np.append(fi.stadev, np.zeros(fi.fallback))
    fi.bb_lower = np.append(fi.bb_lower, np.zeros(fi.fallback))
    fi.bb_middle = np.append(fi.bb_middle, np.zeros(fi.fallback))
    fi.bb_upper = np.append(fi.bb_upper, np.zeros(fi.fallback))
    fi_three_teen = np.append(fi_three_teen, np.zeros(fi.fallback))
    # This runs the zip() method and exports the values to a CSV file. Only enabled in stable packages, otherwise commented out.
    # DO NOT DELETE THE BELOW LINES
    all_indicators = zip(fi.t_df_dates, fi.li_open, fi.li_high, fi.li_low, fi.li_close, fi.d_percent_change, fi.sma1_ol,
                         fi.sma2_ol, fi.macd, fi.macd_9ema, fi.m_histogram, fi.ppo, fi.ppo_sl, fi.ppo_hist, fi.rsi_ol, fi.stadev, fi.bb_lower, fi.bb_middle, fi.bb_upper, fi.bbw, fi.percent_b, fi.waldo_vola_indicator, fi.tenkan_sen, fi.kijun_sen, fi.senkou_a, fi.senkou_b, fi.chikou_span, fi.kaufmans_moving_avg, senv_l_ol, senv_u_ol, eenv_l_ol, eenv_u_ol,fi.bop,fi.cmf, fi_three_teen)
    ai_df = pd.DataFrame(all_indicators, columns=pd.MultiIndex.from_tuples([("Timestamp", "Timestamp"),
                                                                            ("Open","Open"), ("High","High"), ("Low","Low"), ("Close", "Close"), ("Daily Percent Change", "Daily Percent Change"), (
        "Moving Averages", "50-Day MA"), ("Moving Averages", "200-Day MA"), ("MACD", "MACD"),
        ("MACD", "MACD Signal Line"), ("MACD", "MACD Histogram"), ("PPO", "PPO"), ("PPO",
                                                                                   "PPO Signal Line"), ("PPO", "PPO Histogram"), ("RSI", "RSI"), ("Standard Deviation", "Standard Deviation"),
        ("Bollinger Bands", "Lower Bollinger Band"), ("Bollinger Bands", "Middle Bollinger Band"), (
            "Bollinger Bands", "Upper Bollinger Band"), ("Bollinger Band Indicators", "Bollinger BandWidth"),
        ("Bollinger Band Indicators", "%B"), ("Waldo Volatility Indicator", "Waldo Volatility Indicator"), (
            "Ichimoku Clouds", "Tenkan Sen"), ("Ichimoku Clouds", "Kijun Sen"), ("Ichimoku Clouds", "Senkou A"),
        ("Ichimoku Clouds", "Senkou B"), ("Ichimoku Clouds", "Chikou Span"), ("Kaufman's Adaptive Moving Average",
                                                                              "KAMA"), ("Simple Moving Average Enevlopes", " Lower SMA Envelope"),
        ("Simple Moving Average Enevlopes", " Higher SMA Envelope"), ("Exponential Moving Average Enevlopes", " Lower EMA Envelope"), ("Exponential Moving Average Enevlopes", " Higher EMA Envelope"),("Balance of Power","Balance of Power"),
        ("Chaikin Money Flow","Chaikin Money Flow"),("Force Index","Thirteen-Day Force Index")]))

    # os.chdir("c:/Users/magnu/.vscode/extensions/ms-python.python-2022.6.2/pythonFiles/lib/python/debugpy/launcher")
    print(os.getcwd())
    ai_df.to_csv(
        f'CSVinfo/{fi.ticker} {time()}.csv', index=False)
