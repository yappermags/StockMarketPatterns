"""
This is a program that attempts to find common bullish and bearish patterns of stocks
"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import csv

thistory = {}

ticker = "AMZN"
icker = yf.Ticker(ticker)
exporter = yf.download(ticker, period="ytd")
# earnings = icker.calendar

t_df = pd.DataFrame(data=exporter)
ticker_df = t_df.to_dict()
# print(exporter)
keys_tr = ("Open", "High", "Low", "Adj Close", "Volume")
# print(ticker_df)

# print(datetime.now())
# print(datetime(2021,1,1)+timedelta(1))
# print(datetime(2022,1,1).weekday())
print(timedelta())


def trading_days(year):
    """
    This is a function used to find a whole year, then remove all the weekends from it (Note that STAT holidays are not removed as of right now)

    year -> Year you want to find (int only)
    """
    if int(year) == False:
        raise NameError("Only integers are acceptable here")
    else:
        # weekends = []
        trading_list = []
        trading_list.append(datetime(year, 1, 1))
        for x in range(1, 365, 1):
            eg = pd.Timestamp(datetime(year, 1, 1)+timedelta(x))
            trading_list.append(eg)
        print(len(trading_list))
        for y in range(0, 313, 1):
            if trading_list[y].weekday() == 5:
                trading_list.pop(y)
        for y in range(0, 260, 1):
            if trading_list[y].weekday() == 6:
                trading_list.pop(y)
        print(trading_list)


trading_days(2021)


def pop_multiple(var, *args):
    """
    var -> The variable you want to pop the items or keys

    *args -> The items or keys you want to pop
    """
    args = args[0]
    iter(list(args))
    for x in args:
        var.pop(x)


pop_multiple(ticker_df, keys_tr)
# print(ticker_df)


print(ticker_df["Close"])



# percent_change = (eg[3]-eg[0])/eg[0]
# print(percent_change)
# eg.append(percent_change)

# class PcPatterns():
#     """
#     This is the class which I am going to use to run all my programs
#     """
#     def __init__(self,rub=1) -> None:
#         self.rub = rub

#     def pc_history(self):
#         return self.rub * 3.14
#         pass
# pc = PcPatterns()
# print(pc.rub)
# exx = pc.rub = 10
# pc.pc_history()
