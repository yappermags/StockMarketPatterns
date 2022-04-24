# import pandas as pd

# kes = ['1', '2', '3']
# vlues = ["one", "two", "three"]
# vlues2 = ["four", "five", "six"]

# d = zip(kes, vlues,vlues2)
# # dd = dict(d)


# # print(dd)
# df = pd.DataFrame.from_dict(d)
# print(df)
# dfd = df.to_dict('list')
# print(dfd)


# Program to calculate moving average
from math import prod
from re import L


# arr = [1, 2, 3, 7, 9]
# window_size = 3


# i = 0
# # Initialize an empty list to store moving averages
# moving_averages = []

# # Loop through the array to consider
# # every window of size 3
# while i < len(arr) - window_size + 1:


# 	# Store elements from i to i+window_size
# 	# in list to get the current window
# 	window = arr[i : i + window_size]

# 	# Calculate the average of current window
# 	window_average = round(sum(window) / window_size, 2)

# 	# Store the average of current
# 	# window in moving average list
# 	moving_averages.append(window_average)

# 	# Shift window to right by one position
# 	i += 1

# print(moving_averages)

price = [22.2734, 22.1940, 22.0847, 22.1741, 22.1840, 22.1344, 22.2337, 22.4323, 22.2436, 22.2933, 22.1542, 22.3926, 22.3816, 22.6109, 23.3558, 24.0519, 23.7530, 23.8324, 23.9516, 23.6338, 23.8225, 23.8722, 23.6537, 23.1870, 23.0976, 23.3260, 22.6805, 23.0976, 22.4025, 22.1725]
window_size = 10


def exponential_ma():
    """
    Calculation for a 20-day exponential moving average is:

    Multiplier = (2 / (time period + 1)) 

    EMA = (Close - EMA(previous day)) * multiplier + EMA (previous day)
    """
    multiplier = (2 / (window_size+1))
    # print(multiplier)
    i = 0
    e_moving_averages = []
    e_fv = sum(price[0:window_size]) / window_size
    e_moving_averages.append(e_fv)
    while i < len(price) - window_size+1:
        window = price[i:i+window_size]
        e_window_average = (
            window[-1] - e_moving_averages[-1]) * multiplier + e_moving_averages[-1]
        e_moving_averages.append(e_window_average)
        i += 1
    print(e_moving_averages)


exponential_ma()
