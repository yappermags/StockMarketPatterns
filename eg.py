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


arr = [1, 2, 3, 7, 9]
window_size = 3


i = 0
# Initialize an empty list to store moving averages
moving_averages = []

# Loop through the array to consider
# every window of size 3
while i < len(arr) - window_size + 1:
	

  
	# Store elements from i to i+window_size
	# in list to get the current window
	window = arr[i : i + window_size]

	# Calculate the average of current window
	window_average = round(sum(window) / window_size, 2)
	
	# Store the average of current
	# window in moving average list
	moving_averages.append(window_average)
	
	# Shift window to right by one position
	i += 1

print(moving_averages)


