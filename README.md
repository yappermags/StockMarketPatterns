# Canine

I am creating this tool to attempt to find some common stock market patterns which will either end up in the price going up long-term or going down long-term.  

## Currently Tracked Trading Tools

### Required Methods to Run
> These methods and variables are required in order for the other indicators to work  

- `fi.ticker` = `str`
    - fi.ticker is the ticker you want the program to run

- `fi.stock_symbol()`
    - fi.stock_symbol is what downloads the stock data for the stock you selected.

Here is a quick list of what is currently tracked in the FinanceIndicators() class, and what they are:  

> If you see ^ next to a method parameter, it means that it is not required to provide a value, and probably better off not being changed in most cases.  

- Closing Prices
    - This extracts what the closing price is for a set time (usually max for most accuracy) and turns it into a list. This makes it easier to extract the actual values we use in the program. 
    - Called with `fi.extract_cprices()`
- Percentage Change
    - This one is pretty simple. Daily % change = (today's price - yesterday's price) / yesterday's price * 100. If I needed to do anything with the percentage change, I would remove the (* 100) at the end to make it easier. 
    - Called with `fi.percent_change()`
- Simple Moving Average
    - I figured how to write the code for this one from [geeksforgeeks.com](https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/). A simple moving average is the sum of the number of days you want to cover (the average timeframe or window) / the timeframe or window. 
    - Called with `fi.simple_ma(prices, period, output_list)`
- Exponential Moving Average
    - I used the template for the simple_ma to loop through the timeframe, but the formula came from [stockcharts.com](https://school.stockcharts.com/doku.php?id=technical_indicators:moving_averages). The formula for this is really complicated and better explained in the stockcharts link. 
    - Called with `fi.exponential_ma(prices, period, output_list)`
- MACD
    - The MACD by itself is calculated by subracting the 12-period EMA from the 26-period EMA. The MACD signal line is the 9-day EMA of the MACD. A MACD histogram is the MACD subtracted from the signal line.
    - Called with `fi.macd_whistogram(macd12^, macd24^, macd9^)`
- Relative Strength Index (RSI)
    - RSI is a very complicated indicator considering what we have done up to this point. RSI works by calculating an average gain / average loss = RS, then 100 - (100/RS) to get RSI. For the first average, it is simply the mean of the first 14 numbers. After that, it is ((last avg. gain * 13 + current gain) / 14) / ((last avg. loss * 13 + current loss) / 14). THen RS and RSI are the same.
    - Called with `fi.rsi(length^)`
- Percentage Price Oscillator (PPO)
    - PPO is considered the cousin of the MACD, because it is similar. PPO is calculated by taking the MACD, and dividing it by 26. To get a percentage, you multiply your answer by 100. PPO also has a Signal Line (9-day EMA of PPO), and a histogram (PPO - PPO Signal Line)
    - Called with `fi.ppo_whistogram()`
- Standard Deviation
    - Standard Deviation is one of the ways people attempt to measure volatility. To figure out the Standard Deviation, you need to calculate the mean of your population. You then subtract the mean from the population. Then you square each of the answers. After you square the answers, you find the mean of those answers. You square root the mean, and there is your Standard Deviation.  
    - Note that Standard Deviation is very memory intensive at the moment, especially for companies that have been on the stock market for a long time.  
    - Called with `fi.standard_deviation(input_list, output_list, sd_length=10)`

- Bollinger Bands&reg;
    - Bollinger Bands&reg; as created and teademarked by John Bollinger, are three sets of lines used to measure a secruities expected potential. The further the lines are from each other, the more volatile the market is, and vise-versa.
    - Called with `fi.bollinger_bands(bb_length=20^, multiplier=2^)`

- Ichimoku Clouds
    - Ichimoku clouds are 5 sets of trackers that measure the future potential of a stock. The Tenkan Sen is an average of the 9-Day High High and the Low Low. The Kijun Sen is an average of the 26-Day High High and Low Low. The Senkou A is an average of the Tenkan and Kijun Sen, and is set 26 days into the future. The Senkou B is the 52-Day High High and Low Low, and also set 26 days in the future. Finally, the Chikou Span is the close set 26 days behind.  
    - Called with `fi.ichimoku_clouds(tenkan_sen_len=9^, kijun_sen_len=26^, senkou_b_len=52^, fallback=26^)`

- Kaufman's Adaptive Moving Average
    - K


## Coming up

#### Indicators
In the future, we plan to release Money Flow Index, Chaikin Money Flow, Ulcer Index, Chaikin Oscillator  

#### Overlays
In the future, I plan to release ZigZag, ZigZag (Retrace), and the SMA and EMA Envelopes.  
