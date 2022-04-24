# Stock Market Patterns

I am using this tool to attempt to find some common stock market patterns which will either end up in the price going up long-term or going down long-term.  

## Currently Tracked Trading Tools
Here is a quick list of what is currently tracked in the FinanceIndicators() class, and what they are:
- Closing Prices
    - This extracts what the closing price is for a set time (usually max for most accuracy) and turns it into a list. This makes it easier to extract the actual values we use in the program. Called with fi.extract_cprices()
- Percentage Change
    - This one is pretty simple. Daily % change = (today's price - yesterday's price) / yesterday's price * 100. If I needed to do anything with the percentage change, I would remove the (* 100) at the end to make it easier. Called with fi.percent_change()
- Simple Moving Average
    - I figured how to write the code for this one from [geeksforgeeks.com](https://www.geeksforgeeks.com). A simple moving average is the sum of the number of days you want to cover (the average timeframe or window) / the timeframe or window. Called with fi.simple_ma()
- Exponential Moving Average
    - I used the template for the simple_ma to loop through the timeframe, but the formula came from [stockcharts.com](https://school.stockcharts.com/doku.php?id=technical_indicators:moving_averages). The formula for this is really complicated and better explained in the stockcharts link. This is called with fi.exponential_ma()

## Version History  
> There has not been any released versions yet, but stay tuned as they should come soon!

