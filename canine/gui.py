import numpy as np
import matplotlib.pyplot as plt
import finance_indicators
fi = finance_indicators.FinanceIndicators()
short_day_ma = np.array([])
long_day_ma = []
fi.ticker = "AMZN"
fi.stock_symbol()
fi.percent_change()
short_day_ma = fi.simple_ma(fi.li_close, 50)
short_day_ma = np.insert(short_day_ma,0,np.zeros(49))
long_day_ma = fi.simple_ma(fi.li_close, 200)
long_day_ma = np.insert(long_day_ma,0,np.zeros(249))

plt.plot([fi.t_df_dates],[short_day_ma],"ro")

plt.show()
