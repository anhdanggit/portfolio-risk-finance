'''
#FILE: TIME-SERIES in PYTHON
Project: portfolio-risk-finance
-------------------
By: Anh Dang
Date: 2019-07-23
Description:
Overview some basic tasks with timeseries in Python
'''

## Import modules
import pandas as pd
import numpy as np
import os
import scipy as sp
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse ## parse most human-intelligible date (automatic)
from pandas.tseries.offsets import Hour, Minute, Day, MonthEnd, MonthBegin 
from scipy.stats import percentileofscore
import pytz


## 1 - Data Types and Basic Tools -------
now = datetime.now()
now 
now.year, now.month, now.day

delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
delta 
delta.days 
delta.seconds

start = datetime(2011, 1, 7)
start + timedelta(12)
start - 2 * timedelta(12)


## 2 - Converting between String and Datetime -------
stamp = datetime(2011, 1, 3)
str(stamp)
stamp.strftime('%Y-%m-%d')

value = '2011-01-03'
datetime.strptime(value, '%Y-%m-%d')

datestrs = ['7/6/2011','8/6/2011']
[datetime.strptime(x, '%m/%d/%Y') for x in datestrs]

## Auto parsing
parse('Jan 31, 1997 10:45 PM')
parse('6/12/2011', dayfirst=True)

## by pandas
datestrs = ['2011-07-06 12:00:00', '2011-08-06 00:00:00']
pd.to_datetime(datestrs)

idx = pd.to_datetime(datestrs + [None]) ## unable to parse, return as NaT
pd.isnull(idx)


## 3 - Time-Series Basics -------
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
        datetime(2011, 1, 7), datetime(2011, 1, 8),
        datetime(2011, 1, 10), datetime(2011, 1, 12)]

ts = pd.Series(np.random.randn(6), index=dates)
ts 
ts.index 
ts + ts[::2] ## auto align on the date index
ts.index.dtype 
ts.index[0]

#### Indexing, selection, subsetting
ts['2011-01-02'] ##index by a string in date format

longer_ts = pd.Series(np.random.randn(1000), 
                        index=pd.date_range('2000-01-01', periods=1000))
longer_ts['2001'] ##index by a string with year
longer_ts['2001-05']
longer_ts[datetime(2001, 5, 1):datetime(2001, 5, 20)]
ts.truncate(after='2011-09-01')

dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
long_df = pd.DataFrame(np.random.randn(100, 4),
                        index=dates,
                        columns=['Colorado', 'Texas','New York', 'Ohio'])
long_df.loc['2001-05']

#### duplicated date index
dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000','1/2/2000', '1/3/2000'])
dup_ts = pd.Series(np.arange(5), index=dates)
dup_ts.index.is_unique
dup_ts['2000-01-02']
## aggregate dup
dup_ts.groupby(level=0).sum()


## 3 - Date Ranges, Frequencies, Shifting -------
index = pd.date_range('2012-04-01', '2012-06-01') ##Day freq.
pd.date_range(start='2012-04-01', periods=20) ##Day freq.
pd.date_range('2000-01-01','2000-12-01', freq='BM') ## check, MS, M, etc.

pd.date_range('2012-05-02 12:56:31', periods=5)
pd.date_range('2012-05-02 12:56:31', periods=5, normalize=True) ##normalize to midnight


## 4 - Frequencies and Date Offsets -------
hour = Hour() ## an offset, base frequency
four_hours  = Hour(4)
pd.date_range('2000-01-01', '2000-01-03 23:59', freq='4h')

Hour(2) + Minute(30)
pd.date_range('2000-01-01', periods=10, freq='1h30min')
pd.date_range('2012-01-01','2012-09-01', freq='WOM-1SUN')


## 5 - Shifting (Lead-Lag) Date -------
ts = pd.Series(np.random.randn(4),
                index=pd.date_range('1/1/2000', periods=4, freq='M'))
ts.shift(1) ##lag
ts.shift(-1) ##lead
ts / ts.shift(1) - 1

ts.shift(2, freq='M') ##shift by frequencies
ts.shift(3, freq='D')
ts.shift(1, freq='90T')


## 6 - Shifting Date with Offsets -------
now = datetime.now()
now + 3 * Day()
now + MonthEnd()
now + MonthEnd(2)

offset = MonthEnd()
offset.rollback(now) ## End of last month
offset.rollforward(now) ## Next end of month

ts = pd.Series(np.random.randn(20),
                index=pd.date_range('1/15/2000', periods=20, freq='4d'))

ts.groupby(MonthEnd().rollforward).mean() ##aggregate to the end of month
ts.groupby(MonthBegin().rollback).mean()


## 7 - Time Zone Handling -------
pytz.common_timezones[-5:]
tz = pytz.timezone('America/New_York')

ts_naive = pd.date_range('3/9/2012 9:30', periods=10, freq='D')
ts_naive = pd.Series(np.random.randn(len(ts_naive)), index=ts_naive)
ts = pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='America/New_York')
ts_naive = ts_naive.tz_localize('UTC') ## localize
ts_naive
ts = ts.tz_convert('UTC') ## convert


## 8 - Operations with Timestamp Objects -------
stamp = pd.Timestamp('2012-11-04 00:30', tz='US/Eastern')
stamp + 2*Hour()

##### Different Time Zones (will be converted to UTC)
rng = pd.date_range('3.7.2012 9:30', periods=10, freq='B')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts1 = ts[:7].tz_localize('Europe/London')
ts2 = ts1[2:].tz_convert('Europe/Moscow')
result = ts1 + ts2 
result.index


## 9 - Periods & Period Arithmetic -------
p = pd.Period(2007, freq='A-DEC') ## Periods represent timespans
p + 5 
p - 2
pd.Period('2014', freq='A-DEC') - p ## operate two periods with same freq

rng = pd.period_range('2000-01-01','2000-06-30', freq='M')

values = ['2001Q3', '2002Q2', '2003Q1']
pd.PeriodIndex(values, freq='Q-DEC')

### Period Frequency Conversion
p = pd.Period('2007', freq='A-DEC')
p.asfreq('M', how='start') ## convert freq by .asfreq()
p.asfreq('M', how='end')

p = pd.Period('2007', freq='A-JUN')
p.asfreq('M', how='start')
p.asfreq('M', how='end')

p = pd.Period('Aug-2007', 'M')
p.asfreq('A-DEC')

rng = pd.period_range('2006', '2009', freq='A-DEC')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts.asfreq('M', how='start')
ts.asfreq('B', how='end')

### Quarterly Period Freq
p = pd.Period('2012Q4', freq='Q-JAN') ## useful for fiscal year end
p 
p.asfreq('D', 'start') ## output the beginning of the period
p.asfreq('D', 'end')

p4pm = (p.asfreq('B','e') - 1).asfreq('T','s') + 16*60
p4pm 
p4pm.to_timestamp()

rng = pd.period_range('2011Q3', '2012Q4', freq='Q-JAN')
ts = pd.Series(np.arange(len(rng)), index=rng)
new_rng = (rng.asfreq('B','e') - 1).asfreq('T','s') + 16*60
ts.index = new_rng
ts 

### Converting Timestamp to Periods (and Back)
rng = pd.date_range('2000-01-01', periods=3, freq='M')
ts = pd.Series(np.random.randn(3), index=rng)
pts = ts.to_period() ## periods, refer to non-overlapping timespans
pts.to_timestamp() ## back to timestamp

### PeriodIndex from Arrays
data = pd.read_csv('00-Data/macrodata.csv')
data.head()
ix = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
data.index = ix
data.cpi 

### Resampling and Freq Conversion
#### Resampling: converting a time series from one freq to another
#### downsampling: aggregate to lower freq.
#### upsampling: converting to higher freq
rng = pd.date_range('2000-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts.head(10)

#### resample() similar to groupby() and use with agg func
ts.resample('M', kind='period').mean() ## downsampling
ts.resample('M', convention='start', loffset='+1d').mean()
ts.resample('M', kind='period').ohlc() ## open-high-low-close

#### Upsampling
ts.resample('T', fill_method='ffill')
ts.resample('T').ffill(limit=5)


## 10 - Rolling Window -------
close_px_all = pd.read_csv('00-Data/stock_px_2.csv', parse_dates=True, index_col=0)
close_px_all.head()
close_px = close_px_all[['MSFT', 'XOM', 'AAPL']]
close_px.resample('B').ffill()

close_px.AAPL.plot()
plt.show()

#### 10.1 - Rolling 
## rolling() performs as groupby and resample, taking the rolling window
close_px.AAPL.plot()
## less noisy, agg to 250-day window, all values is non-NA
close_px.AAPL.rolling(250).mean().plot() 
plt.show()

## accept shorter window
close_px.AAPL.rolling(250).mean()
close_px.AAPL.rolling(250, min_periods=10).mean()

#### 10.2 - Expanding
'''Start at the beginning and increase the size of window till get the whole'''
close_px.AAPL.plot()
close_px.AAPL.expanding(250).mean().plot() ## cummulative
close_px.AAPL.rolling(250).mean().plot() 
plt.show()

#### rolling than applying the transformtion func
close_px.rolling(60).mean().plot(logy=True)
## fixed-size time offset
close_px.rolling('20D').mean().head()

#### 10.3 - Exponentially Weighted Functions
'''Give more weight to more recent observations'''
aapl_px = close_px.AAPL['2006':'2007']
ma30 = aapl_px.rolling(30, min_periods=10).mean()
ewma30 = aapl_px.ewm(span=30).mean()

ma30.plot(style='-', label = 'Simple MA')
ewma30.plot(style='--', label = 'EW MA')

#### 10.4 - Binary Moving Window Functions
spx_px = close_px_all['SPX']
spx_rets = spx_px.pct_change()
returns = close_px.pct_change()
spx_rets.head()
returns.head()
corr = returns.AAPL.rolling(125, min_periods=100).corr(spx_rets)
corr.plot()
returns.rolling(125, min_periods=100).corr(spx_rets).plot()

#### 10.5 - User-Defined Moving Window Functions
score_at_2pct = lambda x: percentileofscore(x, 0.02)
returns.AAPL.rolling(250).apply(score_at_2pct).plot()


