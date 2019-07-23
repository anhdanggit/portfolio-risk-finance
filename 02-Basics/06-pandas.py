'''
#FILE: PANDAS TRICKS
Project: Basic concepts in Python
-------------------
By: Anh Dang
Date: 2019-07-17
Description:
Some illustrations for basic concepts in Python
'''

## import modules
import pandas as pd
import numpy as np
import pandas_datareader.data as web

## Filter the null/NA values from a series
series = pd.Series([1, np.nan, 2, 3, 4])
series[~series.isnull()]

## Rearrage columns in DataFrame
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
frame_arrange = pd.DataFrame(frame, columns=['year','state','pop'])
frame_arrange

## Auto add by index
val = pd.Series([-1.2, -1.5, -1.7], index=frame_arrange[frame_arrange['state'] != 'Ohio'].index)
frame_arrange['debt'] = val ## new col can't create with systax frame_arrange.debt
frame_arrange


## Essential Functionality ------------
#### 1. Reindex (rearrange + missing value)
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a','b','c','d','e'])

obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill') ## forward fillinf for missing values

#### 2. Reindexing by columns
frame = pd.DataFrame(np.arange(9).reshape((3, 3)), 
                    index=['a', 'c', 'd'],
                    columns=['Ohio', 'Texas', 'California'])

states = ['Texas','Ohio','California']
frame.reindex(columns=states) ## sort by values in Texas then Ohio then California

#### 3. Add two DataFrame with fill method 
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))

df1 + df2
df1.add(df2, fill_value=0)

## DataFrame and Series
### By default, they match the index of Series and the col of DataFrame
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)), 
        columns=list('bde'), 
        index=['Utah', 'Ohio', 'Texas', 'Oregon'])

series = frame.iloc[0]
series2 = pd.Series(range(3), index=['b', 'e', 'f'])
frame - series 
frame - series2 

### By index
series3 = frame['d']
frame.sub(series3, axis=0)


## Function Application & Mapping ---------------
f0 = lambda x: x.max() - x.min()
frame.apply(f0, axis=0)
frame.apply(f0, axis=1) ## apply to columns -> return rows

## Function could return the series with several values
def f(x):
    return pd.Series([x.min(), x.max(), x.max()-x.min()], index=['min', 'max','range'])
f(frame.iloc[1])

frame.apply(f, axis=0).T
frame.apply(f, axis=1)

## Element-wise in dataframe
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])

format_e = lambda x: '%.2f' % x

frame.applymap(format_e) ## applymap on DF
frame['b'].map(format_e) ## map on a series


## Sorting & Ranking ---------------
frame = pd.DataFrame(np.arange(8).reshape((2, 4)),   
                    index=['three', 'one'],   
                    columns=['d', 'a', 'b', 'c'])

frame.sort_index()
frame.sort_index(axis=1)

frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame.sort_values(by='b', ascending=False)
frame.sort_values(by=['a', 'b']) ## sort by multiple columns

## is unique
frame.b.is_unique
frame.a.is_unique

## aggregate by col/row
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], 
    index=['a', 'b', 'c', 'd'], 
    columns=['one', 'two'])

df.sum()
df.cumsum()
df.sum(axis=1, skipna=False)

## return the max id 
df.idxmax()

## describe
df.describe() 


## Corr and Cov ---------------
web.get_quote_yahoo('IBM')
ticker_get = ['IBM', 'AAPL', 'IBM', 'MSFT', 'GOOG']
all_data = {ticker: web.get_data_yahoo(ticker) for ticker in ticker_get}
all_data['IBM']['Adj Close']

price = pd.DataFrame({ticker: data['Adj Close'] for ticker, data in all_data.items()})
volume = pd.DataFrame({ticker: data['Volume'] for ticker, data in all_data.items()})

returns = price.pct_change()
returns.tail()

### compute the correlation
returns['IBM'].corr(returns['AAPL'])
returns.corr() ## whole table
returns.corrwith(returns['GOOG']) ## pairwise with GOOG


## Unique values, counts and membership ---------------
obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
obj.unique()

obj.value_counts()
pd.value_counts(obj, sort=False)

obj[obj.isin(['a','c'])]

to_match = pd.Series(['c', 'a', 'b', 'b', 'c', 'a'])
map_val = pd.DataFrame({'key': ['c','b','a'], 'value': [5, 10, 15]})
map_ix = pd.Index(map_val.key).get_indexer(to_match)
map_ls = map_val['value'][map_ix].reset_index(drop=True)

match_value = pd.DataFrame({'to_match': to_match, 'mapped_value': map_ls}).sort_values(by='to_match')
match_value