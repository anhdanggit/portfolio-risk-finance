'''
#FILE: DATA WRANGLING
Project: portfolio-risk-finance
-------------------
By: Anh Dang
Date: 2019-07-21
Description:
Overview some common tricks in join, combine, and reshapre
'''

## Import modules
import pandas as pd
import numpy as np
import os
import scipy as sp
import matplotlib.pyplot as plt
from datetime import datetime

## 1 - Indexing -------
### Hierarchical index
data = pd.Series(np.random.randn(9), 
                index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                [1, 2, 3, 1, 3, 1, 2, 2, 3]])
data.index 
data['b'] ## partial indexing
data.loc[:, 2] ## inner level
data.unstack()

### Hierarchical columns
frame = pd.DataFrame(np.arange(12).reshape((4, 3)), 
                    index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                    columns=[['Ohio', 'Ohio', 'Colorado'],
                            ['Green', 'Red', 'Green']])
frame.index.names = ['key1','key2']
frame.columns.names = ['state','color']
frame['Ohio']
frame.loc['a']['Ohio']

frame.swaplevel('key1', 'key2')
frame.sort_index(level=1)

### Group columns into index
frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
                    'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                    'd': [0, 1, 2, 0, 1, 2, 3]})
frame2 = frame.set_index(['c','d'])
frame.set_index(['c','d'], drop=False)
frame2.reset_index()

### Summary statistic by level
frame.sum(level='key2')
frame.sum(level='color',axis=1)


## 2 - Combine & Merge -------
df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                    'data1': range(7)})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                    'data2': range(3)})
pd.merge(df1, df2)
pd.merge(df1, df2, on='key')

df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                    'data1': range(7)})
df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'],
                    'data2': range(3)})
pd.merge(df3, df4, left_on='lkey', right_on='rkey')
pd.merge(df1, df2, how='outer') ## 'inner', 'left', 'right', 'outer'

left = pd.DataFrame({'key1': ['foo', 'foo', 'bar'],
                    'key2': ['one', 'two', 'one'],
                    'lval': [1, 2, 3]})
right = pd.DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                    'key2': ['one', 'one', 'one', 'two'],
                    'rval': [4, 5, 6, 7]})
pd.merge(left, right, on='key1', suffixes=('_left','_right'))
pd.merge(left, right, left_index=True,  right_index=True)

### join is on index for df with no dup col
left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
                    index=['a', 'c', 'e'],
                    columns=['Ohio', 'Nevada'])
right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                    index=['b', 'c', 'd', 'e'],
                    columns=['Missouri', 'Alabama'])

left2.join(right2, how='inner')

another = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                        index=['a', 'c', 'e', 'f'],
                        columns=['New York', 'Oregon'])

left2.join([right2, another], how='outer', sort=True)

### Concatenating along an axis (when the key or index is diff)
arr = np.arange(12).reshape((3,4))
arr
np.concatenate([arr, arr])
np.concatenate([arr, arr], axis=1)

s1 = pd.Series([0,1], index=['a','b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])

pd.concat([s1, s2, s3])
pd.concat([s1, s2, s3], axis=1, keys=['one','two','three'])

df1 = pd.DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
pd.concat([df1,df2])
pd.concat([df1,df2], ignore_index=True)

### Combine overlapped index, if null choose another one
a = pd.Series([np.nan, 2.5, 0.0, 3.5, 4.5, np.nan], index=['f', 'e', 'd', 'c', 'b', 'a'])
b = pd.Series([0., np.nan, 2., np.nan, np.nan, 5.], index=['a', 'b', 'c', 'd', 'e', 'f'])
b.combine_first(a)


## 3 - Reshapre + Pivoting -------
data = pd.DataFrame(np.arange(6).reshape((2, 3)),   
                    index=pd.Index(['Ohio', 'Colorado'], name='state'),   
                    columns=pd.Index(['one', 'two', 'three'],
                    name='number'))
result = data.stack() ## stack by innermost level
result.unstack()
result.unstack(0) ## stack by the level
result.unstack('state')

#### 3.1 - Pivot from "Long" to "Wide" 
data = pd.read_csv('00-Data/IBM.csv')
data.head()
#data.Date = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d'), data.Date))
data['Date'] = pd.to_datetime(data['Date'])
periods = pd.DatetimeIndex(data['Date'])

columns = pd.Index(['Open','Volume','Close'])
data1 = data.reindex(columns=columns)
data1.index = periods
long_data = data1.stack().reset_index().rename(columns={0:'value'}) ## long-format

long_data['value2'] = np.random.randn(len(long_data))
long_data.pivot('Date','level_1','value')
long_data.pivot('Date','level_1')
long_data.pivot('Date','level_1')['value']
## equivalent to
long_data.set_index((['Date','level_1'])).unstack('level_1')


#### 3.2 - Pivot from "Wide" to "Long" 
df = pd.DataFrame({'key': ['foo', 'bar', 'baz'],
                    'A': [1, 2, 3],
                    'B': [4, 5, 6],
                    'C': [7, 8, 9]})
melted = pd.melt(df, ['key'])
melted.pivot('key','variable','value').reset_index()
pd.melt(df, id_vars=['key'], value_vars=['B','C'])
pd.melt(df, value_vars=['key','B','C'])