'''
#FILE: AGGREGATE & GROUP-BY
Project: Basic concepts in Python
-------------------
By: Anh Dang
Date: 2019-07-17
Description:
Some illustrations for basic concepts in Python for aggregate and group by
'''

## Import modules
import pandas as pd
import numpy as np
import os
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm

## 1 - GroupBy -------
## Split-Apply-Combine
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                    'key2' : ['one', 'two', 'one', 'two', 'one'],
                    'data1' : np.random.randn(5),
                    'data2' : np.random.randn(5)})

key1_mean = df['data1'].groupby(df['key1']).mean()
key12_mean = df['data1'].groupby([df['key1'], df['key2']]).mean()
key12_mean.unstack()

df.groupby('key1').mean()
df.groupby(['key1','key2']).size()

## Iteratable Groups
for name, group in df.groupby('key1'):
    print(name)
    print(group)

for (k1, k2), group in df.groupby(['key1', 'key2']):
    print((k1, k2))
    print(group)

pieces = dict(list(df.groupby('key1')))
pieces['a']

## Select column or subsets of columns (better syntax)
df.groupby('key1')['data1'].mean()
df.groupby('key1')[['data2']].mean()

## Grouping with Dicts and Series
people = pd.DataFrame(np.random.randn(5, 5),
                        columns=['a', 'b', 'c', 'd', 'e'],
                        index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people.iloc[2:3, [1,2]] = np.nan

mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
            'd': 'blue', 'e': 'red', 'f' : 'orange'}
map_series = pd.Series(mapping)

people.groupby(mapping, axis=1).sum()
people.groupby(map_series, axis=1).sum()


## 2 - Data Aggregation -------
def peak_to_peak(arr):
    return arr.max() - arr.min()

df.groupby('key1').agg(peak_to_peak) ## put the aggr function to .agg()
df.groupby('key1').describe()


## 3 - Column-Wise + Multiple Function Application -------
tips = pd.read_csv('00-Data/tips.csv')
tips.groupby(['day','smoker'])['total_bill'].agg(['mean','std', peak_to_peak])
### put is (<name-col-output>,<function>)
tips.groupby(['day','smoker'])['total_bill'].agg([('avg','mean'), ('st. deviation', np.std)])

### multiple col summarise
functions = ['count','mean','max']
tips.groupby(['day','smoker'])['total_bill','tip'].agg(functions)

### map the function with colname
tips.groupby(['day','smoker']).agg({'tip': np.max, 'size': 'sum'})
tips.groupby(['day','smoker']).agg({'tip': ['min','max','mean','std'], 'size': 'sum'})
tips.groupby(['day','smoker'], as_index=False).agg({'tip': ['min','max','mean','std'], 'size': 'sum'})


## 3 - Apply -------
tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])
## top 5 tip_pct
def top(df, n=5, column='tip_pct'):
    return df.sort_values(by=column)[-n:] ## be creative, as long as return a df or scalar

top(tips, n=6)
tips.groupby('smoker').apply(top)
tips.groupby('smoker', group_keys=False).apply(top) ## disable group_keys
tips.groupby(['smoker','day']).apply(top, n=1, column='total_bill') ## pass arguments

tips.groupby('smoker')['tip_pct'].describe()
tips.groupby('smoker')['tip_pct'].describe().unstack('smoker')

## join the aggregated values to the df
max_tip_pct = tips.groupby(['smoker','day'], as_index=False)['tip_pct'].max()
pd.merge(tips, max_tip_pct, on=['smoker','day'], suffixes=['','_max'])


## Example: Quantile and bucket analysis
frame = pd.DataFrame({'data1': np.random.randn(1000),
                    'data2': np.random.randn(1000)})

quantiles = pd.qcut(frame.data1, 4)
frame.data2.groupby(quantiles).agg(['min','max','count','mean']).reset_index()

def get_stat(v):
    return{'min': v.min(), 'max': v.max(), 'count': v.count(), 'mean': v.mean()}
grouping = pd.qcut(frame.data1, 10, labels=False)
frame.data2.groupby(grouping).apply(get_stat).unstack()


## Example: Filling missing values by group-specific values
states = ['Ohio', 'New York', 'Vermont', 'Florida',
            'Oregon', 'Nevada', 'California', 'Idaho']
group_key = ['East'] * 4 + ['West'] * 4

data = pd.Series(np.random.randn(8), index=states)
data[['Vermont', 'Nevada', 'Idaho']] = np.nan ## put some NA value
data
data.groupby(group_key).mean()

#### fill NA by mean of the group 
fill_mean = lambda g: g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)

#### fill NA by values assigned to group
fill_values = {'East': 0.5, 'West': -1}
fill_func = lambda g: g.fillna(fill_values[g.name])
data.groupby(group_key).apply(fill_func)


## Example: Random Sampling & Permutation
# Hearts, Spades, Clubs, Diamonds
suits = ['H', 'S', 'C', 'D']
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ['A'] + list(range(2, 11)) + ['J', 'K', 'Q']
cards = []
for suit in ['H', 'S', 'C', 'D']:
    cards.extend(str(num) + suit for num in base_names)

deck = pd.Series(card_val, index=cards)

def draw(deck, n=5):
    return deck.sample(n)

draw(deck)

get_suit = lambda card: card[-1]
deck.groupby(get_suit).apply(draw, n=2) ## apply the get_suit to the index, and then draw function within group
deck.groupby(get_suit, group_keys=False).apply(draw, n=2)


## Example: Group Weighted Average
df = pd.DataFrame({'category': ['a', 'a', 'a', 'a','b', 'b', 'b', 'b'],
                    'data': np.random.randn(8),
                    'weights': np.random.rand(8)})

get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
df.groupby('category').apply(get_wavg)


## Example: Correlation
close_px = pd.read_csv('00-Data/stock_px_2.csv', parse_dates=True, index_col=0)
close_px.info()
close_px.head()

spx_corr = lambda g: g.corrwith(g['SPX']) ## computing func
get_year = lambda x: x.year ## grouping func

rets = close_px.pct_change().dropna()
rets.groupby(get_year).apply(spx_corr)
rets.groupby(lambda ix: ix.year).apply(lambda g: g['AAPL'].corr(g['MSFT']))


## Example: Group-Wise Linear Regression
def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1. 
    result = sm.OLS(Y, X).fit()
    return result.params

### run yearly linear regression of APPL and SPX return
rets.groupby(lambda ix: ix.year).apply(regress, 'AAPL', ['SPX'])


## 4 - Pivot Tables and Cross-Tabulation -------
tips.pivot_table(index=['day','smoker']).reset_index()
tips.pivot_table(values=['size','tip'], 
                index=['time','day'], ## row levels
                columns='smoker', ## columns level
                aggfunc='sum',
                margins=True) ## adding All row and column tables

tips.pivot_table('tip_pct',
                index=['time','smoker'],
                columns='day', aggfunc=len, margins=True, fill_value=0)   

pd.crosstab(tips.day, tips.smoker)
pd.crosstab(tips.day, tips.smoker, margins=True)