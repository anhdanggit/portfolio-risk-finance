'''
#FILE: ADVANCED PANDAS
Project: portfolio-risk-finance
-------------------
By: Anh Dang
Date: 2019-07-24
Description:
Some tricks in pandas
'''

## Import modules
import pandas as pd
import numpy as np
import os
import scipy as sp
import matplotlib.pyplot as plt

## 1 - Categorical Data -------
values = pd.Series(['apple', 'orange', 'apple', 'apple']*2)
values 
pd.unique(values) ##or values.unique()
pd.value_counts(values) ##or values.value_counts()

## Dic-encoded: from integer (as codes) to cat
dim = pd.Series(['apple', 'orange'])
values = pd.Series([0, 1, 0, 0] * 2)
dim.take(values) ## still object (just encode by dict)

#### 1.1 - Categorical Type 
fruits = ['apple', 'orange', 'apple', 'apple'] * 2
N = len(fruits) 
df = pd.DataFrame({'fruit': fruits,
                    'basket_id': np.arange(N),
                    'count': np.random.randint(3, 15, size=N),
                    'weight': np.random.uniform(0, 4, size=N)},
                    columns=['basket_id', 'fruit', 'count', 'weight'])

fruits_cat = df['fruit'].astype('category')
fruits_cat = fruits_cat.values
type(fruits_cat.values) 
fruits_cat.categories ## get categories
fruits_cat.codes  ## get int codes

## convert the column in df
df['fruit'] = df['fruit'].astype('category')
## create cat series separately
my_cat = pd.Categorical(['foo','bar','baz','foo','bar'])
my_cat 
## construct by from_codes
cat = ['foo','bar','baz']
codes = [0, 1, 2, 0, 0, 1]
cat_encode = pd.Categorical.from_codes(codes, cat) ## unorder
cat_encode 
cat_encode2 = pd.Categorical.from_codes(codes, cat, ordered=True)
cat_encode2 
cat_encode.as_ordered() ## put an unorder to order

#### 1.2 - Computation w Categorical
draws = np.random.randn(1000) 
bins = pd.qcut(draws, 4, labels=['Q1','Q2','Q3','Q4'])
bins 
bins.codes[:10]

bins = pd.Series(bins, name='quartile')
results = (pd.Series(draws)
            .groupby(bins)
            .agg(['count','min','max'])
            .reset_index())
results 

#### 1.3 - Categorical Methods 
s = pd.Series(['a','b','c','d']*2)

## Convert to Catergorical type
cat_s = s.astype('category')
cat_s.cat.codes 
cat_s.cat.categories 

## Extend the set of categories
actual_categories = ['a','b','c','d','e']
cat_s2 = cat_s.cat.set_categories(actual_categories)
cat_s.value_counts()
cat_s2.value_counts() 

## trim unobserved categories
cat_s3 = cat_s[cat_s.isin(['a','b'])]
cat_s3
cat_s3.cat.remove_unused_categories()

#### 1.4 - Dummy Variables (for Modeling)
pd.get_dummies(cat_s)


## 2 - Advanced GroupBy -------

#### 2.1 - Transform 
df = pd.DataFrame({'key': ['a', 'b', 'c'] * 4,
                    'value': np.arange(12.)})

df.groupby('key').value.mean() ## aggregating
df['group_mean'] = df.groupby('key').value.transform(lambda x: x.mean()) ## similar to group_by, mutate in R
df.head()

## similar result to apply (but the aggregate function would be diff)
def normalize(x):
    return (x - x.mean()) / x.std()
df['value_norm'] = df.groupby('key').value.transform(normalize)
## Transform 
df['value_norm2'] = ((df['value'] - df.groupby('key').value.transform('mean')) / 
    df.groupby('key').value.transform('std'))

df.head()

#### 2.2 - Grouped Time Resampling
## GroupBy time freq
df = pd.DataFrame({'time': pd.date_range('2017-05-20 00:00', freq='1min', periods=15),
                    'value': np.arange(15)})
df.set_index('time').resample('5min').count()

## GroupBy both cat and time
N = 15
times = pd.date_range('2017-05-20 00:00', freq='1min', periods=N)
df2 = pd.DataFrame({'time': times.repeat(3),
                    'key': np.tile(['a','b','c'], N),
                    'value': np.arange(N*3.)})
df2.head(7)
(df2.set_index('time')
    .groupby(['key', pd.TimeGrouper('5min')]) ## time must be the index of DF or series
    .sum())

## 3 - Techniques for Method Chaining -------
df2 = df2.set_index('time')
df2 = df2.assign(value_sq = df2.value**2)

res = (pd.DataFrame({'key': ['a', 'b', 'c'] * 4,
                    'value': np.arange(12.)}) ## output pass-to the []
                    [lambda x: x['value'] < 5])
(pd.DataFrame({'key': ['a', 'b', 'c'] * 4,
                    'value': np.arange(12.)}) ## output pass-to the []
                    [lambda x: x['value'] < 5]
                    .groupby('key')
                    .value.sum())

#### 3.1 - Pipe Method 

##### Sequence of functions
df = pd.DataFrame({'key': ['a', 'b', 'c'] * 4,
                    'value': np.arange(12.)})

def value_squr(df, cols):
    result = df.copy()
    for c in cols:
        result[c] = df[c]**2
    return result 
def group_demean(df, by, cols):
    result = df.copy()
    for c in cols:
        result[c] = df[c] - df.groupby(by)[c].transform('mean')
    return result

## pipeline
(df[df.value < 10]
    .pipe(value_squr, ['value'])
    .pipe(group_demean, ['key'], ['value']))




