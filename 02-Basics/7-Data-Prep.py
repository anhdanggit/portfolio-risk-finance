'''
#FILE: DATA CLEANING and PREP
Project: portfolio-risk-finance
-------------------
By: Anh Dang
Date: 2019-07-20
Description:
Overview some common tricks in clean and prep data
'''
## Import modules
import pandas as pd
import numpy as np
import os
import scipy as sp
import matplotlib.pyplot as plt
from numpy import nan as NA


## 1 - Missing Data -------
string_data = pd.Series(['aaaa','bbbbb', np.nan, 'avocado'])
string_data.isnull()
string_data[1] = None
string_data.dropna(inplace=True)

### Only drop rows with all NA
data = pd.DataFrame([[1., 6.5, 3., NA], [1., NA, NA, NA], [NA, NA, NA, NA], [NA, 6.5, 3., NA]])
data.dropna(how='all')
data.dropna(how='all', axis=1)
data.dropna(thresh=2) ## keep rows contains at least 2 obsvations.

### Filling Missing data 
data.columns = ['One','SixHalf','Three','Zero']
data.fillna(0)
data.fillna({'One': 1, 'SixHalf': 6.5, 'Three': 3, 'Zero': 0}) ## different fill value for each column

### Filling method
df = pd.DataFrame(np.random.randn(6, 3))
df.iloc[2:, 1] = NA
df.iloc[4:, 2] = NA
df.fillna(method='ffill', limit=2) ## limit for maxium number of consecutive periods to fill 


## 2 - Remove Dup -------
data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
                    'k2': [1, 1, 2, 3, 3, 4, 4]})
data.duplicated()
data.k1.duplicated()
data.drop_duplicates()
data['v1'] = range(7)
data.drop_duplicates(['k1']) ## subset, drop by col
data.drop_duplicates(['k1','k2'], keep='last') ## keep non-dup, the last 


## 3 - Encode / Transform value-value by Functions -------

### Mapping values
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                            'Pastrami', 'corned beef', 'Bacon',
                            'pastrami', 'honey ham', 'nova lox'],
                    'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})

meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}

data['animal'] = data['food'].str.lower().map(meat_to_animal) ## str.lower() do it element-wise
data['food'].map(lambda x: meat_to_animal[x.lower()]) ## call key within dict to map it with other values

### Replace
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data.replace([-999, -1000], NA)
data.replace([-999, -1000],[0, NA])
data.replace({-999: NA, -1000: 0})

### Transform indices
data = pd.DataFrame(np.arange(12).reshape((3, 4)),
                    index=['Ohio', 'Colorado', 'New York'],
                    columns=['one', 'two', 'three', 'four'])

transform = lambda x: x[:4].upper()
data.index = data.index.map(transform)
data.rename(index=str.title, columns=str.upper)
data.rename(index={'OHIO':'O2O'}, columns={'three': 3})


## 4 - Discretization / Binning -------
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]

cats = pd.cut(ages, bins)
pd.DataFrame({'ages': ages, 'cats': cats})
cats.codes ## values of cat in int
cats.categories
pd.value_counts(cats)

## Add labels
group_names = ['Youth', 'YoungAdult', 'MiddleAged','Senior']
pd.cut(ages, bins, labels=group_names)

## Equal-length bins
pd.cut(ages, 4, precision=2)
pd.cut(ages, 4, precision=2, labels=group_names)

## cut by quantiles
