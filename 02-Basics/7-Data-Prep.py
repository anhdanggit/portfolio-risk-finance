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
import re ## pattern matching, substitution, and splitting


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
data = np.random.randn(1000)  # Normally distributed
qtile = pd.qcut(data, 4, precision=2)
qtile.categories
pd.value_counts(qtile)

qtile2 = pd.qcut(data, [0, 0.01, 0.05, 0.95, 0.99, 1])
qtile2.categoriess
pd.value_counts(qtile2)


## 5 - Detect/Filter Out Outliers -------
data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()
col = data.iloc[:,2]
col[np.abs(col) > 3]

data[(np.abs(data) > 3).any(1)] ## any row with any value at any col > 3
data[np.abs(data) > 3] = np.sign(data) * 3 ## cap -3 amd 3
data.describe().T

np.sign(data).head()


## 6 - Permutation & Random Sampling -------
df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))

### permutating: randomly reordring the series
sampler = np.random.permutation(5)
df.take(sampler)
df.iloc[sampler, :]

### sampling without replacement
df.sample(n=3)

choices = pd.Series([5,7,-1,6,4])
choices.sample(n=10, replace=True)


## 7 - Dummy Variables/Indicators -------
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],   
                 'data1': range(6)})

dummies = pd.get_dummies(df['key'], prefix='key')
df_trans = df[['data1']].join(dummies)


## 8 - String Manipulation -------
val = 'a.b.    eabcd'

### for 1 string
len(val)
val.index('b')
val.find('o') ## not found, return -1
val.count('b')
val.replace('.','-')

s1, s2, s3 = val.split('.')
strings = val.split('.')
' || '.join(strings)


## 9 - Regular Expressions -------
text = "foo    bar\t baz  \tqux" 
re.split('\s+', text) ## 1+ whitespace

regex = re.compile('\s+')
regex.split(text)
regex.findall(text) ## find all part matched your regex pattern

text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""

email_pattern = r'[A-Z0-9._%+-]+@[A-Z0-9._]+\.[A-Z]{2,4}'
regex_email = re.compile(email_pattern, flags=re.IGNORECASE)
regex_email.findall(text) ## return all matched string
positions = regex_email.search(text)
text[positions.start():positions.end()]

print(regex_email.sub('CENSORED', text))

## Concept of Groups in Regex pattern
email_pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex_email_gr = re.compile(email_pattern, flags=re.IGNORECASE)
m = regex_email_gr.match('hello@gmail.com')
m.groups()

regex_email_gr.findall(text)
t = regex_email_gr.sub(r'name: \1, domain(\2), suffix: \3', text) ## access groups by \1, \2
print(t)


## 10 - Vectorize string functions -------
data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com',
        'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = pd.Series(data)
data.str.contains('gmail') ## put str to vectorize

## Vectorize regex functions
data.str.findall(email_pattern, flags=re.IGNORECASE)
data.str[:5]