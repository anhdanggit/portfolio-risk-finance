'''
#FILE: MODELING LIBRARIES
Project: portfolio-risk-finance
-------------------
By: Anh Dang
Date: 2019-07-24
Description:
Some basic modeling libraries in Python
'''

## Import modules
import pandas as pd
import numpy as np
import os
import scipy as sp
import matplotlib.pyplot as plt
import random
import patsy ## produce design matrices for a linear model
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score


## 1 - DataFrame and NumPy -------
data = pd.DataFrame({
    'x0': [1,2,3,4,5],
    'x1': [0.01, -0.01, 0.25, -4.1, 0.],
    'y' : [-1.5, 0., 3.6, 1.3, -2.]
})

data
data.columns
data.values ## to numpy.array 

df2 = pd.DataFrame(data.values, columns = ['one','two','three']) # array to df
df3 = data.copy()
df3['strings'] = 'a,b,c,d,e'.split(',')
df3.values

## subset the columns
model_cols = ['x0', 'x1']
data[model_cols].values

## categorical variable
data['category'] = pd.Categorical('a,b,a,a,b'.split(','), categories=['a','b'])
dummies = pd.get_dummies(data.category, prefix='category')
data_w_dummies = data.drop('category', axis=1).join(dummies)
data_w_dummies


## 2 - Patsy: Model Descriptions, Design Matrices -------
data = pd.DataFrame({
    'x0': [1,2,3,4,5],
    'x1': [0.01, -0.01, 0.25, -4.1, 0.],
    'y' : [-1.5, 0., 3.6, 1.3, -2.]
})

y, X = patsy.dmatrices('y ~ x0 + x1', data)
y1, X1 = patsy.dmatrices('y ~ x0 + x1 + 0', data)
y 
X
np.asarray(y) 
np.asarray(X)

### pasty objects can post directly to the algorithms
coef, resid, _, _ = np.linalg.lstsq(X, y)
coef 
coef.squeeze() ## turn it to 1-dimensional
coef = pd.Series(coef.squeeze(), index=X.design_info.column_names)
coef 

### mix Python code into Patsy formula
y, X = patsy.dmatrices('y ~ x0 + np.log(np.abs(x1) + 1)', data)
y1, X1 = patsy.dmatrices('y ~ standardize(x0) + center(x1)', data)
X1  

### Apply transformation to new out-of-sample data (scoring)
new_data = pd.DataFrame({'x0': [6, 7, 8, 9],
                        'x1': [3.1, -0.5, 0, 2.3],
                        'y': [1, 2, 3, 4]})
new_X = patsy.build_design_matrices([X.design_info], new_data)

### the + in patsy not means addition
y, X = patsy.dmatrices('y ~ I(x0 + x1)', data)
X 

### Categorical Data and Patsy
data = pd.DataFrame({'key1': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b'],
                    'key2': [0, 1, 0, 1, 0, 1, 0, 0],
                    'v1': [1, 2, 3, 4, 5, 6, 7, 8],
                    'v2': [-1, 0, 2.5, -0.5, 4.0, -1.2, 0.2, -1.7]})
y, X = patsy.dmatrices('v2 ~ key1', data)
X ## patsy would convert non-numeric to dummy variables by default

y, X = patsy.dmatrices('v2 ~ C(key2)', data) ## to turn numeric to categorical
X 

### interaction term of categorical var (ANOVA: Analysis of Variance)
data['key2'] = data['key2'].map({0: 'zero', 1: 'one'})
y, X = patsy.dmatrices('y ~ key1 + key2 + key1:key2', data)
X 


## 3 - statsmodels: fitting statistial model, text, explore, viz -------

#### 3.1 - Linear Models 
def dnorm(mean, variance, size=1):
    if isinstance(size, int):
        size = size
    return mean + np.sqrt(variance) * np.random.randn(size)

np.random.seed(12345)
N = 100
X = np.c_[dnorm(0, 0.4, size=N), ## numpy concancate as columns
        dnorm(0, 0.6, size=N),
        dnorm(0, 0.2, size=N)]
eps = dnorm(0, 0.1, size=N)

beta = [0.1, 0.3, 0.5] ## true model

y = np.dot(X, beta) + eps ## X.beta + eps 

X_model = sm.add_constant(X) ## add constant
X_model[:5]

### fitting OLS model
model = sm.OLS(y, X)
results = model.fit()

results.params
results.summary()

### DataFrame
data = pd.DataFrame(X, columns=['col0','col1', 'col2'])
data['y'] = y
data.head()
### Combine of sm and patsy
results = smf.ols('y ~ col0 + col1 + col2', data=data).fit()
results.params 
results.tvalues 
results.predict(data[:5])

#### 3.2 - Estimate Time-series Processes 
### Autoregressive Structure and Noise
init_x = 4
values = [init_x, init_x]
N = 1000

b0 = 0.8
b1 = -0.4 
noise = dnorm(0, 0.1, N)

for i in range(N):
    new_x = values[-1] * b0 + values[-2] * b1 + noise[i] ## AR(2) structure
    values.append(new_x)


MAXLAGS = 5
model = sm.tsa.AR(values)
results = model.fit(MAXLAGS)

results.params 


## 4 - Introduction to scikit-learn -------
data = pd.read_csv('00-Data/titanic_train.csv')

## sample train/test
ix = np.arange(len(data))
n_train = np.int(len(data)*0.75)
n_test = len(data) - n_train
train_ix = random.sample(list(ix), k=n_train)
test_ix = [i for i in ix if i not in train_ix]

train = data.iloc[train_ix, :]
test = data.iloc[test_ix, :]

train.Survived.mean()
test.Survived.mean()

## Checking the missing value
train.isnull().sum()
test.isnull().sum()
## Impute values
impute_value = train['Age'].median()
train['Age'] = train['Age'].fillna(impute_value)
test['Age'] = test['Age'].fillna(impute_value)
## Convert valye
train.Sex.value_counts()
train['is_female'] = (train['Sex'] == 'female').astype('int')
test['is_female'] = (test['Sex'] == 'female').astype('int')

## Take model variables
predictors = ['Pclass','is_female', 'Age']
X_train = train[predictors].values
y_train = train['Survived'].values
X_test = test[predictors].values

## Logit model
model = LogisticRegression()
model.fit(X_train, y_train)
## Predict
y_predict = model.predict(X_train)
(y_predict == train['Survived'].values).mean() ## 0.7934
y_predict = model.predict(X_test)
(y_predict == test['Survived'].values).mean() ## 0.8027

## Logit model with CV
model_cv = LogisticRegressionCV(10)
model_cv.fit(X_train, y_train)
## CV
scores = cross_val_score(model_cv, X_train, y_train, cv=4)
scores
scores1 = cross_val_score(model, X_train, y_train, cv=4)
scores1

## Predict
y_predict = model_cv.predict(X_train)
(y_predict == train['Survived'].values).mean() ## 0.7889
y_predict = model_cv.predict(X_test)
(y_predict == test['Survived'].values).mean() ## 0.7802
