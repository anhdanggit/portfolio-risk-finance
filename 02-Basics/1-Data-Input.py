'''
This is the scripts to play with some basic Py 
Data set: IBM from Yahoo
'''

## Import modules
import pandas as pd
import numpy as np
import scipy as sp
import pandas_datareader.data as web
import os 

## Initial Set-up
data_input = '00-Data'
data_ouput = '01-Output'


## Read Data ---------
ibm_price = pd.read_csv(os.path.join(data_input, 'ibm.csv'))
ibm_price.head()

mthly_ff = pd.read_pickle(os.path.join(data_input, 'ffMonthly.pkl'))
mthly_ff = pd.DataFrame(mthly_ff)
mthly_ff.head()
## change the index to one columns
mthly_ff['date'] = mthly_ff.index
mthly_ff.reset_index()


## Data Output ---------
mthly_ff.to_csv(os.path.join(data_ouput, 'mthly_ff.csv'), index=False)

### Trick: to find all methods related to an object: 
### dir(<obj-by-that-class>)
### help(''.split)
## string handles
string = 'Hello World !'
string.split()
string.lower()

## Read Data from pandas_reader ---------
web.get_quote_yahoo('IBM')
ticker_get = ['IBM', 'AAPL', 'IBM', 'MSFT', 'GOOG']
all_data = {ticker: web.get_data_yahoo(ticker) for ticker in ticker_get}
all_data['IBM']['Adj Close']

price = pd.DataFrame({ticker: data['Adj Close'] for ticker, data in all_data.items()})
volume = pd.DataFrame({ticker: data['Volume'] for ticker, data in all_data.items()})



## Mathematics -----------
g = np.array([[2,2,2],[3,3,3]])
g.sum()
g.flatten()
g.reshape(3,2)
## Matrics manipulation
A = np.array([[1,2,3],[3,4,5]], float) ##matrix: 2x3
B = np.array([[1,2],[3,4],[5,6]], float) ##matrix: 3x2
AB = np.dot(A, B) ##matrix: 2x2
## Using some from scipy
ret = sp.array([0.1, 0.05, -0.02]) ##return series
arithmeticMean = ret.mean()
geoMean = pow(sp.prod(ret+1), 1./len(ret)) - 1

## (1) PV perpetuity Cashflow $50 annually
### PV = C/r
def discount_r(pv, cf, type='annual'):
    return cf/pv
### Compute the discount rate annually
discount_r(pv=124, cf=50)

## (2) PV perpetuity CF growing (the growth rate is constant)
### PV = C/(r-g)
def pv_grow(cf, g, r):
    return(cf/(r-g))
### Compute the pv of growing perpetuity CF
pv_grow(cf=12.5, g=0.025, r=0.085)

## (3) n-day variance
def n_day_variance(daily_std, nday):
    n_day_var = nday * (daily_std**2)
    return n_day_var
### Compute the n-day variance
n_day_variance(daily_std=0.2, nday=10)

## (4) PV of Cashflow
def pv(cf, r):
    T = len(cf)
    sum_pv = 0
    for t in range(T):
        sum_pv += cf[t]/((1+r)**t)
    return sum_pv
### Compute the PV of a cashflow of 25,000 in 5 years, with discount rate of 4.5%
pv(cf=np.repeat(25000/5, 5), r=0.045)

## (5) Sharpe Ratio
def sharpe_ratio(mean_return, riskfree_return, risk):
    sharpe = (mean_return - riskfree_return)/risk
    return sharpe
### Compute Sharpe Ratio
sharpe_ratio(mean_return = 0.5, riskfree_return = 0.1, risk = 0.1)