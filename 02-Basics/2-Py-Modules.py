'''
This is the scripts to play with some basic Py modules
in Financial and Portfolio Management
'''

## Import modules
import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pylab 
import os 
import datetime ##timeseries
from matplotlib.dates import MonthLocator, DateFormatter ##timeseries
import pandas_datareader as web ##to pull the financial data from webs
import numpy.lib.financial as finance ##to compute the financial calculator

## Initial Set-up
data_input = '00-Data'
data_ouput = '01-Output'


## 1. Numpy ------------
### create a matrix 2x3
x = np.array([[1,2,3],[3,4,5]])
### check size
np.size(x)
np.size(x, axis=0) ##2 rows
np.size(x, axis=1) ##3 cols
### manipulate matrix col and row
np.std(x, axis=1) ##std by col
x.sum()
x.sum(axis=0) ##sum by row
### simulation
ret = np.array(range(1, 100), float)/100 ##create a grid of return pct
rand_50 = np.random.rand(50) ##50 rand num btw 0 and 1
rnorm_100 = np.random.normal(size=100)


## 2. Scipy ------------
### Compute the NPV of a Cashflow
cf = [-100, 50, 40, 20, 10, 50]
sp.npv(rate=0.1, values=cf)
### Monthly payment: APR = 0.045, compounded monthly, mortgage = 250k, T=30yrs
sp.pmt(rate=0.045/12, nper=12*30, pv=250000)
### PV of one future cf
sp.pv(rate=0.1,nper=5,pmt=0,fv=100)
### PV of Annuity (series of payments)
sp.pv(rate=0.1, nper=5, pmt=100)
### Geometric mean returns
def geoMeanreturn(ret):
    return pow(sp.prod(ret+1), 1./len(ret)) - 1
ret = np.array([0.1, 0.05, -0.02])
geoMeanreturn(ret)
### other fundamental stuff
sp.unique([1,1,2,3,4,1,1,1])
sp.median([1,1,2,3,4,1,1,1])


## 3. matplotlib ------------
### Series of Normal Distribution
plt.plot(rnorm_100)
plt.xlabel("index")
plt.ylabel("random values")
plt.title("Normal Distribution")
plt.show()
### Sin & Cosine Functions
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
cos, sin = np.cos(x), np.sin(x)
plt.plot(cos), plt.plot(sin)
plt.xlabel("index")
plt.ylabel("random values [-pi,pi]")
plt.title("Cosine and Sine")
plt.show()
### Scatter of Normal Distribution
X = np.random.normal(size=1000)
Y = np.random.normal(size=1000)
plt.scatter(X,Y)
plt.show()
### Time-series
ibm = pd.read_csv(os.path.join(data_input, 'IBM.csv'))
ibmClosePrice = ibm['Close']
ibmClosePrice.index = ibm['Date']
### plot time-series
ibmClosePrice.plot()
plt.show()


## 4. statsmodels ------------
### Simulation
beta_actual = 2.8
alpha_actual = 3

x = np.random.random_sample(size = 1000)
y = alpha_actual + beta_actual*x
noise = np.random.normal(size = 1000)
y_noise = y + noise 
# plt.scatter(y, y_noise)
# plt.show()

### OLS fitting
x = sm.add_constant(x)  ## sm considering things a matrix
ols_results = sm.OLS(y_noise, x, hasconst=True).fit()
ols_results.summary()
'''
Params Predicted: 3.04, 2.81
Quite close to our actual parameters: 3, 2.8
'''


## 5. pandas ------------
### Create the time-series index
dates = pd.date_range('2018-01-01', periods=5, freq='M') - \
    pd.offsets.MonthBegin(1) ##to floor to the beginning of the month
### Create the dataframe
np.random.seed(123)
x = pd.DataFrame(np.random.rand(5,2),index=dates,columns=('A','B'))
x.describe()

### Create a pd.Series
x = pd.Series([1,2,3,4, np.nan, 6, 7, 8, 9])
median = x.median()
x.fillna(median) ##not replace the actual values in x

### interpolate
np.random.seed(123)                   # fix the random numbers 
x = np.arange(1, 10.1, .25)**2      
n = np.size(x)
y = pd.Series(x + np.random.randn(n))
bad = np.array([4,13,14,15,16,20,30,31,33,34])   # generate a few missing values
y[bad] = np.nan                       # missing code is np.nan
methods = ['linear', 'quadratic', 'cubic']
df = pd.DataFrame({m: y.interpolate(method=m) for m in methods})
df.plot()

### Merge
x = pd.DataFrame({'key': ['A','B','C','D'], 'value': [0.1,0.2,0.3,0.4]})
y = pd.DataFrame({'key':['B','D','D','E'],'value': [2, 3, 4, 6]})
z = pd.merge(x, y, on='key')
z2 = pd.merge(y, x, how='left', on='key')
z2

### Datetime
date1 = pd.datetime(2019, 7, 12)
date2 = pd.datetime(2018, 6, 12)
date1 - date2
date1.weekday()

### Stake data
df_stock = pd.DataFrame(np.random.randn(4,2), columns=['Stock A',' Stock B'])
k = df_stock.stack()
k.unstack()


## 5. pandas_reader ------------
df = web.get_data_yahoo("ibm")
df.head()

## 6. numpy.lib.financials for fin calculator ------------
finance.pv(rate=0.04, nper=12,fv=2000, pmt=500)