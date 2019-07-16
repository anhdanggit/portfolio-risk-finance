'''
This is the scripts to play with some fundamental concepts in Bonds & Stocks:
1. Annual Percentage Rate (APR)
2. Effective Annual Rate (EAR)
3. Compounding Frequency
4. Term structure of Interest
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

## 1 - Interest Rates --------------
'''
FV(simple interest) = PV * (1 + R*n)
FV(compounded interest) = PV * (1+R)^n
'''
periods = np.arange(0, 10, 1)
pv = 1000
r = 0.05

reference = np.ones(len(periods))*pv
simple_values = pv * (1 + r*periods)
compounded_values = pv * (1+r)**periods
values_df = pd.DataFrame({'periods':periods, 'simple_values':simple_values, 'compounded_values':compounded_values})

### Plot
plt.title('Compare the simple and compounded interest')
plt.xlabel('Numbers of Periods')
plt.ylabel('Values')
plt.xlim(0,11) 
plt.ylim(800,1600) 
plt.plot(periods, reference, 'b-')
plt.plot(periods, simple_values, 'b-')
plt.plot(periods, compounded_values, 'r--')
plt.show()

## 2 - Convert from Annual Percentage rate (APR) to Effective Period Rate (Rm) ---------------
'''
FV1 = FV2 
-> (1 + APR1/m1)^m1 = (1 + APR2/m2)^m2
'''
def APR1toR2(APR1, m1, m2):
    R2 = (1 + APR1/m1)**(m1/m2) - 1
    return R2

def APR1toAPR2(APR1, m1, m2):
    return m2 * APR1toR2(APR1, m1, m2)

APR1toR2(APR1=0.1, m1=2, m2=4) ## convert from semi-annual APR to R effective for quaterly
APR1toAPR2(APR1=0.1, m1=2, m2=4) ## convert from semi-annual APR to quaterly APR

## 3 - EAR (Effective Annual Rate) --------------
def EAR(APR, m):
    EAR = (1 + APR/m)**m - 1
    return EAR

EAR(APR=0.1, m=2)

### Effective Annula Rate for different frequencies
annual = 1; semi = 2; quaterly = 4; month = 12
d = 365; h = d*24; m = h*60; s = m*60; ms = s*1000
rate = 0.1

freq_series = [annual, semi, quaterly, month, d, h, m, s, ms]
ear_values = list(map(EAR, np.repeat(rate, len(freq_series)), freq_series))
df = pd.DataFrame([freq_series, ear_values]).transpose()
df.columns = ['frequency','EAR_values']
dtypes_dict = {'frequency': int, 'EAR_values': float}
df = df.astype(dtypes_dict)

## 4 - Effective Continuous Rate (Rc) --------------
### Infinite periods: Continuously compounded
sp.exp(rate) - 1

'''
FV = PV * (1+R)^m
FV = PV * exp^(Rc*T)
-> (1+Rm)^m = exp^(Rc*T)
'''

def APRtoRc(APR, m):
    Rc = m * sp.log(1+APR/m)
    return Rc

def RctoAPR(Rc, m):
    APR = (sp.exp(Rc/m) - 1)*m
    return APR

def RctoRm(Rc,m):
    APR = RctoAPR(Rc, m)
    Rm = APR/m
    return Rm


APRtoRc(APR=0.024, m=2)
RctoAPR(Rc=0.02, m=2)
RctoRm(Rc=0.02, m=2)

## 5 - Compare several situations with APR and m --------------
compare_options = pd.DataFrame({'APR':[0.1,0.1025,0.09878,0.09798,0.09759], 'm':[2,1,4,12,365]})
compare_options['EAR'] = pd.Series(map(EAR, compare_options.APR, compare_options.m))
compare_options

## 6 - Compare diff situation with Effective Period Rate and Frequencies --------------
compare_reverse = pd.DataFrame({'R_m':[0.05,0.024695,0.0081648,0.0002673], 'm':[2,4,12,365]})
compare_reverse['EAR'] = pd.Series(map(EAR, compare_reverse['R_m']*compare_reverse['m'], compare_reverse['m']))
compare_reverse
RctoAPR(Rc=0.0975803,m=1)

## 7 - Analog --------------
'''
Assume that the monthly effective rate is 0.25%. 
This means that in January, Mary would borrow $5,000 for 11 months 
because she would pay it back at the end of the year. This is true for February and other months.
'''
sp.fv(rate=0.0025, nper=12, pmt=5000, pv=0)
