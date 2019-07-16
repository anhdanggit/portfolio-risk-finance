'''
TERM STRUCTURE OF INTEREST RATES

To define the relationship between risk-free rate and time.
Risk-free rate is defined as the default-free treasury rate.

- Serve as a benchmark to estimate Yield-to-Maturity (YTM)
- YTM: period return if the bonds holds till maturity (~IRR)
- Spread (measure default risk): YTM - (risk-free rate) 
- Duration: number of years to recover our initial investment
'''

## Import modules
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

## Initial Set-up
input = '00-Data'
output = '01-Output'


## 1 - Yield Curve -------
time = [3/12, 6/12, 2, 3, 5, 10, 30] ## years
rate = [0.47, 0.6, 1.18, 1.53, 2, 2.53, 3.12]  ## risk-free rate

plt.title('Term Structure of Interest Rate')
plt.xlabel('Time (years)')
plt.ylabel('Risk-free rate (%)')
plt.plot(time, rate, 'o-', color='orange')
plt.show()

### interpolate the missing values
x = pd.Series([1,2,np.nan, np.nan, 6])
x.interpolate()

## 2 - Credit Rating (based on Spread = YTM - risk-free rate) ------
spread = pd.read_pickle(os.path.join(input, 'spreadBasedOnCreditRating.pkl'))

## 3 - Duration: Years to recover the initial investment ------
def duration(t, cash_flow, y):async def funcname(parameter_list):
    pass
    n = len(t)
    B = 0 # Bond PV
    for i in range(n):
        B += cash_flow[i] * sp.exp(-y*t[i]) ## 1/(1+R)^n = -exp(Rc*t)
    D = 0 # Duration
    for i in range(n): 
        D += t[i]*cash_flow[i]*sp.exp(-y*t[i]) / B
    return D








