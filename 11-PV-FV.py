'''
This is the scripts to play with some fundamental concepts:
1. Present Value
2. IRR 

FV(t) = PV*(1+R)^t 
* R: rate per period
* t: number of periods
'''

## Load
import scipy as sp
import numpy.lib.financial as fin
import pandas as pd
import matplotlib.pyplot as plt


## 1 - FV-PV ---------------
'''
==> The idea is: As $1 today is more valuable than $1 tmr. You receive pv=100,
means you need to pay 121 (neg.) later in future
'''
sp.fv(rate=0.1, nper=2, pmt=0, pv=100) ## positive FV, means negative PV, vice versa
fin.fv(rate=0.1, nper=2, pmt=0, pv=100)
100*(1+0.1)**2 ## amount of pay-back in future (as pv positive), means negative sign

## 2 - Perpetuity ---------------
'''
* Same constant CF, same time intervals for forever
PV = CV/(1+R) + CV/(1+R)^2 + ...
PV(perpetuity) = C/R

* Perpetuity with constant growth of CV
PV(perpetuity-growth = C/(R-g))

* Perpetuity with CV receive at the end of kth period
PV = (C/(R-g)) * (1/(1+R)^(k-1)) 
=> Discount the values of CF to the end of kth period, 
then discount to the t=0

* Annuity: Perpetuity in n periods
PV = C/R(1 - 1/((1+R)^n))
=> Discount the perpetuity CF to t = 0, then
minus for discounted the CF after n to t = 0
The result is the value behaves like perpetuity from t=0 to t=n

* Annuity due:
= PV(Annuity)*(1+R) (or PV_Annuity / (1/(1+R)))
'''
fin.fv(0.01, 10, 20, 0, 1) ## 1 means 'ends'
fin.fv(rate=0.01, nper=10, pmt=20, pv=0, when=0)*(1+0.01)

def pvGrowingAnnuity(discountrate,growth,cashvalue, nper):
    v = cashvalue/(discountrate-growth) 
    factor = 1 - (1+growth)**nper/(1+discountrate)**nper
    return v*factor

pvGrowingAnnuity(discountrate=0.1, growth=0.05, cashvalue=10000, nper=30)

## 3 - Compute the monthly payment for a debt ---------------
'''
John is planning to buy a used car with a price tag of $5,000. 
Assume that he would pay $1,000 as the download payment and borrow the rest. 
The annual interest rate for a car load is 1.9% compounded monthly. 
What is his monthly payment if he plans to retire his load in three years?

PV = 5000 - 1000
rate_period = 1.9% / 12
periods = 3 years * 12
pmt = ? (Cashflow)

==> PV Annuity in n periods = 4000
'''
def pmtCompute(pv, rate, nperiod):
    pmt = pv*rate / (1 - (1/((1+rate)**nperiod)))
    return pmt

pmtCompute(pv=4000,rate=(0.019/12), nperiod=3*12)
fin.pmt(rate=(0.019/12), nper=3*12, pv=4000) ## compare with the pmt from numpy
sp.pmt(rate=(0.019/12), nper=3*12, pv=4000)

## 4 - Compute monthly effective rate --------------
'''
A company plans to lease a limousine for its CEO. 
If the monthly payment is $2,000 for the next three years and 
the present value of the car is $50,000, what is the implied annual rate?
'''
mthly_eff_r = sp.rate(nper=3*12, pmt=2000, pv=-50000, fv=0) ## this is effective mthly rate
annual_rate = mthly_eff_r*12

## 5 - Compute number of repayment (periods) --------------
'''
Peter borrows $5,000 to pay the cost to get a Python certificate. 
If the monthly rate is 0.12% and he plans to pay back $200 per month, 
how many months will he need to repay his loan?
'''
sp.nper(rate=0.0012, pmt=200, pv=-5000, fv=0)

## 6 - NPV Rules
'''
NPV = PV(Benefits) - PV(Cost)
- NPV >= 0, accept
- NPV < 0, reject
'''
discount_rate = 0.112
cashflows_series = [-100,50,60,70,100,20]
sp.npv(discount_rate, cashflows_series) ## 121.56

## 7 - IRR Rules -------------------
'''
IRR (Internal Rate of Return):
Discount rate that make the NPV of the cashflows = 0
- IRR >= R_c, accept (with R_c is the cost of finance)
- IRR < R_c, reject
'''
sp.irr(cashflows_series) ## irr = 0.5199%

## 8 - Payback Period Rules --------------------
'''
A payback period is defined as the number of years needed 
to recover the initial investment.
T =< T_c, accept (T_c: maximum years required)
T > T_c, reject
'''
cf_series = [-100,30,30,30,30,30]

def payback_period(cf_series):
    cumsum_cf = sp.cumsum(cf_series)
    count_neg = sp.sum(cumsum_cf < 0) - 1 ## start with t = 0
    pbp = count_neg + abs(cumsum_cf[count_neg])/cf_series[count_neg+1]
    return pbp

payback_period(cf_series) ## 3.33 yrs

#########################################################
#---- EXERCISES ----------------------------------------#
#########################################################

## Ex1: PV of $206 received in 10 years with an annual discount rate of 2.5%?
sp.pv(rate = 0.025, nper=10, pmt=0, fv=206)

## Ex2: FV of perpetuity with a periodic annual payment of $1 and a 2.4% annual discount rate?
def fvPerpetuity(pmt, r):
    return pmt/r
fvPerpetuity(pmt=1, r=0.024)

## Ex3: NPV is negatively correlated with the discount rate. Why?
discount_rates = np.random.random(10)
cf_series = [-100, 10, 20, 30, 40, 50]
npv_values = []
for i,r in enumerate(discount_rates):
    print(i)
    print(r)
    npv_values.append(sp.npv(r, cf_series))
df = pd.DataFrame({'discount_rates': discount_rates, 'npv_values': npv_values})
df = df.sort_values('discount_rates')
### Plot
plt.plot(df.discount_rates, df.npv_values)
plt.show()