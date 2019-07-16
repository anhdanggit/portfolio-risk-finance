'''
BOND EVALUATION

Bond is fixed income security, differs by maturity.
- Convertible bond: able to convert to stock before maturity with determined shares
- Callable: bond issuers could buy back before maturity
- Puttable: bond buyers could sell back for bond originals before maturity
'''

## Import Modules
import scipy as sp


## 1 - Coupon Present Value -------
'''
Coupon Paymnet = (Coupon rate * FV) / frequency
PV(Face Value) = FV / (1+R)^n
PV(Coupon Payment - Annuity) = C/R (1 - 1/(1+R)^n)
Bond Price = PV(Face Value) + PV(Coupon Payment - Annuity)
'''
fv = 100
effective_annual_rate = 0.024
coupon_rate = 0.08
freq = 1 # pay annually
coupon_payment = coupon_rate*fv/freq

sp.pv(effective_annual_rate, 3, coupon_payment, fv) ## price of coupon is 116.02


## 2 - Yield-to-Maturity -------
'''
Same concept to IRR, the rate that make the FV = PV
- Coupon rate > YTM: Price > FV
- Coupon rate = YTM: Price = FV
- Coupon rate < YTM: Price < FV
'''

### YTM of bonds bought at 818, maturity 5, coupon rate 3% annual, fv = 1000
sp.rate(nper=5, pmt = 0.03*1000, pv=-818, fv=1000) ## 7.498% as YTM