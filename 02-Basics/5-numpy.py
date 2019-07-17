'''
#FILE: NUMPY TRICKS
Project: Basic concepts in Python
-------------------
By: Anh Dang
Date: 2019-07-17
Description:
Some illustrations for basic concepts in Python
'''

import numpy as np

## Change the type
arr = np.array(['1','2','3'], dtype='str')
arr = arr.astype('float')

## need to use the copy, as the slice is still based on arr
arr = np.arange(10)
arr_slice = arr[5:8].copy()
arr_slice[1] = 12345
arr

## Boolean mask
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names[names != 'Bob']
names[~(names == 'Bob')]

## create 2-dim array 
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points) ## produce all pairs of points, spread the space
z = np.sqrt(xs**2 + ys**2)

points.shape #1000
xs.shape #1000x1000

import matplotlib.pyplot as plt
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")


# Expressing Conditional Logic as Array Operations
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
[ (x if c else y) for x, c, y in zip(xarr, cond, yarr)]
## equivalent
np.where(cond, xarr, yarr)


# Mathematical & Statistical Method
arr = np.random.randn(100)
(arr < 0).sum()
(arr < 0).any()
(arr < 0).all()


# Linear Algebra
from numpy.linalg import inv, qr

X = np.random.randn(5, 5)
mat = X.T.dot(X) ## X'X
inv(mat) ## (X'X)^-1
mat.dot(inv(mat)) ## (X'X) x (X'X)^-1

q, r = qr(mat) ## QR decomposition
q
r

series = np.array([0,0,1,4,1,1,1])
series.argmax()