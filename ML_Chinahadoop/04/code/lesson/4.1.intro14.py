#coding:utf-8

import numpy as np

np.set_printoptions(linewidth=200,suppress=True)

a = np.arange(1, 10).reshape(3,3)
print('a = ',a)
b = a + 10
print('b = ',b)
print(np.dot(a, b)) #Dot product of two arrays.
print(a * b)
