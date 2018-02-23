#coding:utf-8

import numpy as np

np.set_printoptions(linewidth=200,suppress=True)

# 4.3 stack and axis
a = np.arange(1, 7).reshape((2, 3))
b = np.arange(11, 17).reshape((2, 3))
c = np.arange(21, 27).reshape((2, 3))
d = np.arange(31, 37).reshape((2, 3))
print('a = \n', a)
print('b = \n', b)
print('c = \n', c)
print('d = \n', d)
print('*'*40)
s = np.stack((a, b, c, d), axis=0)
print('axis = 0 ', s.shape, '\n', s)
print('*'*40)
s = np.stack((a, b, c, d), axis=1)
print('axis = 1 ', s.shape, '\n', s)
print('*'*40)
s = np.stack((a, b, c, d), axis=2)
print('axis = 2 ', s.shape, '\n', s)