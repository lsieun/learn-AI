#coding:utf-8

import numpy as np

np.set_printoptions(linewidth=200,suppress=True)

a = np.arange(1, 10)
print(a)
b = np.arange(20,25)
print(b)
print(np.concatenate((a, b)))