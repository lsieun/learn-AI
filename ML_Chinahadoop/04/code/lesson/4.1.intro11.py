#coding:utf-8

import time
import math
import numpy as np

np.set_printoptions(linewidth=200,suppress=True)

# 4.1 numpy与Python数学库的时间比较
for j in np.logspace(0, 7, 8):
    x = np.linspace(0, 10, j)
    start = time.clock()
    y = np.sin(x)
    t1 = time.clock() - start

    x = x.tolist()
    start = time.clock()
    for i, t in enumerate(x):
        x[i] = math.sin(t)
    t2 = time.clock() - start
    print(j, ": ", t1, t2, t2/t1)