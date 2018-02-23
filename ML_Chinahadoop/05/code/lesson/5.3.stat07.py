# coding:utf-8
#

import math
import numpy as np

if __name__ == '__main__':
    d1 = np.random.randn(10)
    density1, edges1 = np.histogramdd(d1, bins=[5])
    print('density1 = ',density1)
    print('edges1 = ',edges1)
    print(np.sum(density1))
    d2 = np.random.randn(100,2)
    density2, edges2 = np.histogramdd(d2, bins=[5,5])
    print('density2 = ', density2)
    print('edges2 = ', edges2)
    print(np.sum(density2))