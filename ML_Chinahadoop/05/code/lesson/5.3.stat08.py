# coding:utf-8
#

import math
import numpy as np

if __name__ == '__main__':
    x = y = np.arange(20)
    print('x = ', x)
    print('y = ', y)
    t = np.meshgrid(x, y)
    print(t)