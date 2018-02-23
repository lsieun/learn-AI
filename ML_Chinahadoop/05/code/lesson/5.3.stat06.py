# coding:utf-8
# 主要是对于np.random.randn的使用，
# 如果有两个参数，第一个数可以理解成的“样本个数”，第2个数理解成每个样本中“元素的个数”
# 如果是三个或以上参数，每个参数就可以理解成“维度”了

import math
import numpy as np

if __name__ == '__main__':
    d1 = np.random.randn(5)
    d2 = np.random.randn(2, 2)
    print('d1 = \n',d1)
    print('d2 = \n',d2)
    d3 = np.random.randn(5,2,2)
    print('d3 = \n', d3)

