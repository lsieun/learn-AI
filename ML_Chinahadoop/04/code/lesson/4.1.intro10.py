#coding:utf-8

import numpy as np

np.set_printoptions(linewidth=200,suppress=True)

# 3.3 二维数组的切片
# [[ 0  1  2  3  4  5]
#  [10 11 12 13 14 15]
#  [20 21 22 23 24 25]
#  [30 31 32 33 34 35]
#  [40 41 42 43 44 45]
#  [50 51 52 53 54 55]]
a = np.arange(0, 60, 10)    # 行向量
print('a = ', a)
b = a.reshape((-1, 1))      # 转换成列向量
print('b = \n', b)
c = np.arange(6)
print('c = ', c)
d = b + c   # 行 + 列
print('d = \n', d)
print('****************************************')

# 合并上述代码：
f = np.arange(0, 60, 10).reshape((-1, 1)) + np.arange(6)
print('f = \n', f)
print('****************************************')

# 二维数组的切片
print(f[[0, 1, 2], [2, 3, 4]])
print(f[4, [2, 3, 4]])
print(f[4:, [2, 3, 4]])
i = np.array([True, False, True, False, False, True])
print(f[i])
print(f[i, 3])