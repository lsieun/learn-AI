#coding:utf-8

import numpy as np

b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(b)
print("="*40)

#知识点：“修改shape”和“转置”的区别

# （1）强制修改shape
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]
# b.shape = 4, 3
# print(b)

# （2）转置
# [[ 1  5  9]
#  [ 2  6 10]
#  [ 3  7 11]
#  [ 4  8 12]]
print(b.transpose())
print("="*40)
print(b.T)
