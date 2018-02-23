#coding:utf-8

import numpy as np

# 3.2 整数/布尔数组存取
# 3.2.1
# 根据整数数组存取：当使用整数序列对数组元素进行存取时，
# 将使用整数序列中的每个元素作为下标，整数序列可以是列表(list)或者数组(ndarray)。
# 使用整数序列作为下标获得的数组不和原始数组共享数据空间。
a = np.logspace(0, 9, 10, base=2)
print('a = ',a)
i = np.arange(0, 10, 2)
print('i = ',i)
# 利用i取a中的元素
b = a[i]
print('b = ',b)
# b的元素更改，a中元素不受影响
b[2] = 1.6
print('b = ',b)
print('a = ',a)