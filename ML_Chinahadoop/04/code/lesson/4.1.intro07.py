#coding:utf-8

import numpy as np

# 3.存取
# 3.1常规办法：数组元素的存取方法和Python的标准方法相同
a = np.arange(10)
print('a = ',a)
# 获取某个元素
print('a[3] = ',a[3])
# 切片[3,6)，左闭右开
print('a[3:6] = ',a[3:6])
# 省略开始下标，表示从0开始
print('a[:5] = ',a[:5])
# 下标为负表示从后向前数
print('a[3:] = ',a[3:])
# 步长为2
print('a[1:9:2] = ',a[1:9:2])

# 步长为-1，即翻转
print('a[::-1] = ',a[::-1])
# 切片数据是原数组的一个视图，与原数组共享内容空间，可以直接修改元素值
a[1:4] = 10, 20, 30
print('a = ',a)
# 因此，在实践中，切实注意原始数据是否被破坏，如：
b = a[2:5]
b[0] = 200
print('a = ',a)