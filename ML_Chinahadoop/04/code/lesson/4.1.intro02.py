#coding:utf-8

import numpy as np

#知识点：通过numpy的array方法将python的list对象转换成NDArray对象

# 正式开始  -:)
# 标准Python的列表(list)中，元素本质是对象。
# 如：L = [1, 2, 3]，需要3个指针和三个整数对象，对于数值运算比较浪费内存和CPU。
# 因此，Numpy提供了ndarray(N-dimensional array object)对象：存储单一数据类型的多维数组。

# # 1.使用array创建
# 通过array函数传递list对象
L = [1, 2, 3, 4, 5, 6]
print("L = ", L)
a = np.array(L)
print("a = ", a)
print(type(a), type(L))
print("="*40)
# # 若传递的是多层嵌套的list，将创建多维数组
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(b)
print("="*40)
#
# # # # # 数组大小可以通过其shape属性获得
print(a.shape)
print(b.shape)
print("="*40)
# # # 也可以强制修改shape
b.shape = 4, 3
print(b)
# # 注：从(3,4)改为(4,3)并不是对数组进行转置，而只是改变每个轴的大小，数组元素在内存中的位置并没有改变