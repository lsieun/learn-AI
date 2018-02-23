#coding:utf-8

import numpy as np

b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(b)
print("="*40)


#知识点：（1）修改shape属性和调用reshape()方法;（2）-1的使用;(3)共享内存

# # # 当某个轴为-1时，将根据数组元素的个数自动计算此轴的长度
b.shape = 2, -1
print(b)
print(b.shape)
print("="*40)
#
b.shape = 3, 4
print(b)
print("="*40)

# # # # 使用reshape方法，可以创建改变了尺寸的新数组，原数组的shape保持不变
c = b.reshape((4, -1))
print("b = \n", b)
print('c = \n', c)
print("="*40)


# # # # 数组b和c共享内存，修改任意一个将影响另外一个
b[0][1] = 20
# print("b = \n", b)
# print("c = \n", c)