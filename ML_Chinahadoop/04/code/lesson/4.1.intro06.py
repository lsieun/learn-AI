#coding:utf-8

import numpy as np

# 2.使用函数创建
# 如果生成一定规则的数据，可以使用NumPy提供的专门函数
# arange函数类似于python的range函数：指定起始值、终止值和步长来创建数组
# 和Python的range类似，arange同样不包括终值；但arange可以生成浮点类型，而range只能是整数类型
np.set_printoptions(linewidth=100, suppress=True)
a = np.arange(1, 10, 0.5)
print('a = ', a)

# linspace函数通过指定起始值、终止值和元素个数来创建数组，缺省包括终止值
b = np.linspace(1, 10, 10)
print('b = ', b)

# 可以通过endpoint关键字指定是否包括终值
c = np.linspace(1, 10, 10, endpoint=False)
print('c = ', c)

# 和linspace类似，logspace可以创建等比数列
# 下面函数创建起始值为10^1，终止值为10^2，有10个数的等比数列
d = np.logspace(1, 4, 4, endpoint=True, base=2)
print('d = ', d)

# 下面创建起始值为2^0，终止值为2^10(包括)，有10个数的等比数列
f = np.logspace(0, 10, 11, endpoint=True, base=2)
print('f = ', f)

# 使用 frombuffer, fromstring, fromfile等函数可以从字节序列创建数组
s = 'abcdzzzz'
g = np.fromstring(s, dtype=np.int8)
print('g = ', g)