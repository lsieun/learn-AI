#coding:utf-8

import numpy as np

np.set_printoptions(linewidth=200,suppress=True)

# 4.2 元素去重
# 4.2.1直接使用库函数
a = np.array((1, 2, 3, 4, 5, 5, 7, 3, 2, 2, 8, 8))
print('原始数组：', a)
# 使用库函数unique
b = np.unique(a)
print('去重后：', b)
# 4.2.2 二维数组的去重，结果会是预期的么？
c = np.array(((1, 2), (3, 4), (5, 6), (1, 3), (3, 4), (7, 6)))
print('二维数组：\n', c)
print('去重后：', np.unique(c))
# # # 4.2.3 方案1：转换为虚数
r, i = np.split(c, (1,), axis=1)
x = r + i * 1j
# x = c[:, 0] + c[:, 1] * 1j
print('转换成虚数：', x)
print('虚数去重后：', np.unique(x))
print(np.unique(x, return_index=True))  # 思考return_index的意义
idx = np.unique(x, return_index=True)[1]
print('二维数组去重：\n', c[idx])
# # 4.2.3 方案2：利用set
print('去重方案2：\n', np.array(list(set([tuple(t) for t in c]))))