#coding:utf-8

import numpy as np

np.set_printoptions(linewidth=200,suppress=True)

# # 3.2.2
# 使用布尔数组i作为下标存取数组a中的元素：返回数组a中所有在数组b中对应下标为True的元素
# 生成10个满足[0,1)中均匀分布的随机数
a = np.random.rand(10)
print(a)
# 大于0.5的元素索引
print(a > 0.5)
# 大于0.5的元素
b = a[a > 0.5]
print(b)
# 将原数组中大于0.5的元素截取成0.5
a[a > 0.5] = 0.5
print(a)
# b不受影响
print(b)