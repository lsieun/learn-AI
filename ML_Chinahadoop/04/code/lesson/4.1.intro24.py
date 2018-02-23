#coding:utf-8

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200,suppress=True)

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
mpl.rcParams['axes.unicode_minus'] = False

# 6.2 验证中心极限定理
t = 1000
a = np.zeros(10000)
for i in range(t):
    a += np.random.uniform(-5, 5, 10000)
a /= t
plt.hist(a, bins=30, color='g', alpha=0.5, normed=True, label=u'均匀分布叠加')
plt.legend(loc='upper left')
plt.grid()
plt.show()