#coding:utf-8

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200,suppress=True)

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
mpl.rcParams['axes.unicode_minus'] = False

# 6. 概率分布
# 6.1 均匀分布
x = np.random.rand(10000)
t = np.arange(len(x))
# plt.hist(x, 30, color='m', alpha=0.5, label=u'均匀分布')
plt.plot(t, x, 'g.', label=u'均匀分布')
plt.legend(loc='upper left')
plt.grid()
plt.show()