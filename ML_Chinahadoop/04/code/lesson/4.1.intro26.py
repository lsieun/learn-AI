#coding:utf-8

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

np.set_printoptions(linewidth=200,suppress=True)

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
mpl.rcParams['axes.unicode_minus'] = False

# 6.3 Poisson分布
x = np.random.poisson(lam=5, size=10000)
print(x)
pillar = 15
a = plt.hist(x, bins=pillar, normed=True, range=[0, pillar], color='g', alpha=0.5)
plt.grid()
plt.show()
print(a)
print(a[0].sum())