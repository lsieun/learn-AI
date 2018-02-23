#coding:utf-8

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

np.set_printoptions(linewidth=200,suppress=True)

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
mpl.rcParams['axes.unicode_minus'] = False

# 6.4 直方图的使用
mu = 2
sigma = 3
data = mu + sigma * np.random.randn(1000)
h = plt.hist(data, 30, normed=1, color='#FFFFA0')
x = h[1]
y = norm.pdf(x, loc=mu, scale=sigma)
plt.plot(x, y, 'r-', x, y, 'ro', linewidth=2, markersize=4)
plt.grid()
plt.show()