#coding:utf-8

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200,suppress=True)

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
mpl.rcParams['axes.unicode_minus'] = False

# x ** x        x > 0
# (-x) ** (-x)  x < 0
def f(x):
    y = np.ones_like(x)
    i = x > 0
    y[i] = np.power(x[i], x[i])
    i = x < 0
    y[i] = np.power(-x[i], -x[i])
    return y

# 5.3 x^x
plt.figure(facecolor='w')
x = np.linspace(-1.3, 1.3, 101)
y = f(x)
plt.plot(x, y, 'g-', label='x^x', linewidth=2)
plt.grid()
plt.legend(loc='upper left')
plt.show()