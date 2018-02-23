# coding:utf-8
#

import math
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 一维直方图
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(num=1, facecolor='w')
    t = np.arange(-4, 4, 0.05)
    y = np.exp(-t**2 / 2) / math.sqrt(2*math.pi)
    plt.plot(t, y, 'r-', lw=2)
    # plt.title('高斯分布，样本个数：%d' % d.shape[0])
    plt.grid(True)
    plt.show()