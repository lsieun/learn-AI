# coding:utf-8
#

import math
import numpy as np
from scipy import stats
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

def calc_statistics(x):
    # 使用系统函数验证
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    skew = stats.skew(x)
    kurtosis = stats.kurtosis(x)
    return mu, sigma, skew, kurtosis

if __name__ == '__main__':
    d = np.random.randn(100000, 2)
    mu, sigma, skew, kurtosis = calc_statistics(d)
    print('函数库计算均值、标准差、偏度、峰度：', mu, sigma, skew, kurtosis)
    # 二维图像
    N = 20
    density, edges = np.histogramdd(d, bins=[N, N])
    print('样本总数：', np.sum(density))
    density /= density.max()
    x = y = np.arange(N)
    print('x = ', x)
    print('y = ', y)
    t = np.meshgrid(x, y)
    print(t)
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(t[0], t[1], density, c='r', s=50*density, marker='o', depthshade=True)
    ax.plot_surface(t[0], t[1], density, cmap=cm.Accent, rstride=1, cstride=1, alpha=0.9, lw=0.75)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('二元高斯分布，样本个数：%d' % d.shape[0], fontsize=15)
    plt.tight_layout(0.1)
    plt.show()
