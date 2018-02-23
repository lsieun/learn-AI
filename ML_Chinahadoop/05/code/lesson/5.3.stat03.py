# coding:utf-8
#

import math
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

def calc_statistics(x):
    # 使用系统函数验证
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    skew = stats.skew(x)
    kurtosis = stats.kurtosis(x)
    return mu, sigma, skew, kurtosis

if __name__ == '__main__':
    d = np.random.randn(10000)
    print(d)
    print(d.shape)
    mu, sigma, skew, kurtosis = calc_statistics(d)
    print('函数库计算均值、标准差、偏度、峰度：', mu, sigma, skew, kurtosis)

    # 一维直方图
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(num=1, facecolor='w')
    y1, x1, dummy = plt.hist(d, bins=30, normed=True, color='g', alpha=0.75)
    print(x1.shape)
    print(y1.shape)
    print('x1 = ',x1)
    print('y1 = ',y1)
    print(x1.min(), x1.max(),y1.min(), y1.max())
    t = np.arange(x1.min(), x1.max(), 0.05)
    y = np.exp(-t**2 / 2) / math.sqrt(2*math.pi)
    plt.plot(t, y, 'r-', lw=2)
    plt.title('高斯分布，样本个数：%d' % d.shape[0])
    plt.grid(True)
    plt.show()