# coding:utf-8
#

import numpy as np
from scipy import stats

def calc_statistics(x):
    n = x.shape[0]  # 样本个数

    # 手动计算
    m = 0
    m2 = 0
    m3 = 0
    m4 = 0
    for t in x:
        m += t
        m2 += t*t
        m3 += t**3
        m4 += t**4
    m /= n
    m2 /= n
    m3 /= n
    m4 /= n

    mu = m
    sigma = np.sqrt(m2 - mu*mu)
    skew = (m3 - 3*mu*m2 + 2*mu**3) / sigma**3
    kurtosis = (m4 - 4*mu*m3 + 6*mu*mu*m2 - 4*mu**3*mu + mu**4) / sigma**4 - 3
    print('手动计算均值、标准差、偏度、峰度：', mu, sigma, skew, kurtosis)

def calc_statistics2(x):
    # 使用系统函数验证
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    skew = stats.skew(x)
    kurtosis = stats.kurtosis(x)
    print('函数库计算均值、标准差、偏度、峰度：', mu, sigma, skew, kurtosis)


if __name__ == '__main__':
    d = np.random.randn(5)
    print(d)
    print(d.shape)
    calc_statistics(d)
    calc_statistics2(d)