import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy import stats
from scipy.stats import norm, poisson
from scipy.interpolate import BarycentricInterpolator


np.set_printoptions(linewidth=200,suppress=True)

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
mpl.rcParams['axes.unicode_minus'] = False

def residual(t, x, y):
    return y - (t[0] * x ** 2 + t[1] * x + t[2])

# 8.1 scipy
# 线性回归例1
x = np.linspace(-2, 2, 50)
A, B, C = 2, 3, -1
y = (A * x ** 2 + B * x + C) + np.random.rand(len(x))*0.75

t = leastsq(residual, [0, 0, 0], args=(x, y))
theta = t[0]
print('真实值：', A, B, C)
print('预测值：', theta)
y_hat = theta[0] * x ** 2 + theta[1] * x + theta[2]
plt.plot(x, y, 'r-', linewidth=2, label=u'Actual')
plt.plot(x, y_hat, 'g-', linewidth=2, label=u'Predict')
plt.legend(loc='upper left')
plt.grid()
plt.show()