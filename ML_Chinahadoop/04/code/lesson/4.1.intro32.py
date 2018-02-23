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

def residual2(t, x, y):
    print(t[0], t[1])
    return y - (t[0]*np.sin(t[1]*x) + t[2])

# # 线性回归例2
x = np.linspace(0, 5, 100)
a = 5
w = 1.5
phi = -2
y = a * np.sin(w*x) + phi + np.random.rand(len(x))*0.5

t = leastsq(residual2, [3, 5, 1], args=(x, y))
theta = t[0]
print('真实值：', a, w, phi)
print('预测值：', theta)
y_hat = theta[0] * np.sin(theta[1] * x) + theta[2]
plt.plot(x, y, 'r-', linewidth=2, label='Actual')
plt.plot(x, y_hat, 'g-', linewidth=2, label='Predict')
plt.legend(loc='lower left')
plt.grid()
plt.show()