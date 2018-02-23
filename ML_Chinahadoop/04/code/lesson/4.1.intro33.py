import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy import stats
from scipy.stats import norm, poisson
from scipy.interpolate import BarycentricInterpolator
import scipy.optimize as opt

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

# # 8.2 使用scipy计算函数极值
a = opt.fmin(f, 1)
b = opt.fmin_cg(f, 1)
c = opt.fmin_bfgs(f, 1)
print(a, 1/a, math.e)
print(b)
print(c)

# marker	description
# ”.”	point
# ”,”	pixel
# “o”	circle
# “v”	triangle_down
# “^”	triangle_up
# “<”	triangle_left
# “>”	triangle_right
# “1”	tri_down
# “2”	tri_up
# “3”	tri_left
# “4”	tri_right
# “8”	octagon
# “s”	square
# “p”	pentagon
# “*”	star
# “h”	hexagon1
# “H”	hexagon2
# “+”	plus
# “x”	x
# “D”	diamond
# “d”	thin_diamond
# “|”	vline
# “_”	hline
# TICKLEFT	tickleft
# TICKRIGHT	tickright
# TICKUP	tickup
# TICKDOWN	tickdown
# CARETLEFT	caretleft
# CARETRIGHT	caretright
# CARETUP	caretup
# CARETDOWN	caretdown