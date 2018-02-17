# Python数据可视化：正态分布（高斯分布） #

正态分布（Normal distribution）又称为高斯分布（Gaussian distribution）

## 1、什么是高斯分布 ##

若随机变量![](http://latex.codecogs.com/gif.latex?X)服从一个数学期望为![](http://latex.codecogs.com/gif.latex?\mu)、标准方差为![](http://latex.codecogs.com/gif.latex?\sigma^{2})的高斯分布，记为：![](http://latex.codecogs.com/gif.latex?N\sim&space;N(\mu,\sigma^{2}))。

则其概率密度函数为：

![](http://latex.codecogs.com/gif.latex?f(x)=\frac{1}{\sqrt{2\pi}\sigma}&space;e&space;^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}})

正态分布的期望值![](http://latex.codecogs.com/gif.latex?\mu)决定了其位置，其标准差![](http://latex.codecogs.com/gif.latex?\sigma)决定了分布的幅度。因其曲线呈钟形，因此人们又经常称之为钟形曲线。我们通常所说的标准正态分布是![](http://latex.codecogs.com/gif.latex?\mu=0,\sigma=1)的正态分布： 

![](images/matplotlib/20180218012038.png)

## 2、Python代码画图实现高斯分布 ##

编程的时候，可以将![](http://latex.codecogs.com/gif.latex?x,\mu,\sigma)作为输入参数，那么概率密度函数变成这个样子：![](http://latex.codecogs.com/gif.latex?f(x,\mu,\sigma)=\frac{1}{\sqrt{2\pi}\sigma}&space;e&space;^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}})。

Python代码如下：

```python
import math
import numpy as np
import matplotlib.pyplot as plt

# Python实现正态分布
# 绘制正态分布概率密度函数

u = 0 # 均值μ
sigma = math.sqrt(1) # 标准差δ

x = np.linspace(u - 3*sigma, u + 3*sigma, 50)
y = np.exp(-(x - u) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma)

plt.plot(x, y, "r-", linewidth=2)
plt.grid(True)
plt.show()
```

## 3、画4条高斯分布曲线进行对比 ##

```python
import math
import numpy as np
import matplotlib.pyplot as plt

# Python实现正态分布
# 绘制正态分布概率密度函数

def plot_normal_distribution(u,sigma,color):
    x = np.linspace(u - 3*sigma, u + 3*sigma, 50)
    y = np.exp(-(x - u) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma)
    plt.plot(x, y, color, linewidth=2)

u01 = 0 # 均值μ
u02 = 0
u03 = 0
u04 = -2 #期望值左移
sigma01 = math.sqrt(1) # 标准差δ
sigma02 = math.sqrt(0.2) #变高变瘦
sigma03 = math.sqrt(5) #变矫变胖
sigma04 = math.sqrt(0.5)

plot_normal_distribution(u01,sigma01,"r-")
plot_normal_distribution(u02,sigma02,"g-")
plot_normal_distribution(u03,sigma03,"b-")
plot_normal_distribution(u04,sigma04,"m-")

plt.grid(True)
plt.show()
```

![](images/matplotlib/20180218015801.png)


## 4、参考 ##

[Python数据可视化](http://blog.csdn.net/column/details/13510.html)

[Markdown：插入数学公式](https://www.jianshu.com/p/fd97e1f8f699)



