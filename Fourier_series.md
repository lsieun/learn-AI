# Fourier series #

法国数学家傅里叶发现，任何周期函数都可以用正弦函数和余弦函数构成的无穷级数来表示（选择正弦函数与余弦函数作为基函数是因为它们是正交的），后世称傅里叶级数为一种特殊的三角级数，根据欧拉公式，三角函数又能化成指数形式，也称傅立叶级数为一种指数级数。



## 一次谐波 ##


```python
def getY(x,T):
    if x < 0:
        if x >= -(T / 2):
            return x
        else:
            return getY(x+T,T)
    else:
        if x < (T/2):
            return x
        else:
            return getY(x-T,T)
# 界内值
print("getY(2, 5) = ", getY(2, 5))
print("getY(0, 5) = ", getY(0, 5))
print("getY(-2, 5) = ", getY(-2, 5))
print("="*40)
# 越界值
print("getY(7, 5) = ", getY(7, 5))
print("getY(-7, 5) = ", getY(-7, 5))
print("="*40)
# 边界值
print("getY(2.5, 5) = ", getY(2.5, 5))
print("getY(-2.5, 5) = ", getY(-2.5, 5))
print("="*40)
```

007.py

```python
import numpy as np
import matplotlib.pyplot as plt

def getY(x,T):
    if x < 0:
        if x >= -(T / 2):
            return x
        else:
            return getY(x+T,T)
    else:
        if x < (T/2):
            return x
        else:
            return getY(x-T,T)

X = np.linspace(-3 * np.pi, 3 * np.pi, 256)
(col,) = X.shape
print(col)

listY = []
for i in range(col):
    y = getY(X[i], 2 * np.pi)
    listY.append(y)
Y = np.array(listY)

sinY = 2 * np.sin(X) - np.sin(2 * X) + np.sin(3 * X) * 2 /3 - np.sin(4 * X) / 2 + np.sin(5 * X) * 2 / 5

plt.plot(X,Y)
plt.plot(X,sinY)
plt.show()
```


008.py

```python
import numpy as np
import matplotlib.pyplot as plt

def getY(x,T):
    if x < 0:
        if x >= -(T / 2):
            return x
        else:
            return getY(x+T,T)
    else:
        if x < (T/2):
            return x
        else:
            return getY(x-T,T)

def getSinY(X,n):
    sinY = np.zeros(X.shape[0])
    for i in range(n):
        sinY += np.sin((i + 1) * X) * 2 / (i + 1) * np.power(-1,i)
    return sinY

X = np.linspace(-3 * np.pi, 3 * np.pi, 256)
(col,) = X.shape
print(col)

listY = []
for i in range(col):
    y = getY(X[i], 2 * np.pi)
    listY.append(y)
Y = np.array(listY)

sinY = getSinY(X, 100)

plt.plot(X,Y)
plt.plot(X,sinY)
plt.show()
```















