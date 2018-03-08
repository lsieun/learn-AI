# Numpy #

[Numpy tutorial](http://www.labri.fr/perso/nrougier/teaching/numpy/numpy.html)

[100 numpy exercises](http://www.labri.fr/perso/nrougier/teaching/numpy.100/index.html)


### 计算指数 ###

np.exp

```python
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 1 / (1 + np.exp(-x))

# Return evenly spaced numbers over a specified interval.
xdata = np.linspace(-8, 8, 160,endpoint=True)
ydata = func(xdata)

plt.plot(xdata,ydata)

plt.show()
```

## 100 numpy exercises ##

Import the numpy package under the name np (★☆☆)

	import numpy as np

Create a null vector of size 10 (★☆☆)

	Z = np.zeros(10)

Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

	Z = np.zeros(10)
	Z[4] = 1

Create a vector with values ranging from 10 to 49 (★☆☆)

	Z = np.arange(10,50)

Reverse a vector (first element becomes last) (★☆☆)

	Z = np.arange(50)
	Z = Z[::-1]

Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

	Z = np.arange(9).reshape(3,3)

Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)

	nz = np.nonzero([1,2,0,0,4,0])

Create a 3x3 identity matrix (★☆☆) 单位矩阵

	Z = np.eye(3)

Create a 3x3x3 array with random values (★☆☆)

	Z = np.random.random((3,3,3))

Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

	Z = np.random.random((10,10))
	Zmin, Zmax = Z.min(), Z.max()
	print(Zmin, Zmax)

Create a random vector of size 30 and find the mean value (★☆☆)

	Z = np.random.random(30)
	m = Z.mean()

Create a 2d array with 1 on the border and 0 inside (★☆☆)

	Z = np.ones((10,10))
	Z[1:-1,1:-1]=0

What is the result of the following expression? (★☆☆)

	0 * np.nan       #nan
	np.nan == np.nan #False
	np.inf > np.nan  #False
	np.nan - np.nan  #nan
	0.3 == 3 * 0.1   #False
	0.3 == 0.3       #True
	3 * 0.1          #0.30000000000000004
	3 * 0.1 == 3 * 0.1 #True

Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

	Z = np.diag(1+np.arange(4),k=-1)

输出：

	array([[0, 0, 0, 0, 0],
	       [1, 0, 0, 0, 0],
	       [0, 2, 0, 0, 0],
	       [0, 0, 3, 0, 0],
	       [0, 0, 0, 4, 0]])

Create a 8x8 matrix and fill it with a checkerboard(西洋跳棋盘) pattern (★☆☆)

	Z = np.zeros((8,8),dtype=int)
	Z[1::2,::2] = 1
	Z[::2,1::2] = 1
	print(Z)

输出：

	array([[0, 1, 0, 1, 0, 1, 0, 1],
	       [1, 0, 1, 0, 1, 0, 1, 0],
	       [0, 1, 0, 1, 0, 1, 0, 1],
	       [1, 0, 1, 0, 1, 0, 1, 0],
	       [0, 1, 0, 1, 0, 1, 0, 1],
	       [1, 0, 1, 0, 1, 0, 1, 0],
	       [0, 1, 0, 1, 0, 1, 0, 1],
	       [1, 0, 1, 0, 1, 0, 1, 0]])

Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? 不懂了

	print(np.unravel_index(100,(6,7,8)))

输出:

	(1, 5, 4)

Create a checkerboard 8x8 matrix using the tile function (★☆☆)

	Z = np.tile( np.array([[0,1],[1,0]]), (4,4))

输出:

	array([[0, 1, 0, 1, 0, 1, 0, 1],
	       [1, 0, 1, 0, 1, 0, 1, 0],
	       [0, 1, 0, 1, 0, 1, 0, 1],
	       [1, 0, 1, 0, 1, 0, 1, 0],
	       [0, 1, 0, 1, 0, 1, 0, 1],
	       [1, 0, 1, 0, 1, 0, 1, 0],
	       [0, 1, 0, 1, 0, 1, 0, 1],
	       [1, 0, 1, 0, 1, 0, 1, 0]])

Normalize a 5x5 random matrix (★☆☆)

	Z = np.random.random((5,5))
	Zmax, Zmin = Z.max(), Z.min()
	Z = (Z - Zmin)/(Zmax - Zmin)
	print(Z)

Create a custom dtype that describes a color as four unisgned bytes (RGBA) (★☆☆)


## 2018-03-04 ##

### 生成矩阵np.ones ###

```python
def ones(shape, dtype=None, order='C'):
```

示例1：
```python
import numpy as np

a = np.ones(10)
print(a)
```
输出1：

	[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]


示例2：输出一个3行4列全是1的矩阵
```python
import numpy as np

b = np.ones((3,4))
print(b)
```
输出2：

	[[ 1.  1.  1.  1.]
	 [ 1.  1.  1.  1.]
	 [ 1.  1.  1.  1.]]

### 生成一个范围的值np.arange ###

方法定义：

```python
def arange(start=None, stop=None, step=None, dtype=None):
```

示例1：输入1个参数
```python
import numpy as np

a = np.arange(10)
print(a)
```

输出1：

	[0 1 2 3 4 5 6 7 8 9]

示例2：输入2个参数

```python
import numpy as np

b = np.arange(5,10)
print(b)
```

输出2：

	[5 6 7 8 9]

示例3：输入3个参数，加上步长

```python
import numpy as np

c = np.arange(5,10,2)
print(c)
```

输出3：

	[5 7 9]

示例4：进行reshape

reshape的定义如下：

```python
def reshape(self, shape, *shapes, order='C')
```

```python
import numpy as np

d = np.arange(0,8).reshape((2,4))
print(d)
```

输出4：

	[[0 1 2 3]
	 [4 5 6 7]]

### 合并矩阵np.c_ ###

使用`np.ones`生成一个2行4列的矩阵放到a中，接着将2行1列的矩阵放到b中，最后使用`np.c_`将a和b组成一个2行5列的新矩阵c。

```python
import numpy as np

a = np.ones((2,4))
print(a)
print("="*20)
b = np.arange(4,6).reshape((2,1))
print(b)
print("="*20)
c = np.c_[a,b]
print(c)
```

输出：

	[[ 1.  1.  1.  1.]
	 [ 1.  1.  1.  1.]]
	====================
	[[4]
	 [5]]
	====================
	[[ 1.  1.  1.  1.  4.]
	 [ 1.  1.  1.  1.  5.]]


其实，还有一个`np.r_`能够对两个矩阵进行按“行”合并。

```python
import numpy as np

a = np.ones((2,4))
print(a)
print("="*20)
b = np.arange(4,8).reshape((1,4))
print(b)
print("="*20)
c = np.r_[a,b]
print(c)
```

输出：

	[[ 1.  1.  1.  1.]
	 [ 1.  1.  1.  1.]]
	====================
	[[4 5 6 7]]
	====================
	[[ 1.  1.  1.  1.]
	 [ 1.  1.  1.  1.]
	 [ 4.  5.  6.  7.]]

这样看来的话，`np.c_`中的`c`应该就是指column，即按照“列”进行合并，而`np.r_`中的`r`应该代表row，即按照行进行合并。

### 数学np.ceil ###

示例：
```python
import numpy as np

a = 10
b = 3
c = np.ceil(a/b)
print('type(c) = ',type(c))
print(c)
```

输出：

	type(c) =  <class 'numpy.float64'>
	4.0


```python

```

## 2018.03.08 ##

### np.mgrid ###

主要作用是：Construct a multi-dimensional "meshgrid".

mgrid的step参数不使用complex number，此时不包括stop value

	If the step length is not a complex number, then the **stop** is not inclusive.

mgrid的step参数使用complex number，就代表the number of points，并且包括stop value

    However, if the step length is a **complex number** (e.g. 5j), then the
    integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.



```python
import numpy as np

if __name__ == "__main__":
    x1, y1 = np.mgrid[-10:10:3,-10:10:3]
    x2, y2 = np.mgrid[-10:10:3j,-10:10:3j]
    print(x1)
    print("-"*40)
    print(y1)
    print("-"*40)
    print(x2)
    print("-"*40)
    print(y2)
    print("-"*40)
```

输出如下：

	[[-10 -10 -10 -10 -10 -10 -10]
	 [ -7  -7  -7  -7  -7  -7  -7]
	 [ -4  -4  -4  -4  -4  -4  -4]
	 [ -1  -1  -1  -1  -1  -1  -1]
	 [  2   2   2   2   2   2   2]
	 [  5   5   5   5   5   5   5]
	 [  8   8   8   8   8   8   8]]
	----------------------------------------
	[[-10  -7  -4  -1   2   5   8]
	 [-10  -7  -4  -1   2   5   8]
	 [-10  -7  -4  -1   2   5   8]
	 [-10  -7  -4  -1   2   5   8]
	 [-10  -7  -4  -1   2   5   8]
	 [-10  -7  -4  -1   2   5   8]
	 [-10  -7  -4  -1   2   5   8]]
	----------------------------------------
	[[-10. -10. -10.]
	 [  0.   0.   0.]
	 [ 10.  10.  10.]]
	----------------------------------------
	[[-10.   0.  10.]
	 [-10.   0.  10.]
	 [-10.   0.  10.]]
	----------------------------------------

注意：在上述输出结果中，x1、x2的增长方向是按行增加，而y1、y2是按列增加

### np.mgrid和np.stack生成表格 ###

```python
import numpy as np

if __name__ == "__main__":
    x, y = np.mgrid[-2:2:3j,-2:2:3j]
    print(x)
    print("-"*40)
    print(y)
    print("-"*40)
    grid_test = np.stack((x.flat,y.flat), axis=1)
    print(grid_test)
```

输出结果：

	[[-2. -2. -2.]
	 [ 0.  0.  0.]
	 [ 2.  2.  2.]]
	----------------------------------------
	[[-2.  0.  2.]
	 [-2.  0.  2.]
	 [-2.  0.  2.]]
	----------------------------------------
	[[-2. -2.]
	 [-2.  0.]
	 [-2.  2.]
	 [ 0. -2.]
	 [ 0.  0.]
	 [ 0.  2.]
	 [ 2. -2.]
	 [ 2.  0.]
	 [ 2.  2.]]


```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```




