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


















