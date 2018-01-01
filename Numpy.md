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








