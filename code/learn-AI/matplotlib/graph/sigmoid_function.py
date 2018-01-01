import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 1 / (1 + np.exp(-x))

# Return evenly spaced numbers over a specified interval.
xdata = np.linspace(-8, 8, 960,endpoint=True)
ydata = func(xdata)

plt.plot(xdata,ydata)

plt.show()