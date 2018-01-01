import numpy as np
import matplotlib.pyplot as plt

# Return evenly spaced numbers over a specified interval.
X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
# print(type(X)) #<class 'numpy.ndarray'>
# print(X)

C,S = np.cos(X), np.sin(X)

plt.plot(X,C)
plt.plot(X,S)

plt.show()
