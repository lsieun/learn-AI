import numpy as np

x = 2 * np.random.rand(100,1)
y = 4 + 3 * x + np.random.randn(100,1)
X_b = np.c_[np.ones((100,1)),x]
# print(X_b)

learning_rate = 0.3
n_iterations = 1000
m = 100

# myr = np.random
# print(type(myr))
theta = np.random.randn(2,1)

for iteration in range(n_iterations):
    gradients = 1 / m * X_b.T.dot(X_b.dot(theta)-y)
    theta = theta - learning_rate * gradients

print(theta)
