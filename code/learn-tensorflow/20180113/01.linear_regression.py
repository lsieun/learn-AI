# 线性回归
import tensorflow as tf
import numpy as np

# 生成100个随机点
x_data = np.random.rand(100)
y_data = x_data * 0.2 + 0.3

# 构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

# 二次loss函数
loss = tf.reduce_mean(tf.square(y_data -y))

# 定义一个梯度下降法来训练
optimizer = tf.train.GradientDescentOptimizer(0.2)

# 最小化loss函数
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for i in range(200):
        sess.run(train)
        if i % 20 == 0:
            print(i,sess.run([k,b]))