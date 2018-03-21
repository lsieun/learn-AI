import tensorflow as tf

v1 = tf.Variable(0, dtype=tf.float32, name="v1")
step = tf.Variable(0, trainable=False)

ema = tf.train.ExponentialMovingAverage(0.99, step)
maintain_averages_op = ema.apply([v1])

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    init_op.run()

    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(v1, 5))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

