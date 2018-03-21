import tensorflow as tf

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32, shape=(2,3), name="v")
reduce_mean_op = tf.reduce_mean(v)

with tf.Session() as sess:
    result = reduce_mean_op.eval()
    print("result = ", result)