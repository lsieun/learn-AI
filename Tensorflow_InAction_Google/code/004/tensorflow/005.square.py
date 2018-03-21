import tensorflow as tf

v1 = tf.constant([1.0, 2.0], dtype=tf.float32, shape=(1,2), name="v1")
v2 = tf.constant([3.0, 4.0], dtype=tf.float32, shape=(1,2), name="v2")
square_op = tf.square(v1 - v2)

with tf.Session() as sess:
    result = square_op.eval()
    print("result = ", result)