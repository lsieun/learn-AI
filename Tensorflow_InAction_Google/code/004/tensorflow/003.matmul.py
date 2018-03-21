import tensorflow as tf

v1 = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32, shape=(2,2), name="v1")
v2 = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32, shape=(2,2), name="v2")

result1 = v1 * v2
result2 = tf.matmul(v1, v2)

with tf.Session() as sess:
    result1_value = result1.eval()
    result2_value = result2.eval()
    print("v1_value = \n", v1.eval())
    print("v2_value = \n", v2.eval())
    print("result1_value = \n", result1_value)
    print("result2_value = \n", result2_value)
