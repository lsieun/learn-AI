import tensorflow as tf

v = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32, shape=(1,3), name="v")
log_value = tf.log(v, name="log_value")
print("v = ", v)
print("log_value = ", log_value)

with tf.Session() as sess:
    result = log_value.eval()
    print(result)