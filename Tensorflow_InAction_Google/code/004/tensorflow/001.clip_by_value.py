import tensorflow as tf

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32,shape=(2,3),name="v")
clip_op = tf.clip_by_value(v, clip_value_min=2.5, clip_value_max=4.5, name="clip_op")
print("clip_op = ", clip_op)

with tf.Session() as sess:
    result = clip_op.eval()
    print(result)