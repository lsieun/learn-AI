import tensorflow as tf

weights = tf.constant([[1.0, 2.0], [-3.0, 4.0]], dtype=tf.float32, shape=(2,2), name="weights")
# 输出为 (|1|+|-2|+|-3|+|4|) * 0.5 = 5。其中，0.5为正则化项的权重
l1_op = tf.contrib.layers.l1_regularizer(.5)(weights)
# 输出为 ((1)^{2} + (-2)^{2} + (-3)^{2} + (4)^{2}) / 2 * 0.5 = 7.5。
# Tensorflow会将L2的正则化损失除以2使得求导得到的结果更加简洁。
l2_op = tf.contrib.layers.l2_regularizer(.5)(weights)

with tf.Session() as sess:
    l1_value = l1_op.eval()
    l2_value = l2_op.eval()
    print("l1_value = ", l1_value)
    print("l2_value = ", l2_value)