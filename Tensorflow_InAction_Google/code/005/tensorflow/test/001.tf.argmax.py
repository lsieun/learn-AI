import tensorflow as tf

a = tf.constant([1.,2.,3.,0.,9.,5.])
b = tf.constant([[1,2,3],[3,2,1],[4,5,6],[6,5,4]])

argmax_a_0_op = tf.argmax(a, 0)
# argmax_a_1_op = tf.argmax(a, 1)  # InvalidArgumentError
argmax_b_0_op = tf.argmax(b, 0)
argmax_b_1_op = tf.argmax(b, 1)

with tf.Session() as sess:
    argmax_a_0 = argmax_a_0_op.eval()
    argmax_b_0 = argmax_b_0_op.eval()
    argmax_b_1 = argmax_b_1_op.eval()
    print("a = ", a.eval())
    print("argmax_a_0 = ", argmax_a_0)
    print("b = \n", b.eval())
    print("argmax_b_0 = ", argmax_b_0)
    print("argmax_b_1 = ", argmax_b_1)


