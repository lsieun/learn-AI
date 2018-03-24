from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(train_dir="./path/to/MNIST_data/",one_hot=True)

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size=batch_size)

print("X shape: ", xs.shape)
print("Y shape: ", ys.shape)