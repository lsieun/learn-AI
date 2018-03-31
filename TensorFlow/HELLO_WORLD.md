# TENSORFLOW'S HELLO WORLD #

## Importing TensorFlow ##

To use TensorFlow, we need to import the library. We imported it and optionally gave it the name "tf", so the modules can be accessed by **tf.module-name**:

```python
import tensorflow as tf
```

## Building a Graph ##

TensorFlow defines computations as Graphs, and these are made with operations (also know as “ops”). So, when we work with TensorFlow, it is the same as defining a series of operations in a Graph.

To execute these operations as computations, we must launch the Graph into a Session. The session translates and passes the operations represented into the graphs to the device you want to execute them on, be it a GPU or CPU.

```python
import tensorflow as tf

a = tf.constant([2])
b = tf.constant([3])
c = tf.add(a, b)

session = tf.Session()
result = session.run(c)
print('result = ', result)
session.close()
```
To avoid having to close sessions every time, we can define them in a **with** block, so after running the **with** block the session will close automatically:

```python
with tf.Session() as session:
    result = session.run(c)
    print('result = ', result)
```

## Defining multidimensional arrays using TensorFlow ##

```python
import tensorflow as tf

Scalar = tf.constant([2])
Vector = tf.constant([5,6,2])
Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Tensor = tf.constant([ [[1,2,3],[2,3,4],[3,4,5]],
                       [[4,5,6],[5,6,7],[6,7,8]],
                       [[7,8,9],[8,9,10],[9,10,11]] ])
with tf.Session() as session:
    result = session.run(Scalar)
    print("Scalar (1 entry):\n %s \n" % result)
    result = session.run(Vector)
    print("Vector (3 entries) :\n %s \n" % result)
    result = session.run(Matrix)
    print("Matrix (3x3 entries):\n %s \n" % result)
    result = session.run(Tensor)
    print("Tensor (3x3x3 entries) :\n %s \n" % result)
```

```python
import tensorflow as tf

Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

first_operation = tf.add(Matrix_one, Matrix_two)
second_operation = Matrix_one + Matrix_two

with tf.Session() as session:
    result = session.run(first_operation)
    print("Defined using tensorflow function :")
    print(result)
    result = session.run(second_operation)
    print("Defined using normal expressions :")
    print(result)
```

```python
import tensorflow as tf

Matrix_one = tf.constant([[2,3],[3,4]])
Matrix_two = tf.constant([[2,3],[3,4]])

first_operation = tf.matmul(Matrix_one, Matrix_two)
second_operation = Matrix_one * Matrix_two

with tf.Session() as session:
    result = session.run(first_operation)
    print("Defined using tensorflow function :")
    print(result)
    result = session.run(second_operation)
    print("Defined using normal expressions :")
    print(result)
```











