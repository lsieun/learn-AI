
## 第01个示例：打印tensorflow的版本 ##

```python
import tensorflow

print(tensorflow.__version__)
```
## 第02个示例：计算f=xxy+y+2 ##

只有`session`内才执行代码

示例代码：

```python
import tensorflow as tf

x = tf.Variable(3,name="x")
y = tf.Variable(2,name="y")
f = x * x * y + y + 2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
sess.close()
print("result = ",result)
```

输出：

	result =  22

接下来，我们看一看x,y,f的**类型**

```python
import tensorflow as tf

x = tf.Variable(3,name="x")
y = tf.Variable(2,name="y")
f = x * x * y + y + 2
print("type(x) = ",type(x))
print("type(y) = ",type(y))
print("type(f) = ",type(f))

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print("type(x.initializer) = ",type(x.initializer))
print("type(y) = ",type(y))
print("type(f) = ",type(f))
sess.close()
print("type(result) = ",type(result))
print("result = ",result)
```

输出：

	type(x) =  <class 'tensorflow.python.ops.variables.Variable'>
	type(y) =  <class 'tensorflow.python.ops.variables.Variable'>
	type(f) =  <class 'tensorflow.python.framework.ops.Tensor'>
	type(x.initializer) =  <class 'tensorflow.python.framework.ops.Operation'>
	type(y) =  <class 'tensorflow.python.ops.variables.Variable'>
	type(f) =  <class 'tensorflow.python.framework.ops.Tensor'>
	type(result) =  <class 'numpy.int32'>
	result =  22

通过上面的执行结果，可以看到x,y是Variable类型，而f是Tensor类型。（忽然想起，上课讲的：tensor就是tensorflow中流动的数据。

再查看一下x,y,f的**值**

```python
import tensorflow as tf

x = tf.Variable(3,name="x")
y = tf.Variable(2,name="y")
f = x * x * y + y + 2
print('x = ',x)
print('y = ',y)
print('f = ',f)

sess = tf.Session()
a = sess.run(x.initializer)
b = sess.run(y.initializer)
result = sess.run(f)
print('x = ',x)
print('y = ',y)
print('f = ',f)
print('a = ',a)
print('b = ',b)
sess.close()
print("result = ",result)
```

输出：

	x =  <tf.Variable 'x:0' shape=() dtype=int32_ref>
	y =  <tf.Variable 'y:0' shape=() dtype=int32_ref>
	f =  Tensor("add_1:0", shape=(), dtype=int32)
	x =  <tf.Variable 'x:0' shape=() dtype=int32_ref>
	y =  <tf.Variable 'y:0' shape=() dtype=int32_ref>
	f =  Tensor("add_1:0", shape=(), dtype=int32)
	a =  None
	b =  None
	result =  22

通过这个示例，可以看到x,y,f并没有被真正赋值，即使在session中直接打印。只有当调用`session.run`时才能读取。其中，a,b都是None，我想大概是因为只是初始化值吧，而并没有返回值。

再来看，如果不对x,y进行调用`sess.run(x.initializer)`会怎么样。

```python
import tensorflow as tf

x = tf.Variable(3,name="x")
y = tf.Variable(2,name="y")
f = x * x * y + y + 2

sess = tf.Session()
# sess.run(x.initializer)  # 注意：这里注释掉了
# sess.run(y.initializer)  # 注意：这里注释掉了
result = sess.run(f)
sess.close()
print("result = ",result)
```

输出：

	tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value y

由此可见，不进行初始化是不行的。

## 第03个示例：在with块内使用tf.Session() ##

换一种形式，使用`with tf.Session() as sess:`，不需要调用`session.close()`方法。
```python
import tensorflow as tf

x = tf.Variable(3,name="x")
y = tf.Variable(2,name="y")
f = x * x * y + y + 2

with tf.Session() as sess:
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
    print("result = ",result)
```

输出：

	result =  22

## 第04个示例：统一对变量进行初始化 ##

统一对变量进行初始化，由有两行代码来完成：`init = tf.global_variables_initializer()`和`init.run()`。

其中，`init = tf.global_variables_initializer()`是在session外面，也就是在**构建图**的阶段。而`init.run()`是在session内，是在**运行图**的阶段。



```python
import tensorflow as tf

x = tf.Variable(3,name="x")
y = tf.Variable(2,name="y")
f = x * x * y + y + 2

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()
    print("result = ",result)
```

还有一点值得注意：在with/session块内，并没有明确的使用sess变量，而是使用了`init.run()`和`f.eval()`来完成的。事实上，`init.run()`和`sess.run(init)`是等价的，`f.eval()`和`sess.run(f)`是等价的。因此上面的代码和下面的代码执行效果是一样的。

```python
import tensorflow as tf

x = tf.Variable(3,name="x")
y = tf.Variable(2,name="y")
f = x * x * y + y + 2

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # init.run()          # 注意，这里注释掉了
    sess.run(init)
    # result = f.eval()   # 注意，这里注释掉了
    result = sess.run(f)
    print("result = ",result)
```

## 第05个示例：使用InteractiveSession() ##

InteractiveSession和常规的Session不同在于，自动默认设置它自己为默认的session，即无需放在with块中了，但是这样需要自己来close session。

```python
import tensorflow as tf

# 第一阶段
x = tf.Variable(3,name="x")
y = tf.Variable(2,name="y")
f = x * x * y + y + 2

init = tf.global_variables_initializer()

# 第二阶段
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print("result = ",result)
sess.close()
```

TensorFlow程序会典型的分为两部分，第一部分是创建计算图，叫做**构建阶段**，这一阶段通常建立表示机器学习模型的的计算图，和需要去训练模型的计算图；第二部分是**执行阶段**，执行阶段通常运行Loop循环重复训练步骤，每一步训练小批量数据，逐渐的改进模型参数。

## 第05个示例：tensorflow与图 ##

这个例子主要是演示了变量x1,x2,x3与graph之间的关系。默认情况下，变量是放在默认图`tf.get_default_graph()`中的，而x3通过with块（`with graph.as_default():`）放到了graph当中。

```python
import tensorflow as tf

# 任何创建的节点会自动加入到默认的图
x1 = tf.Variable(1,name="x1")
print(x1.graph is tf.get_default_graph())   # True

# 大多数情况下上面运行的很好，有时候或许想要管理多个独立的图
# 可以创建一个新的图并且临时使用with块是的它成为默认的图
graph = tf.Graph()
x2 = tf.Variable(2,name="x2")
with graph.as_default():
    x3 = tf.Variable(3,name="x3")

print(x2.graph is graph)                    # False
print(x2.graph is tf.get_default_graph())   # True
print(x3.graph is graph)                    # True
print(x3.graph is tf.get_default_graph())   # False
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

![](images/20180315135133.gif)


