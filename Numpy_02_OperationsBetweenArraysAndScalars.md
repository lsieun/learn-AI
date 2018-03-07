# Operations between Arrays and Scalars #

其实只讲两种，

- ndarray和ndarray之间
- ndarray和scalar之间

scalar和scalar之间没有可讲的东西。

ndarray很重要，是因为它不用写loops循环操作就能进行批量操作，这通常叫作vectorization。为什么叫vectorization呢？我是这么理解的，把ndarray中所有数据当成一个vector对象，就可以按照矢量的方法来计算了。

当两个equal-size arrays之间进行arithmetic操作，其实是对ndarray中的元素依次进行操作。

	Arrays are important because they enable you to express batch operations on data
	without writing any for loops. This is usually called vectorization. Any arithmetic op-
	erations between equal-size arrays applies the operation elementwise:

当ndarray与标量（scalars）进行操作时，会将标量（scalars）的value传播到ndarray上的每个元素进行操作。

	Arithmetic operations with scalars are as you would expect, propagating the value to
	each element:

```python
import numpy as np

arr = np.array([[1., 2.,3.],[4.,5.,6.]])
print(arr)
print("-"*40)

print(arr * arr)
print("-"*40)

print(arr - arr)
print("-"*40)

print(1/arr)
print("-"*40)

print(arr * 0.5)
```

