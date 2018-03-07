# Indexing and Slicing #

**NumPy array indexing** is a rich topic, as there are many ways you may want to select
**a subset of your data** or **individual elements**. 

选取数据的几种形式：

- scalar
- slice
- boolean arrays
- integer arrays

## 一维ndarray的索引与切片 ##

One-dimensional arrays are simple; on
the surface they act similarly to Python lists:

```python
import numpy as np

arr = np.arange(10)

print(arr)
print(arr[5])
print(arr[5:8])

arr[5:8]=12
print(arr)

arr_slice = arr[5:8]
arr_slice[1] = 12345
print(arr)

arr_slice[:] = 64
print(arr)
```

An important first dis-
tinction from lists is that array slices are views on the original array. This means that
the data is not copied, and any modifications to the view will be reflected in the source
array.

If you are new to NumPy, you might be surprised by this, especially if they have used
other array programming languages which copy data more zealously. As NumPy has
been designed with **large data use cases** in mind, you could imagine performance and
memory problems if NumPy insisted on copying data left and right.

If you want a **copy** of a slice of an ndarray instead of a **view**, you will
need to explicitly copy the array; for example `arr[5:8].copy()`.

## 高维ndarray的索引与切片 ##

With higher dimensional arrays, you have many more options. In a two-dimensional
array, the elements at **each index** are no longer **scalars** but rather **one-dimensional
arrays**.

Thus, individual elements can be accessed recursively. But that is a bit too much work,
so you can pass **a comma-separated list of indices** to **select individual elements**.

In multidimensional arrays, if you **omit later indices**, the returned object will be **a lower-
dimensional ndarray** consisting of all the data along the higher dimensions.

```python
import numpy as np

arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2d[2])     #这里得到是一维ndarray对象

print(arr2d[0][2])  # 这两种是等价的
print(arr2d[0,2])   # 这两种是等价的
print("-"*40)

arr3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(arr3d)
print("-"*40)
print(arr3d[0])   # 这里是得到一个二维数组
print("-"*40)

old_values = arr3d[0].copy()
arr3d[0] = 42   # Both scalar values and arrays can be assigned to arr3d[0]:
print(arr3d)
print("-"*40)
arr3d[0] = old_values  #Both scalar values and arrays can be assigned to arr3d[0]:
print(arr3d)
print("-"*40)

print(arr3d[1,0])
```

## Indexing with slices ##

Like **one-dimensional** objects such as Python lists, ndarrays can be sliced using the
familiar syntax.

**Higher dimensional objects** give you more options as you can slice one or more axes
and also mix integers.

```python
import numpy as np

arr = np.arange(10)
print(arr[1:6])

arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2d[:2])

print(arr2d[:2,1:])

print(arr2d[1,:2])
print(arr2d[2,:1])

print(arr2d[:,:1])
```

## Boolean Indexing ##

Let’s consider an example where we have some data in an array and an array of names
with duplicates. I’m going to use here the `randn` function in `numpy.random` to generate
some random normally distributed data:

**The boolean array** must be of **the same length** as the axis it’s indexing.

### 使用boolean indexing选取数据 ###

Selecting data from an array by **boolean indexing** always creates **a copy of the data**,
even if the returned array is unchanged.

```python
import numpy as np

names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data = np.random.randn(7,4)

print(names)
print(data)
print("-"*60)

print(names=='Bob')
print(data[names=='Bob'])
print("-"*60)

print(data[names=='Bob',2:])
print("-"*60)
print(data[names=='Bob',3])
print("-"*60)
print(data[names=='Bob',3:])

print(names != 'Bob') #To select everything but 'Bob', you can either use != 
print(data[(names != 'Bob')])
```

## 使用多个boolean conditions选择数据 ##

Selecting two of the three names to combine multiple boolean conditions, use boolean
arithmetic operators like **&** (and) and **|** (or):

注意：The Python keywords `and` and `or` do not work with **boolean arrays**.

```python
import numpy as np

names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data = np.random.randn(7,4)

mask = (names=='Bob') | (names=='Will')
print(mask)

print(data[mask])
```

### 用boolean arrays进行赋值 ###

Setting values with **boolean arrays** works in a common-sense way. To set all of the
negative values in data to 0 we need only do:

```python
import numpy as np

names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data = np.random.randn(7,4)
print(data)
print("-"*60)
data[data < 0] = 0
print(data)
print("-"*40)

data[names != 'Joe'] = 7
print(data)
```

## Fancy Indexing ##

**Fancy indexing** is a term adopted by NumPy to describe indexing using **integer arrays**.



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















