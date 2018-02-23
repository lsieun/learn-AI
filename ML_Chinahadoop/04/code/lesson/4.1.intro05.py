#coding:utf-8

import numpy as np

b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(b)
print("="*40)

#知识点：1、查看ndarry的元素类型；2、创建naarry时，指定元素类型；3、更改ndarry类型

# 数组的元素类型可以通过dtype属性获得
print(b.dtype) #int32

# 可以通过dtype参数在创建时指定元素类型
d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float)
f = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.complex)
print(d)
print(d.dtype) # float64
print("="*40)
print(f)
print(f.dtype) # complex128，这里代表“复数”
print("="*40)

# 如果更改元素类型，可以使用astype安全的转换
g = d.astype(np.int)
print(g)
print(g.dtype) # int32
print("="*40)

# 但不要强制仅修改元素类型，如下面这句，将会以int来解释单精度float类型
np.set_printoptions(linewidth=400)
d.dtype = np.int
print(d)
print(d.shape)
print(d.dtype)