import pandas as pd

s = pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
# print(s)
# print(type(s))        #输出：<class 'pandas.core.series.Series'>

#读取单个元素：通过脚标（0）和index（'a'）
# print(s[0])         #输出：1
# print(s['a'])       #输出：1
# print(type(s[0]))   #输出：<class 'numpy.int64'>，这里应该是指单个元素的类型

#读取多个元素：通过“切片”来获取其中一部分
# print(s[:3])      #这是通过“脚标”来切片
# print(s[-3:])
# print(s['a':'c']) #这种是通过index来进行，切片结果包括'a'、'b'和'c'
# print(type(s[:3]))    #输出：<class 'pandas.core.series.Series'>，说明切片之后，原来的类型不变

#读取多个元素：通过多个index
# print(s['a','c','e']) #这是一种错误写法
# print(s[['a','c','e']])  #正确写法
# print(type(s[['a','c','e']])) #输出：<class 'pandas.core.series.Series'>
