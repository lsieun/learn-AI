# Pandas读取数据 #

## 读取iris数据 ##

```python
import pandas as pd

if __name__ == "__main__":
    path = ".\\data\\iris.data"
    data = pd.read_csv(path,header=None)
    print(data)
```

输出如下：

	       0    1    2    3               4
	0    5.1  3.5  1.4  0.2     Iris-setosa
	1    4.9  3.0  1.4  0.2     Iris-setosa
	..   ...  ...  ...  ...             ...
	147  6.5  3.0  5.2  2.0  Iris-virginica
	148  6.2  3.4  5.4  2.3  Iris-virginica
	149  5.9  3.0  5.1  1.8  Iris-virginica
	
	[150 rows x 5 columns]


## 读取iris的前两列数据 ##

```python
import pandas as pd

if __name__ == "__main__":
    path = ".\\data\\iris.data"
    data = pd.read_csv(path,header=None)
    x = data[[0,1]]  # 此处是读取前两列数据
    print(x)
```

输出如下：

	       0    1
	0    5.1  3.5
	1    4.9  3.0
	2    4.7  3.2
	..   ...  ...
	147  6.5  3.0
	148  6.2  3.4
	149  5.9  3.0
	
	[150 rows x 2 columns]

## 将iris的类别数据进行编码 ##

```python
import pandas as pd

if __name__ == "__main__":
    path = ".\\data\\iris.data"
    data = pd.read_csv(path,header=None)
    y = pd.Categorical(data[4]).codes  #对Iris-setosa、Iris Versicolour、Iris-virginica进行编码
    print(y)
```

输出如下：

	[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
	 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
	 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
	 2 2]










