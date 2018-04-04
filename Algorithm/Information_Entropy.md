# 熵Entropy #

示例代码：

```python
import numpy as np

# 信息量：通过“概率”计算得到
def getAmountOfInformation(p):
    if p < 1.0e-8 or (1-p) < 1.0e-8:
        return 0
    return -1 * np.log2(p)

# 熵：通过多个“信息量”加权平均得到
def getEntropy(num_list):
    total_num = 0
    for num in num_list:
        total_num += num

    entropy_sum = 0.0
    for num in num_list:
        prob = 1.0 * num / total_num
        entropy_sum += prob * getAmountOfInformation(prob)

    return entropy_sum

if __name__ == '__main__':
    total_entropy = getEntropy([3,28,36])
    print('total_entropy = ', total_entropy)
```


