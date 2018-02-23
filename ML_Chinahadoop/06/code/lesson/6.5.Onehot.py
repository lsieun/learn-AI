# coding:utf-8

import pandas as pd

if __name__ == '__main__':
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse=False)
    x = [[1, 2, 1],
         [1, 2, 0],
         [2, 0, 2],
         [0, 2, 2]]
    x_onehot = ohe.fit_transform(x)
    print(x_onehot)
