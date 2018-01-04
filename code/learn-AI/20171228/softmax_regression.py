#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: softmax_regression.py

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

__author__ = 'yasaka'

iris = datasets.load_iris()
print(iris['DESCR'])
print(iris['feature_names'])
X = iris['data'][:, (2, 3)]
print(X)
y = iris['target']

softmax_reg = LogisticRegression(multi_class='multinomial', solver='sag', C=8, max_iter=1000)
softmax_reg.fit(X, y)
print(softmax_reg.predict([[5, 2]]))
print(softmax_reg.predict_proba([[5, 2]]))









