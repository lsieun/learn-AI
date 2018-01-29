from sklearn import linear_model

reg = linear_model.Lasso(alpha=0.1)
print(reg)
reg.fit([[0, 0], [1, 1]], [0, 1])
predict = reg.predict([[1, 1]])
print(predict)