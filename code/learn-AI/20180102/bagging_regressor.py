from sklearn.linear_model import RidgeCV
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


degree = 2
ridge = RidgeCV(alphas=np.logspace(-3, 2, 20), fit_intercept=False)
ridged = Pipeline([('poly', PolynomialFeatures(degree=degree)), ('ridge', ridge)])
bagging_ridged = BaggingRegressor(ridged, n_estimators=100, max_samples=0.2)
decision_tree_reg = DecisionTreeRegressor(max_depth=5)
bagging_tree = BaggingRegressor(decision_tree_reg, n_estimators=100, max_samples=0.2)

regs = [
    ('Ridge Regressor( %d Degree)' % degree, ridged),
    ('DecisionTree Regressor', decision_tree_reg),
    ('Bagging Ridge (%d Degree)' % degree, bagging_ridged),
    ('Bagging DecisionTree Regressor', bagging_tree)
]

data = pd.read_csv("../data/insurance.csv")

x = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = data['charges']

x = x.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')
x.fillna(0, inplace=True)
y.fillna(0, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=42)


for i, (name, reg) in enumerate(regs):
    reg.fit(x_train, y_train)
    y_predict = reg.predict(x_test)
    print(np.sqrt(mean_squared_error(y_test, y_pred=y_predict)))



