from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor


iris = load_iris()
X = iris.data[:, 2:]  # 花瓣长度和宽度
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

export_graphviz(
    tree_clf,
    out_file="./iris_tree.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

# ./dot -Tpng ~/PycharmProjects/mlstudy/bjsxt/iris_tree.dot -o ~/PycharmProjects/mlstudy/bjsxt/iris_tree.png

print(tree_clf.predict_proba([5, 1.5]))
print(tree_clf.predict([5, 1.5]))


tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)

