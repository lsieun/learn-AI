# 聚类 #

注意一点：k近邻 和 k-means算法是不同的，只是名字上有点像而已。k近邻属于监督学习，而k-means是非监督学习。

## 重要的问题 ##

k如何选择最优？ 首先依赖先验知识，其次，elbow method，

Kmeans的初始点怎么选？ 随机的、经验的、

k-means初值敏感吗？敏感，选择不同的初始值，后期的结果可能差异很大。

k-means要求数据正态吗？要求。

kmeans迭代停止的条件是什么？ 迭代次数、簇中心变化率、最小平方误差MSE（Minimum Squared Error）

有没有不用告诉聚几类，算法自适应聚类呢？有，比如密度聚类。

## 算法 ##

k-means算法

k-Mediods聚类（K中值聚类）  K中值是混合拉普拉斯分布



