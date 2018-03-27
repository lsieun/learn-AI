# 使用Python的pip命令安装工具类库 #

## 1、使用pip在线安装 ##

[pypi 镜像使用帮助](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)

pip安装numpy

	pip install numpy

pip安装pandas

	pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas

pip安装sklearn

	pip install -U scikit-learn

pip安装scipy(注：sklearn 依赖scipy包)

	pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy

pip安装tensorflow

	pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.1.0

pip安装keras

	pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade keras

## 2、使用pip离线安装 ##

下载python的工具包地址：[Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/)

pip安装*.whl文件

	pip install opencv_python‑3.4.1+contrib‑cp35‑cp35m‑win_amd64.whl

## 3、Eclipse安装PyDev ##


MyEclipse安装PyDev

打开help-> install from catalog

![](images/pydev/20180125182235.png)

![](images/pydev/20180125182305.png)

![](images/pydev/20180125182330.png)

https://www.cnblogs.com/jym-sunshine/p/4924530.html

http://blog.csdn.net/jielysong117/article/details/39052147

http://scikit-learn.org/stable/

[Eclipse 安装python后pydev不出现](https://www.cnblogs.com/MazeHong/p/7225087.html)

[Python在Myeclipse上配置(解决Pydev插件不出现和安装标准库的方法)](http://blog.csdn.net/danielntz/article/details/51429686)

[MyEclipse10中配置开发Python所需要的PyDev 绝对靠谱 不忽悠！](https://www.cnblogs.com/simith/p/5090716.html)


# An introduction to machine learning with scikit-learn #

[An introduction to machine learning with scikit-learn](http://scikit-learn.org/stable/tutorial/basic/tutorial.html)

n samples --> predict properties of unknown data
each sample --> features

learning problem:

	supervised learning
	unsupervised learning

Loading an example dataset 加载数据集

Learning and predicting 对“已知数据集”进行学习，生成模型，和 使用模型，对“未知数据”进行预测

Model persistence 对模型进行持久化

> 机器学习里经常出现ground truth这个词，表示“真实的值”
> pickle 是 Python 内部的一种序列化方式


[scikit-learn Tutorials](http://scikit-learn.org/stable/tutorial/index.html)



