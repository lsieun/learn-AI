
1、Spark是什么？

Spark是一个分布式计算框架。

2、Spark的JVM进程：

（1）主节点叫Master，是资源管理的主节点，
（2）节点叫Worker，是资源管理的从节点，
（3）client上一个节点叫Driver，用于任务分发。

3、搭建Spark集群需要修改的配置的文件：【lsieun】实际上就主要配置了spark-env.sh文件

slaves:

	Spark02
	Spark03

spark-env.sh：

	SPARK_MASTER_IP=Spark01   #這里Master的主機名
	SPARK_MASTER_PORT=7077    #属于资源类型的通信
	SPARK_WORKER_CORES=1      #应该将25%的core数留给操作系统用
	SPARK_WORKER_MEMORY=1G
	SPARK_WORKER_INSTANCES=1  #在每一个服务器上启动多少个Worker进程
	SPARK_MASTER_WEBUI_PORT=8888

注意：在搭建Spark集群的时候，不能将服务器的全部core用于Spark上，而是应该将25%留给操作系统用。

Spark的默认端口是7077，是属于“资源类型”的通信，例如提交一个新的Application，需要通过7077端口来请求资源；当Worker启动起来之后，同样需要通过7077端口向Master注册。

4、Spark的算子

Spark的算子，分为两种类型：transformation算子（延迟执行）和action算子（立即执行）。

	map         #计算一条，输出一条
	flatmap     #计算一条，可能输出N条结果
	filter      #过滤算子，有可能输入一条，不输出任何结果
	reducebyKey #reduceByKey比groupByKey更高级一点，在reduceByKey的底层就包含了groupByKey
	countByKey  #先对key进行分组，然后再对每组的数据进行count
	groupByKey  #groupByKey对key进行分组；而reduceByKey是分完组后，对key进行聚合；
	sortByKey   #sortByKey只能根据key来进行排序
	sortBy      # sortBy可以由用户指定根据哪个字段进行排序
	
	mapPartitions  #与map算子的区别是，map算子的对象是一条记录，而mapPartitions的的对象是partition。mapPartions的效率比较高，因为它会将一个partition的数据先写到内存中去，然后再进行遍历。如果要将一个RDD的数据写入到MySQL数据库时，应该选择map算子，还是mapPartitions算子呢？应该选择mapPartitions算子，因为它遍历的是partition，遍历的次数比较少，创建MySQL的数据库连接就变少了。

	foreach    #与map算子的区别是，map算子是transformation算子（懒执行），而foreach是action算子（立即执行）
	foreachPartition # foreach和foreachPartition都是action算子，但运算的对象不一样，foreach算子的运算对象是每一条记录，而foreachPartition算子的运算对象是partition。 mapPartition和foreachParition分别在什么情况下使用呢？首先，说一下两者的区别，mapPartition是transformation类型的算子，而foreachParitions算子是action算子；接下来呢，要看是否需要结束或是否需要继续运算，如果结果，就不需要产生新的RDD，这个时候，使用foreachPartition就可以了；如果需要继续运行，就需要新的RDD，这个时候，需要使用mapParition，因为transformation类型的算子一定会返回一个RDD。

	collect算子，会将每一个结果拉回到Driver进程中（回顾，Driver是用于任务的分发 和 执行结果的拉回），这可能会导致Socket通信异常，其中一种解决方法就是提高日志的打印等级（ERROR）。
	saveAsTextFile算子，并不会将结果拉回到Driver进程中
	count算子，统计每一个task的计算结果，然后将“记录数”拉回到Driver端。注意：这里是拉回的“计录数”，而不是全部“记录”，因此风险不大，不会导致OOM(Out of Memory)异常。
	
5、RDD是不存数据的


6、RDD的5大特性

- A list of Partitions
- A function for computing each partition
- A list of dependencies on other RDDs
- Optionally, a Paritioner for key-value RDDs
- Optionally, a list of preferred locations to compute each split on

使用textFile从HDFS中读取数据（分成了2个block），生成一个RDD，
这个rdd当中包含了2个partition，这个rdd再进行filter算子，又会生成一个filterRDD，
这个filterRDD又包含2个parition。。。
我理解这个parition是不是更偏向于物理上去区别数据的存储位置，而RDD偏向于将数据作为一个整体进行处理呢？

如果是从HDFS上读取数据的话，最初的RDD中的partition的数量，是和文件的block数是一致的

RDD是逻辑上的一个整体单元，里边有好多partition。既然是逻辑上的，就意味着可以分离，就是几个机器上的block共同构成一个RDD。


（1）RDD由一系列的partition组成

（2）RDD有一系列的依赖关系，有利于容错

（3）算子，不是作用在RDD上的，而是作用在partition上的

（4）分区器是作用在key-value格式的RDD上

（5）每个RDD都有自己的最佳计算位置，这就有利于“数据的本地化”


计算向数据移动！计算程序离数据源越近越好（尽量减少数据在节点之间的移动，浪费网络浪费时间）

7、RDD之间依赖关系划分：宽依赖、窄依赖

如果父RDD与子RDD，partition之间的依赖关系是一对一的关系的话，那么这个依赖关系就是窄依赖
如果父RDD与子RDD，partition之间的依赖关系是多对一的关系的话，那么这个依赖关系就是宽依赖

一般情况下，宽依赖和shuffle是对应的，即哪个地方是宽依赖，就有很大可能发生shuffle。

哪些算子是宽依赖的算子呢？

	groupByKey
	countByKey
	reduceByKey

只要算子有分组或聚合的功能，这个算子，就会导致shuffle的产生，因为相同的key很有可能在不同的节点上。

map filter flatMap mapPartition这种类型的算子，不会导致shuffle的发生。

那么，宽、窄依赖有什么作用呢？可以进行任务（job）的划分

8、Stage的切割规则

DAG 一个无环的有向图称做有向无环图（directed acycline graph），简称DAG 图。

job的个数是与action类算子的个数是一致的。或者说，一个action算子会触发一个job，也就是说action类算子与job是一一对应的。

在Job执行之前，会根据RDD的宽、窄依赖来切割job，形成不同的Stage。由Job切割成Stage，切割是根据RDD之间的宽、窄依赖关系进行的，而切割的顺序是从后向前的，因为一个RDD只知道它的父RDD，而不知道它的子RDD。切割，是最后一个RDD开始，向前面的RDD回溯，遇到宽依赖关系的RDD就进行切割。

Stage里又可以划分task（目前的层级关系是：Job-->Stage-->task），一个task就是一个thread线程

一个stage（RDD之间都是窄依赖）内进行pipeline操作

MapReduce的计算模式：1+1=2 2+1=3
Spark的计算模式：1+1+1=3


9、任务调度

RDD Objects: 可以看成是我们写的一段代码，可以绘制成有效无环图DAG
DAGScheduler：可以看成Spark中的高层任务调度器，存活于Driver进程中的对象。它的作用是将DAG切割成stages（每个stage是由一组Task组成的）， 提交stages
TaskScheduler：是Spark的底层调度器，也是存活于Driver进程中的对象

Executor进程：是在Worker节点上启动的，在Executor进程中有一个ThreadPool线程池，每一个Task就是在线程池中运行的；因此Executor是真正的“计算”进程，而Task是“计算”层面的。
Worker进程，是“资源管理”的从节点，是“资源”层面的。

ETL，是英文 Extract-Transform-Load 的缩写，用来描述将数据从来源端经过抽取（extract）、转换（transform）、加载（load）至目的端的过程。ETL一词较常用在数据仓库，但其对象并不限于数据仓库。

10、高可用的Spark集群（解决Spark单点故障的问题）

36:00

Spark HA集群原理

Zookeeper会保留master的元数据信息（master管理的资源信息：worker个数、每一个worker管理的资源情况） 这些数据究竟是什么呢？可不可以通过zookeeper查看呢？


需要搭建hdfs、zookeeper
















