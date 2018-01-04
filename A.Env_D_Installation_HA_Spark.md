
启动hdfs集群：

	start-dfs.sh
	stop-dfs.sh
	
	http://mini1:50070

启动zookeeper：

	zkServer.sh start
	zkServer.sh status

启动Spark集群：

	./start-all.sh
	./stop-all.sh
	
	http://192.168.80.131:8080


先启动zookeeper

修改Spark配置信息[参考地址](http://spark.apache.org/docs/1.6.2/spark-standalone.html#high-availability)

cd /usr/local/spark-1.6.0/conf/ && ll
vi spark-env.sh
SPARK_DAEMON_JAVA_OPTS="-Dspark.deploy.recoveryMode=ZOOKEEPER -Dspark.deploy.zookeeper.url=Spark01:2181,Spark02:2181,Spark03:2181 -Dspark.deploy.zookeeper.dir=/zkSpark"

scp -r spark-env.sh root@Spark02:$PWD
scp -r spark-env.sh root@Spark03:$PWD

cd /usr/local/spark-1.6.0/sbin/ && ll
./start-all.sh

查看zookeeper的数据：

zkCli.sh





