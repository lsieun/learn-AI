克隆的虚拟机需要修改三个地方：

	#（1）主机名：
	vi /etc/sysconfig/network
	#（2）网卡地址：
	vi /etc/udev/rules.d/70-persistent-net.rules
	#（3）IP地址：
	vi /etc/sysconfig/network-scripts/ifcfg-eth0
	#重启
	shutdown -r now


配置mini1的hosts文件、

	vi /etc/hosts
	192.168.80.31 mini1
	192.168.80.32 mini2
	192.168.80.33 mini3

配置免密登录

	ssh-keygen
	ssh-copy-id mini1
	ssh-copy-id mini2 
	ssh-copy-id mini3

同步mini1的hosts文件到mini2和mini3

	scp /etc/hosts root@mini2:/etc/hosts
	scp /etc/hosts root@mini3:/etc/hosts


安装JDK

	#上传JDK1.8
	mkdir -p /usr/local/software
	cd /usr/local/software/
	rz -y jdk-8u131-linux-x64.tar.gz
	
	#解压JDK1.8
	tar -zxvf jdk-8u131-linux-x64.tar.gz -C /usr/local/
	cd /usr/local/ && ll
	mv jdk1.8.0_131/ jdk1.8.0
	cd jdk1.8.0/ && pwd #/usr/local/jdk1.8.0
	
	#配置Java环境变量
	vi /etc/profile
	export JAVA_HOME=/usr/local/jdk1.8.0
	export PATH=$JAVA_HOME/bin:$PATH

	source /etc/profile
	java -version

同步mini1的JDK环境到mini2和mini3上

	#拷贝JDK到mini2和mini3
	cd /usr/local/ && ll
	scp -r jdk1.8.0/ root@mini2:$PWD
	scp -r jdk1.8.0/ root@mini3:$PWD
	
	#配置mini2和mini3的Java环境变量
	vi /etc/profile
	export JAVA_HOME=/usr/local/jdk1.8.0
	export PATH=$JAVA_HOME/bin:$PATH
	
	source /etc/profile
	java -version

上传和解压hadoop的tar.gz文件

	cd /usr/local/software/ && ll
	rz -y hadoop-2.6.4.tar.gz
	
	tar -zxvf hadoop-2.6.4.tar.gz -C /usr/local/
	cd /usr/local/ && ll
	mv hadoop-2.6.4/ hadoop
	cd /usr/local/hadoop/ && ll

修改hadoop的5个配置文件：

	cd /usr/local/hadoop/etc/hadoop/ && ll
		hadoop-env.sh
		core-site.xml
		hdfs-site.xml
		mapred-site.xml
		yarn-site.xml

	#（1）修改hadoop-env.sh
	vi hadoop-env.sh
		# The java implementation to use.
		export JAVA_HOME=/usr/local/jdk1.8.0
	#（2）修改core-site.xml
	vi core-site.xml
	<configuration>
	    <property>
	        <name>fs.defaultFS</name>
	        <value>hdfs://mini1:9000</value>
	    </property>
	    <property>
	        <name>hadoop.tmp.dir</name>
	        <value>/root/apps/hadoop/tmp</value>    <!-- TODO: 这个目录,我没有创建,看看会不会自动生成? 回答:在进行namenode格式化的时候会自动创建. -->
	    </property>
	</configuration>
	#（3）修改hdfs-site.xml
	<configuration>
	    <property>
	        <name>dfs.replication</name>
	        <value>3</value>
	    </property>
	    <property>
	        <name>dfs.secondary.http.address</name>
	        <value>mini1:50090</value>
	    </property>
	</configuration>
	#（4）修改mapred-site.xml
	cp mapred-site.xml.template mapred-site.xml
	vi mapred-site.xml
	<configuration>
	    <property>
	        <name>mapreduce.framework.name</name>
	        <value>yarn</value>
	    </property>
	</configuration>
	#（5）修改yarn-site.xml
	vi yarn-site.xml

将hadoop添加到环境变量

	vi /etc/profile
	export JAVA_HOME=/usr/local/jdk1.8.0
	export HADOOP_HOME=/usr/local/hadoop
	export PATH=$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH
	source /etc/profile

将hadoop的安装程序拷贝到mini2和mini3上

	cd /usr/local/ && ll
	scp -r /usr/local/hadoop/ root@mini2:/usr/local/
	scp -r /usr/local/hadoop/ root@mini3:/usr/local/

	scp -r /etc/profile root@mini2:/etc/profile
	scp -r /etc/profile root@mini3:/etc/profile

在mini2和mini3上重新加载环境变量

	source /etc/profile
	echo $HADOOP_HOME

在启动hadoop之前要进行namenode的格式化操作

	hadoop namenode -format

	cd /root/apps/hadoop/tmp/dfs/name && ll
	cd current/ && ll
	cat VERSION

启动和关闭HDFS集群的方式：关闭mini1、mini2、mini3的防火墙service iptables stop

	start-dfs.sh
	stop-dfs.sh

使用start-dfs.sh和stop-dfs.sh的前提是：在${HADOOP_HOME}/etc/hadoop目录下的slaves文件中配置datanode的地址，并将slaves拷贝到mini2和mini3上

	cd /usr/local/hadoop/etc/hadoop/ && ll
	[root@mini1 hadoop]# vi slaves
	mini1
	mini2
	mini3
	[root@mini1 hadoop]# scp -r slaves root@mini2:$PWD
	[root@mini1 hadoop]# scp -r slaves root@mini3:$PWD

此时,打开浏览器,

	http://mini1:50070

如果打不开,查看是否已经关闭的防火墙.

通过Hadoop Shell命令上传一张图片并查看：

	hdfs dfs -put logo.jpg /
	hdfs dfs -ls /

也可以在浏览器中进行查看文件。

查看hadoop的版本方法是

	hadoop version

> 至此结束

