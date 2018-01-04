
cd /usr/local/software/
tar -zxvf zookeeper-3.4.6.tar.gz -C /usr/local/
cd /usr/local/ && ll
mv zookeeper-3.4.6/ zookeeper

cd /usr/local/zookeeper/ && ll
rm -rf *.xml *.txt
rm -rf contrib dist-maven docs recipes src zookeeper-3.4.6.jar.asc zookeeper-3.4.6.jar.md5 zookeeper-3.4.6.jar.sha1

cd /usr/local/zookeeper/conf/ && ll
cp zoo_sample.cfg zoo.cfg
vi zoo.cfg
dataDir=/root/apps/zookeeper/zkdata
dataLogDir=/root/apps/zookeeper/log
server.1=Spark01:2888:3888
server.2=Spark02:2888:3888
server.3=Spark03:2888:3888

cd /usr/local/ && ll
scp -r zookeeper/ root@Spark02:$PWD
scp -r zookeeper/ root@Spark03:$PWD
mkdir -p /root/apps/zookeeper/{zkdata,log}
cd /root/apps/zookeeper/zkdata/ && ll
echo 1 > myid
echo 2 > myid
echo 3 > myid

vi /etc/profile
export ZOOKEEPER_HOME=/usr/local/zookeeper
export PATH=$ZOOKEEPER_HOME/bin:$PATH

scp -r /etc/profile root@Spark02:/etc/profile
scp -r /etc/profile root@Spark03:/etc/profile

source /etc/profile
source /etc/profile
source /etc/profile






