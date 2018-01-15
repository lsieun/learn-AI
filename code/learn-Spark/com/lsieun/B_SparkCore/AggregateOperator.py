#--encoding:utf-8--
'''
Created on 2017年10月20日

@author: root
'''
from pyspark import SparkConf,SparkContext
from __builtin__ import str
conf = SparkConf().setMaster("local").setAppName("AggregateByKey")
sc = SparkContext(conf = conf)

rdd = sc.parallelize(range(1,9),2)

def f(index,items):
    print "partitionId:%d" %index
    for val in items:
        print val
    return items
    
rdd.mapPartitionsWithIndex(f, False).count()


def seqFunc(a,b):
    print "seqFunc:%s,%s" %(a,b)
    return a + str(b) + "~"
def combFunc(a,b):
    print "combFunc:%s,%s" %(a ,b)
    return a + b

rest = rdd.aggregate("", seqFunc, combFunc)
print rest
sc.stop()
