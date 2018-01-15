from pyspark import SparkConf,SparkContext
from __builtin__ import str
conf = SparkConf().setMaster("local").setAppName("AggregateByKey")
sc = SparkContext(conf = conf)

'''
def parallelize(self, c, numSlices=None)
Distribute a local Python collection to form an RDD.
'''

rdd = sc.parallelize(range(1,9),2)

print(type(rdd))
print(rdd)


def f(index, items):
    print "partitionId:%d" % index
    for val in items:
        print val
    return items


rdd.mapPartitionsWithIndex(f, False).count()

sc.stop()