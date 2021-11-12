from pyspark import SparkContext

textFile = SparkContext().textFile("./wikiOfSpark.txt")
wordCount = (
    textFile.flatMap(lambda line: line.split(" "))
    .filter(lambda word: word != "")
    .map(lambda word: (word, 1))
    .reduceByKey(lambda x, y: x + y)
    .sortBy(lambda x: x[1], False)
    .take(5)
)
print(wordCount)

#   ~/Doc/G/Data_Science_Python/08.PySpark on   master !11 ?13 
# ❯ python WordCount.py 
# 21/11/12 11:50:07 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
# Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
# Setting default log level to "WARN".
# To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
# /Users/caihaocui/opt/spark-3.1.2-bin-hadoop3.2/python/lib/pyspark.zip/pyspark/shuffle.py:60: UserWarning: Please install psutil to have better support with spilling
# /Users/caihaocui/opt/spark-3.1.2-bin-hadoop3.2/python/lib/pyspark.zip/pyspark/shuffle.py:60: UserWarning: Please install psutil to have better support with spilling
# [('the', 67), ('Spark', 63), ('a', 54), ('and', 51), ('of', 50)]                

#   ~/Doc/G/Data_Science_Python/08.PySpark on   master !11 ?13 
# ❯ 