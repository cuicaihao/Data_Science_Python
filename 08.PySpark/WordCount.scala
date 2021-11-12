
import org.apache.spark.rdd.RDD
val rootPath: String = "."
val file: String = s"${rootPath}/wikiOfSpark.txt"
// 读取文件内容
val lineRDD: RDD[String] = spark.sparkContext.textFile(file)
// 以行为单位做分词
val wordRDD: RDD[String] = lineRDD.flatMap(line => line.split(" "))
val cleanWordRDD: RDD[String] = wordRDD.filter(word => !word.equals(""))
// 把RDD元素转换为（Key，Value）的形式
val kvRDD: RDD[(String, Int)] = cleanWordRDD.map(word => (word, 1))
// 按照单词做分组计数
val wordCounts: RDD[(String, Int)] = kvRDD.reduceByKey((x, y) => x + y)

// 打印词频最高的5个词汇
wordCounts.map{case (k, v) => (v, k)}.sortByKey(false).take(5)

// ! How to run the scala with your spark shell.
// ❯ spark-shell
// 21/09/15 08:48:06 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
// Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
// Setting default log level to "WARN".
// To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
// Spark context Web UI available at http://caihaos-mini:4040
// Spark context available as 'sc' (master = local[*], app id = local-1631659691172).
// Spark session available as 'spark'.
// Welcome to
//       ____              __
//      / __/__  ___ _____/ /__
//     _\ \/ _ \/ _ `/ __/  '_/
//    /___/ .__/\_,_/_/ /_/\_\   version 3.1.2
//       /_/
         
// Using Scala version 2.12.10 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_291)
// Type in expressions to have them evaluated.
// Type :help for more information.

// scala> :load WordCount.scala
// Loading WordCount.scala...
// import org.apache.spark.rdd.RDD
// rootPath: String = .
// file: String = ./wikiOfSpark.txt
// lineRDD: org.apache.spark.rdd.RDD[String] = ./wikiOfSpark.txt MapPartitionsRDD[1] at textFile at WordCount.scala:26
// wordRDD: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[2] at flatMap at WordCount.scala:26
// cleanWordRDD: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[3] at filter at WordCount.scala:26
// kvRDD: org.apache.spark.rdd.RDD[(String, Int)] = MapPartitionsRDD[4] at map at WordCount.scala:26
// wordCounts: org.apache.spark.rdd.RDD[(String, Int)] = ShuffledRDD[5] at reduceByKey at WordCount.scala:26
// res0: Array[(Int, String)] = Array((67,the), (63,Spark), (54,a), (51,and), (50,of))
