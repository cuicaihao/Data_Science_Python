import org.apache.spark.rdd.RDD
val rootPath: String = "."
val file: String = s"${rootPath}/wikiOfSpark.txt"
// 读取文件内容
val lineRDD: RDD[String] = spark.sparkContext.textFile(file)
// Option 1: 以行为单位做分词 (A, B, C) // val wordRDD: RDD[String] = lineRDD.flatMap(line => line.split(" "))
// Option 2: 以行为单位提取相邻单词 (A-B, B-C)
val wordPairRDD: RDD[String] = lineRDD.flatMap(line => {
    val words: Array[String] = line.split(" ")
    for (i<-0 until words.length -1) yield words(i)+"-"+words(i+1)
})

val cleanedPairRDD: RDD[String] = wordPairRDD.filter(word => !word.equals("")) // 过滤空词
// 把RDD元素转换为（Key，Value）的形式
val kvRDD: RDD[(String, Int)] = cleanedPairRDD.map(word => (word, 1))
// 按照单词做分组计数
val wordCounts: RDD[(String, Int)] = kvRDD.reduceByKey((x, y) => x + y)
// 打印词频最高的5个词汇
wordCounts.map{case (k, v) => (v, k)}.sortByKey(false).take(20)
