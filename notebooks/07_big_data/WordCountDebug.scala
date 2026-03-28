import org.apache.spark.rdd.RDD
val rootPath: String = "."
val file: String = s"${rootPath}/wikiOfSpark.txt"
// 读取文件内容
val lineRDD: RDD[String] = spark.sparkContext.textFile(file)
// 以行为单位提取相邻单词
val wordPairRDD: RDD[String] = lineRDD.flatMap(line => {
    val words: Array[String] = line.split(" ")
    for (i<-0 until words.length -1) yield words(i)+"-"+words(i+1)
})
// 保证是两个单词 start <========= 
def filterLength(s: String): Boolean = {
    val words = s.split("-")
    return words.length==2
}
val wordsPairRDD: RDD[String] = wordPairRDD.filter(filterLength)
// 保证是两个单词 end <========== 注： wordsPairRDD 比wordPairRDD多个s

// 定义特殊字符列表 <========= 
val charlist: List[String] = List("&", "|", "#", "^", "@", "\"", "-", "") // 添加了", -, "" 消除 （-"Spark）（"-Spark）("-Spark)之类的词对
// 定义判定函数f
def f(s: String): Boolean = {
    val words: Array[String] = s.split("-")
    val b1: Boolean = charlist.contains(words(0)) // special words 1
    val b2: Boolean = charlist.contains(words(1)) // 
    return !b1 && !b2 
}  
val cleanedPairRDD: RDD[String] = wordsPairRDD.filter(f)
// 把RDD元素转换为（Key，Value）的形式
val kvRDD: RDD[(String, Int)] = cleanedPairRDD.map(word => (word, 1))
// 按照单词做分组计数
val wordCounts: RDD[(String, Int)] = kvRDD.reduceByKey((x, y) => x + y)
// 打印词频最高的5个词汇
wordCounts.map{case (k, v) => (v, k)}.sortByKey(false).take(20)