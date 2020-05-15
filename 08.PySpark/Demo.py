from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Python Spark SQL basic example").config("spark.some.config.option", "some-value").getOrCreate()

df = spark.read.format('com.databricks.spark.csv').options(header='true',inferschema='true').load("/Users/caihaocui/Documents/github/Data_Science_Python/data/my_df.csv",header=True)

df.show(5) 
df.printSchema()