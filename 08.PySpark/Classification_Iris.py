
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import findspark
findspark.init()
spark = SparkSession.builder.master("local[*]").getOrCreate()


# Step 1: Read data

# Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
struct = StructType([
    StructField('Id', IntegerType(), True),
    StructField('SepalLengthCm', DoubleType(), True),
    StructField('SepalWidthCm', DoubleType(), True),
    StructField('PetalLengthCm', DoubleType(), True),
    StructField('PetalWidthCm', DoubleType(), True),
    StructField('Species', StringType(), True)
])


df_iris = spark.read.csv('iris.csv', header=True, schema=struct)
df_iris.printSchema()
df_iris.show(5)


# Step 2: I/O Sample Pairing
vecAssembler = VectorAssembler(inputCols=[
    "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], outputCol="features")
df_features = vecAssembler.transform(df_iris)

strIndexer = StringIndexer(inputCol="Species", outputCol="label")
df_features_label = strIndexer.fit(df_features).transform(df_features)

df_features_label.show(5)

# Step 3: Train/Test Split

df_train, df_test = df_features_label.randomSplit([.8, .2])

print(f"{df_train.count()=}, {df_test.count()=}")

# Step 4: Model Development: Classifier
# Bayes
nb = NaiveBayes(featuresCol="features", labelCol="label")
nb_model = nb.fit(df_train)


# Step 5: Test Model
df_predict = nb_model.transform(df_test.select("features", "label"))
df_predict.show()


# Step 6: Model Evaluation Analysis
evaluator = MulticlassClassificationEvaluator()
Acc = evaluator.evaluate(df_predict)
print(f"Test {Acc=}")
