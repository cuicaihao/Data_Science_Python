from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import findspark

findspark.init()
spark = SparkSession.builder.master("local[*]").getOrCreate()

# Step 1: Read the data
dataset = spark.read.csv("BostonHousing.csv", inferSchema=True, header=True)
dataset.printSchema()

# Step 2: Set I/O sample pair
# Input all the features in one vector column
assembler = VectorAssembler(
    inputCols=[
        "crim",
        "zn",
        "indus",
        "chas",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "b",
        "lstat",
    ],
    outputCol="Attributes",
)

output = assembler.transform(dataset)

# Input vs Output
finalized_data = output.select("Attributes", "medv")

finalized_data.show()

# Step 3: Train/Test Split
# Split training and testing data
train_data, test_data = finalized_data.randomSplit([0.8, 0.2])


# Step 4: Model Development
regressor = LinearRegression(featuresCol="Attributes", labelCol="medv")

# Learn to fit the model from training set
regressor = regressor.fit(train_data)

# Step 5: Make prediction on Test data
# To predict the prices on testing set
pred = regressor.evaluate(test_data)

# Predict the model
pred.predictions.show()


# Step 6: Model Performance Analysis

# coefficient of the regression model
coeff = regressor.coefficients

# X and Y intercept
intr = regressor.intercept

print("The coefficient of the model is : %a" % coeff)
print("The Intercept of the model is : %f" % intr)


# Compute Metrics
eval = RegressionEvaluator(
    labelCol="medv", predictionCol="prediction", metricName="rmse"
)

# Root Mean Square Error
rmse = eval.evaluate(pred.predictions)
print("RMSE: %.3f" % rmse)

# Mean Square Error
mse = eval.evaluate(pred.predictions, {eval.metricName: "mse"})
print("MSE: %.3f" % mse)

# Mean Absolute Error
mae = eval.evaluate(pred.predictions, {eval.metricName: "mae"})
print("MAE: %.3f" % mae)

# r2 - coefficient of determination
r2 = eval.evaluate(pred.predictions, {eval.metricName: "r2"})
print("r2: %.3f" % r2)
