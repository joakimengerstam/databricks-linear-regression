from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, when
from pyspark.sql.functions import hour, dayofweek, unix_timestamp

# Use Spark ML Linear regression on NYC taxi dataset to train the model

# Load CSV data
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/Volumes/workspace/default/nyctaxi/gkne-dk5s.csv")

# Display the schema to see available columns
print("Dataset Schema:")
df.printSchema()
print(f"\nTotal rows: {df.count()}")

# Show sample data
print("\nSample data:")
display(df.limit(10))

# Keep only realistic trips
df_clean = df.filter((df.trip_distance > 0) & (df.fare_amount > 1))

# Extract hour from timestamp
df_with_hour = df_clean.withColumn('pickup_hour', hour(col('pickup_datetime')))

# Create rush hour indicator (8-9 AM, 5-6 PM)
df_with_features = df_with_hour.withColumn('is_rush_hour',
    when((col('pickup_hour').between(8, 9)) |
         (col('pickup_hour').between(17, 18)), 1)
    .otherwise(0)
)

# Calculate trip duration in minutes
df_with_duration = df_with_features.withColumn('trip_duration_minutes',
                                 (unix_timestamp('dropoff_datetime') - unix_timestamp('pickup_datetime')) / 60
                                 )

#df_with_duration = df_with_duration.filter(col('trip_duration_minutes') > 0.01)

df_with_duration = df_with_duration.filter(
    (col('fare_amount') / col('trip_distance') < 60) &
    (col('trip_distance') > 0.9)
)

# Data cleaning - remove nulls and invalid values
df_clean = df_with_duration.dropna(
    subset=['trip_distance', 'trip_duration_minutes', 'fare_amount'])


feature_columns = ['trip_distance', 'is_rush_hour']
target_column = 'fare_amount'  # Your target variable

# Select only the columns we need
df_clean = df_clean.select(feature_columns + [target_column])

# Save as Delta table in default schema
# (overwrite existing schema)
df_clean.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("newtable")

# Verify table
df_delta = spark.table("newtable")
df_delta.printSchema()

df_clean = df_delta.select(feature_columns + [target_column])

# Create feature vector required by Spark ML
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol='features'
)
df_vectorized = assembler.transform(df_clean)

# Split data into training and test sets (80-20 split)
train_data, test_data = df_vectorized.randomSplit([0.8, 0.2], seed=42)

print(f"\nTraining set size: {train_data.count()}")
print(f"Test set size: {test_data.count()}")

# Create and train the Linear Regression model
lr = LinearRegression(
    featuresCol='features',
    labelCol=target_column,
    predictionCol='prediction',
    maxIter=10,
    regParam=0.3,
    elasticNetParam=0.8
)

print("\nTraining the model...")
lr_model = lr.fit(train_data)

# Print model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {lr_model.intercept}")
print(f"Coefficients: {lr_model.coefficients}")

# Make predictions on test data
predictions = lr_model.transform(test_data)

# Show sample predictions
print("\nSample Predictions:")
predictions.select(target_column, 'prediction', 'features').show(10)

# Evaluate the model
evaluator = RegressionEvaluator(
    labelCol=target_column,
    predictionCol='prediction'
)

# Calculate metrics
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})

print("\nModel Performance Metrics:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Training summary
training_summary = lr_model.summary
print(f"\nTraining RMSE: {training_summary.rootMeanSquaredError:.2f}")
print(f"Training R2: {training_summary.r2:.4f}")

# Optional: Save the model
# model_path = "/Volumes/workspace/default/nyctaxi/lr_model"
# lr_model.save(model_path)
# print(f"\nModel saved to: {model_path}")

# Optional: Save predictions to Delta table
# predictions_path = "/Volumes/workspace/default/nyctaxi/predictions"
# predictions.write.format("delta").mode("overwrite").save(predictions_path)
# print(f"Predictions saved to: {predictions_path}")

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LinearRegression as SklearnLR

# ------------------------------------------------------------
# 4. Convert to a pure scikit-learn model (no Spark needed)
# ------------------------------------------------------------
coef = np.array(lr_model.coefficients)
intercept = float(lr_model.intercept)

sk_model = SklearnLR()
sk_model.coef_ = coef
sk_model.intercept_ = intercept
sk_model.feature_names_in_ = np.array(feature_columns)

# ------------------------------------------------------------
# 5. Prepare example data for signature inference
# ------------------------------------------------------------
example_input = test_data.select("trip_distance", "is_rush_hour").limit(5).toPandas()
example_output = sk_model.predict(example_input)
signature = infer_signature(example_input, example_output)


# ------------------------------------------------------------
# 6. Log and register the model with MLflow
# ------------------------------------------------------------
mlflow.set_experiment("/Shared/nyc_taxi_demo")

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(
        sk_model,
        artifact_path="nyc_taxi_lr_model",
        signature=signature,
        registered_model_name="default.nyc_taxi_lr_model"
    )

    mlflow.log_params({
        "features": feature_columns,
        "training_rows": train_data.count()
    })
    mlflow.log_metrics({
        "r2": lr_model.summary.r2,
        "rmse": lr_model.summary.rootMeanSquaredError
    })

print("✅ Model logged and registered as 'default.nyc_taxi_lr_model'")
print("You can now serve it via Databricks Model Serving REST API.")

