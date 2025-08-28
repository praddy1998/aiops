# Databricks notebook source
# MAGIC %md
# MAGIC ## 01 - Ingest to Bronze (Delta)

# COMMAND ----------

# MAGIC %pip install -r /Workspace/Users/pradhumn.agrahari@optym.com/aiops1/aiops/requirements.txt

# COMMAND ----------

# MAGIC  %pip install mlflow scikit-learn
# MAGIC
# MAGIC  dbutils.library.restartPython()

# COMMAND ----------

# 01_ingest_bronze.py
from pyspark.sql.functions import col

DB = "taxi_mini"
SRC = "/databricks-datasets/nyctaxi-with-zipcodes/subsampled"  # Parquet

df = spark.read.format('delta').load(SRC).limit(50_000)

# Label = total_amount (simplifies training)
(df.withColumnRenamed("total_amount", "label")
   .write.mode("overwrite").format("delta")
   .saveAsTable(f"{DB}.bronze_trips"))

print("Wrote:", f"{DB}.bronze_trips")


# COMMAND ----------

# 02_silver_clean.py
from pyspark.sql.functions import col

DB = "taxi_mini"
df = spark.table(f"{DB}.bronze_trips")

clean = (
  df
  .filter((col("trip_distance") > 0) & (col("fare_amount") > 0))
  .filter(col("label").between(0, 500))
  .dropna(subset=[
      "pickup_longitude","pickup_latitude",
      "dropoff_longitude","dropoff_latitude",
      "pickup_datetime"  # this dataset uses pickup_datetime
  ])
)

(clean.write.mode("overwrite").format("delta")
      .saveAsTable(f"{DB}.silver_trips"))

print("Wrote:", f"{DB}.silver_trips")

