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

