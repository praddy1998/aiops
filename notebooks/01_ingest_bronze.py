# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC ## 01 - Ingest to Bronze (Delta)

# COMMAND ----------
from pyspark.sql.functions import col
spark.conf.set("spark.sql.shuffle.partitions", "auto")

CATALOG = "mlops_demo"
SCHEMA = "taxi"
src = "/databricks-datasets/nyctaxi/tripdata/yellow"

df = spark.read.format("parquet").load(src)

bronze = f"{CATALOG}.{SCHEMA}.bronze_trips"
(
  df
  .withColumnRenamed("total_amount","label")
  .write.mode("overwrite").format("delta").saveAsTable(bronze)
)

print("Wrote:", bronze)
