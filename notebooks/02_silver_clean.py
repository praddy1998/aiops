# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC ## 02 - Clean to Silver

# COMMAND ----------
from pyspark.sql.functions import col

CATALOG = "mlops_demo"
SCHEMA = "taxi"

bronze = f"{CATALOG}.{SCHEMA}.bronze_trips"
silver = f"{CATALOG}.{SCHEMA}.silver_trips"

df = spark.table(bronze)

clean = (
  df
  .filter((col("trip_distance")>0) & (col("fare_amount")>0))
  .filter(col("label").between(0, 500))
  .dropna(subset=["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"])
)

clean.write.mode("overwrite").format("delta").saveAsTable(silver)
print("Wrote:", silver)
