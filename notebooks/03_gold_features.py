# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC ## 03 - Gold features

# COMMAND ----------
from src.features import add_time_feats_spark, select_training_cols_spark

CATALOG = "mlops_demo"
SCHEMA = "taxi"

silver = f"{CATALOG}.{SCHEMA}.silver_trips"
gold   = f"{CATALOG}.{SCHEMA}.gold_training"

df = spark.table(silver)
feat = select_training_cols_spark(add_time_feats_spark(df))
feat.write.mode("overwrite").format("delta").saveAsTable(gold)
print("Wrote:", gold)
