# Databricks notebook source
# MAGIC %md
# MAGIC ## 20 - Batch scoring to predictions table

# COMMAND ----------

# 20_batch_score.py (Workspace Model Registry fix)
import mlflow
from pyspark.sql.functions import monotonically_increasing_id

# Force Workspace (legacy) Model Registry so stage-based loading works
mlflow.set_registry_uri("databricks")

DB = "taxi_mini"
MODEL_NAME = "taxi_fare_rf_mini"
STAGE = "Staging"  # or "Production" if you promoted it
client = mlflow.tracking.MlflowClient()
latest_version = client.get_latest_versions(MODEL_NAME, stages=[])[0].version
model = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{MODEL_NAME}/{latest_version}"
)

gold = spark.table(f"{DB}.gold_training")
pred = (gold.withColumn("prediction", model(*gold.drop("label").columns))
             .withColumn("row_id", monotonically_increasing_id()))

(pred.write.mode("overwrite").format("delta")
     .saveAsTable(f"{DB}.batch_predictions"))

print("Wrote:", f"{DB}.batch_predictions")

# --- If you are actually on Unity Catalog Model Registry (UC) ---
# Comment out the set_registry_uri("databricks") above and use aliases instead of stages:
# mlflow.set_registry_uri("databricks-uc")
# CATALOG, SCHEMA = "main", "ml_models"  # example values
# MODEL_FQN = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
# ALIAS = "prod"  # or "staging" after you create the alias in the UC registry
# model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{MODEL_FQN}@{ALIAS}")


# COMMAND ----------


