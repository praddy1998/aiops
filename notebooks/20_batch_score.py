# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC ## 20 - Batch scoring to predictions table

# COMMAND ----------
import mlflow
from pyspark.sql.functions import monotonically_increasing_id

CATALOG = "mlops_demo"
SCHEMA = "taxi"
MODEL_NAME = "taxi_fare_rf"
STAGE = "Production"   # use 'Staging' until you're ready

model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{MODEL_NAME}/{STAGE}")

gold = spark.table(f"{CATALOG}.{SCHEMA}.gold_training")
pred = (gold.withColumn("prediction", model(*gold.drop("label").columns))
             .withColumn("row_id", monotonically_increasing_id()))

predictions_tbl = f"{CATALOG}.{SCHEMA}.batch_predictions"
pred.write.mode("overwrite").format("delta").saveAsTable(predictions_tbl)
print("Wrote:", predictions_tbl)
