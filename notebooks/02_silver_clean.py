# Databricks notebook source
# MAGIC %md
# MAGIC ## 02 - Clean to Silver

# COMMAND ----------

# 02_silver_clean.py (robust to schema differences)
from pyspark.sql.functions import col

DB = "taxi_mini"
df = spark.table(f"{DB}.bronze_trips")

# --- Ensure a 'label' column exists (handles different source schemas) ---
if "label" not in df.columns:
    if "total_amount" in df.columns:
        df = df.withColumn("label", col("total_amount"))
    elif "fare_amount" in df.columns:
        df = df.withColumn("label", col("fare_amount"))
    else:
        raise ValueError("Neither 'label', 'total_amount', nor 'fare_amount' present in bronze_trips.")

# Pick whichever timestamp column exists
if "tpep_pickup_datetime" in df.columns:
    ts_col = "tpep_pickup_datetime"
elif "pickup_datetime" in df.columns:
    ts_col = "pickup_datetime"
else:
    ts_col = None  # timestamp not strictly required for Silver; we won't dropna on it if missing

# Build a dropna list only with columns that actually exist
nonnull_candidates = [
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    ts_col
]
nonnull_cols = [c for c in nonnull_candidates if c and c in df.columns]

# --- Basic cleaning ---
clean = (
    df
    .filter(col("trip_distance") > 0)
    .filter(col("label") > 0)
    .filter(col("label").between(0, 500))
)

if nonnull_cols:
    clean = clean.dropna(subset=nonnull_cols)

(
    clean.write.mode("overwrite").format("delta")
         .saveAsTable(f"{DB}.silver_trips")
)

print("Wrote:", f"{DB}.silver_trips")

