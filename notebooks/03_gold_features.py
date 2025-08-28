# Databricks notebook source
# MAGIC %md
# MAGIC ## 03 - Gold features

# COMMAND ----------

# 03_gold_features.py (auto-detects timestamp + schema-safe)
from pyspark.sql.functions import hour, dayofweek, col, lit

DB = "taxi_mini"
df = spark.table(f"{DB}.silver_trips")

# --- Pick the correct timestamp column if present ---
if "tpep_pickup_datetime" in df.columns:
    ts_col = "tpep_pickup_datetime"
elif "pickup_datetime" in df.columns:
    ts_col = "pickup_datetime"
else:
    ts_col = None  # not fatal; we'll fall back to constants

# Add time features (or defaults if no timestamp exists)
if ts_col:
    df = df.withColumn("pickup_hour", hour(col(ts_col))) \
           .withColumn("pickup_dow",  dayofweek(col(ts_col)))
else:
    df = df.withColumn("pickup_hour", lit(0)) \
           .withColumn("pickup_dow",  lit(0))

# Ensure label exists (handles different source schemas)
if "label" not in df.columns:
    if "total_amount" in df.columns:
        df = df.withColumn("label", col("total_amount"))
    elif "fare_amount" in df.columns:
        df = df.withColumn("label", col("fare_amount"))
    else:
        raise ValueError("Missing 'label' (or 'total_amount'/'fare_amount') in silver_trips.")

# Ensure passenger_count exists (rarely missing in some samples)
if "passenger_count" not in df.columns:
    df = df.withColumn("passenger_count", lit(1))

# Validate required columns
feat_cols = ["trip_distance","passenger_count","pickup_hour","pickup_dow","label"]
missing = [c for c in feat_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns in silver_trips: {missing}")

feat = df.select(*feat_cols)

(feat.write.mode("overwrite").format("delta")
     .saveAsTable(f"{DB}.gold_training"))

print("Wrote:", f"{DB}.gold_training")

