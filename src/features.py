from pyspark.sql.functions import hour, dayofweek, col

_FEATURES = ["trip_distance","passenger_count","pickup_hour","pickup_dow"]

def add_time_feats_spark(df, ts_col: str = "tpep_pickup_datetime"):
    """Add time-based features (Spark DataFrame)."""
    return (df
            .withColumn("pickup_hour", hour(col(ts_col)))
            .withColumn("pickup_dow", dayofweek(col(ts_col))))

def select_training_cols_columns():
    """Return the list of training columns including label."""
    return _FEATURES + ["label"]

def select_training_cols_spark(df):
    """Select training columns on a Spark DataFrame."""
    return df.select(*(select_training_cols_columns()))
