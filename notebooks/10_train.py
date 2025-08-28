# Databricks notebook source
# MAGIC %md
# MAGIC ## 10 - Train and log to MLflow + Register

# COMMAND ----------

# 10_train.py (fixed experiment path)
import mlflow, mlflow.sklearn
from pyspark.sql.functions import rand
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

DB = "taxi_mini"
mlflow.set_registry_uri("databricks")
# Use a per-user experiment path that always exists
user = spark.sql("SELECT current_user()").first()[0]
exp_path = f"/Users/{user}/taxi_fare_mini"
mlflow.set_experiment(exp_path)

pdf = (spark.table(f"{DB}.gold_training")
       .orderBy(rand())
       .limit(30_000)  # tiny slice for speed
       .toPandas())

X = pdf.drop(columns=["label"])
y = pdf["label"]

with mlflow.start_run():
    params = {"n_estimators": 120, "random_state": 42}
    model = RandomForestRegressor(**params).fit(X, y)
    pred  = model.predict(X)
    rmse  = mean_squared_error(y, pred)

    mlflow.log_params(params)
    mlflow.log_metric("rmse_train", rmse)
    mlflow.sklearn.log_model(model, "model", input_example=X.head(3))
    run_id = mlflow.active_run().info.run_id

result = mlflow.register_model(f"runs:/{run_id}/model", "taxi_fare_rf_mini")
print("Registered model:", result.name, "version:", result.version)


# COMMAND ----------


